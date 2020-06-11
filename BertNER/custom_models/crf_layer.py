#-*- coding:utf-8 -*-
# author: allenNLP
# project: BertNER

from typing import List, Tuple, Union
from .crf_utils import viterbi_decode, logsumexp, allowed_transitions
import torch


VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.
    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf
    Parameters
    ----------
    num_tags : int, required
        Number of tags.
    idx2tag : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Example: {0:"B-LOC", 1:"I-LOC", 2:"O"}
    label_encoding : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", "BMES" and "BIOES",.
            B = Beginning
            I/M = Inside / Middle
            L/E = Last / End
            O = Outside
            U/W/S = Unit / Whole / Single
    include_start_end_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.
    """
    def __init__(
        self,
        num_tags: int,
        label_encoding,
        idx2tag,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        # 标签转移特征函数
        constraints = allowed_transitions(label_encoding, idx2tag)
        self.num_tags = num_tags

        # 待学习便签转移权重
        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # 多了两个位置：start of sequence ， end of sequence tags.
        if constraints is None:  # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous() # [seq_len, batch_size]
        logits = logits.transpose(0, 1).contiguous()     # [seq_len, batch_size, num_tags]

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        # 计算序列得分
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            # 下一个step的每种tag输出word概率得分。来自神经网络的hidden state
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            # tag之间转移的参数矩阵，CRF的学习参数
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            # 当前step，为不同tag的历史最佳路径得分。
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            # 通过log-exp，转换为加和。为下一个step为不同tag时，所取得的最佳路径得分，
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (1 - mask[i]).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        # add the final TAG, that can be goood for decoding the best path.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.LongTensor
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        logits should be in (batch_size, sequence_length, num_tags)
        """
        assert tags.shape == mask.shape, "Mask and tags should be of same shape"
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()      # [seq_len, batch_size, num_tags]
        mask = mask.float().transpose(0, 1).contiguous()  # (seq_length, batch)
        tags = tags.transpose(0, 1).contiguous()          # (seq_length, batch)

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            # logits[i]： [batch_size, num_tags]
            # current_tag.view(batch_size, 1): [batch_size, 1]
            #  根据current_tag的index在logits中输出对应的值
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        # 最后一个标签的tag index
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked. 因为前面计算score循环只计算了sequence_length - 1的长度
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
        self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.ByteTensor = None
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        inputs: (batch, seq_length, num_tags)
        tags: (batch, seq_length)
        mask: (batch, seq_length)
        Returns log_likelihood（一个batch的得分）
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        # 所有可能序列的得分
        log_denominator = self._input_likelihood(inputs, mask)
        # 输入tag序列的得分
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
        self, logits: torch.Tensor, mask: torch.Tensor, top_k: int = None
    ) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)
        If top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).

        top_k为某个数：每个batch member返回k个(tag_sequence, viterbi_score)
        top_k is None： 每个batch member返回最好的tag_sequence
        """
        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)
        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask[:num_tags, :num_tags] \
                                  + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() * \
                                                self._constraint_mask[start_tag, :num_tags].data +\
                                                -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = self.end_transitions.detach() *\
                                              self._constraint_mask[:num_tags, end_tag].data +\
                                              -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1 : (sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_paths, viterbi_scores = viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths

    def best_viterbi_tag(self, logits, mask):
        return self.viterbi_tags(logits, mask, top_k=1)