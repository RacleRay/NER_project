#-*- coding:utf-8 -*-

import pandas as pd
from base_runner import BaseRunner
import numpy as np
from scipy.special import softmax

# Creating train_df  and eval_df for demonstration
train_data = [
    [0, "This", "B-MISC"],
    [0, "is", "I-MISC"],
    [0, "a", "O"],
    [0, "text", "O"],
    [1, "This", "B-MISC"],
    [1, "is", "I-MISC"],
    [1, "a", "O"],
    [1, "NER", "B-MISC"],
]
train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

eval_data = [
    [0, "This", "B-MISC"],
    [0, "is", "I-MISC"],
    [0, "a", "O"],
    [0, "text", "O"],
    [1, "This", "B-MISC"],
    [1, "is", "I-MISC"],
    [1, "a", "O"],
    [1, "NER", "B-MISC"],
]
eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

# Create a NERModel
model = BaseRunner("bert_lstm_crf", "bert-base-cased", args={"overwrite_output_dir": True, "reprocess_input_data": True})

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)

# Predictions on arbitary text strings
sentences = ["This is a sentence"]
predictions, raw_outputs = model.predict(sentences)

print(predictions)

# More detailed preditctions
for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
    print("----------------------------------------")
    print("Sentence: ", sentences[n])
    for pred, out in zip(preds, outs):
        key = list(pred.keys())[0]
        new_out = out[key]
        preds = list(softmax(np.mean(new_out, axis=0)))
        print(key, pred[key], preds[np.argmax(preds)], preds)