import os

for i in os.listdir('./'):
    if '.txt' in i:
        os.rename(i, 'textdata'+i[4:])