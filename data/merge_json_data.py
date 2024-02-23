import os
import json

file1='/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data.json'
file2='/data/circulars/DATA/LayoutLM/docvqa_dataset/first_pages_b2_final.json'

dataset1 = json.load(open(file1))
dataset2 = json.load(open(file2))

print('file 1 length :',len(dataset1))
print('file 2 length :',len(dataset2))

dataset1.extend(dataset2)

print('output file length :',len(dataset1))

with open(os.path.join(os.path.dirname(file1),"full_data.json"), "w") as f:
    json.dump(dataset1, f, indent=4)