import os
import json

file1='/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data_v2.json'
file2='/data/circulars/DATA/LayoutLM/docvqa_dataset/processed_data_v2/middle_pages_final_v2.json'
output_file='/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data_v2.json'

dataset1 = json.load(open(file1))
dataset2 = json.load(open(file2))

print('file 1 length :',len(dataset1))
print('file 2 length :',len(dataset2))

dataset1.extend(dataset2)

print('output file length :',len(dataset1))

with open(output_file, "w") as f:
    json.dump(dataset1, f, indent=4)