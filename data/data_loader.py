import numpy as np
import os
import cv2
from datasets import Features, Sequence, Value, Array2D, Array3D
import torch
from torch.utils.data.dataloader import default_collate


def convert_to_custom_format(original_dataset,image_dir,banned_files):
    custom_dataset = []

    count = 0

    for doc_id, document in enumerate(original_dataset):
        # File Name
        file_name = document["file_name"]

        # Load The File
        image = cv2.imread(f'/data/circulars/DATA/LayoutLM/docvqa_dataset/Images/{file_name}')
        image=cv2.resize(image,(224,224))

        # Skip if the file is in the banned files
        # Skip if the file is in the banned files
        if file_name in banned_files:
            continue

        try:
            for qa_id, qa_pair in enumerate(document["q_and_a"]):
                boxes_arr = np.array(document["boxes"])
                # Pad the boxes array to 512
                padded_boxes = np.pad(boxes_arr, ((0, 512 - len(boxes_arr)), (0, 0)), mode='constant', constant_values=0)
                # # Get the Channels first
                bbox = boxes_arr  # Placeholder for bbox
                image_tensor = torch.tensor(image).clone().detach()
                image_tensor = image_tensor.permute(2, 0, 1)

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image
                input_ids = np.array(qa_pair.get("input_ids", -1))
                # Just take the first 512 tokens
                input_ids = input_ids[:512]

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image

                start_positions = qa_pair.get("start_idx", -1)
                end_positions = qa_pair.get("end_idx", -1)

                if start_positions > 512:
                    start_positions = -1
                    continue

                if end_positions > 512:
                    end_positions = -1
                    continue

                custom_example = {
                    'input_ids': input_ids,
                    'bbox': padded_boxes,
                    'image': image_tensor,
                    'start_positions': start_positions,
                    'end_positions': end_positions,
                }

                custom_dataset.append(custom_example)
                count += 1
        except Exception as e:
            print(f"Error processing Document {doc_id}, QA {qa_id}: {str(e)}")
            count += 1
            continue

    features = {
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'start_positions': Value(dtype='int64'),
        'end_positions': Value(dtype='int64'),
    }

    return custom_dataset

def custom_collate(batch):
    elem_type = type(batch[0])

    if elem_type in (int, float):
        return torch.tensor(batch)
    elif elem_type is torch.Tensor:
        return torch.stack(batch, dim=0)
    elif elem_type is list:
        # Handle lists differently, especially sequences
        return [custom_collate(samples) for samples in zip(*batch)]
    elif elem_type is dict:
        # Handle dictionaries
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        # For other types, use the default_collate behavior
        return default_collate(batch)