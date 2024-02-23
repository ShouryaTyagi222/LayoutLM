# LayoutLM

## Installation
```
pip install -r requirements.txt
```

## Data
- To prepare datafor training use `data/prepare_data.py`.
- To merge multiple json files data use `merge_json_data.py`

## Config
Update the Required configs for the model in `config.py`
- `model`: LayoutLMv2/LayoutLMv3
- `image_dir`: Path to the folder consisting of all the images
- `data_path`: Path to the final Prepared data file(full_data.json)
- `output_path`: Path to the Output folder.
- `banned_txt_path`: Path to the txt file of banned files
- `epochs`: Number of Epochs to train for
- `batch_size`: The Batch-Size
- `learning_rate`: Learning Rate
- `data_split`: Test Split from the data
- `init_checkpoint`: Initial checkpoint to the huggingface model

## Train
```
python train.py
```

## Infer
```
python infer.py -i IMAGE_INPUT -q QUESTION -m MODEL_PATH
```