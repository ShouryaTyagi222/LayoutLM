import os
class Config:
    def __init__(self):
        self.model='LayoutLMv3'

        self.image_dir_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/Images'
        self.data_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data_v3.json'
        self.output_path='/data/circulars/DATA/LayoutLM/model_output_2'
        self.banned_txt_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/banned_txt_files.txt'

        self.epochs=20
        self.batch_size=4
        self.learning_rate=2.5e-5
        # 2 : 0.0000125, 3 : 0.0000125 large, 1 : 0.000025, 4 : 0.00005, 5 : 0.000005, 6 : 0.00000125
        self.data_split=0.2
        self.device=0

        self.tokenizer_checkpoint = "microsoft/layoutlmv3-large"
        # layoutlmv3 : "microsoft/layoutlmv3-large"
        # layoutlmv2 : "microsoft/layoutlmv2-large-uncased"
        self.init_checkpoint = "xhyi/layoutlmv3_docvqa_t11c5000"
        # layoutlmv3 base : "rubentito/layoutlmv3-base-mpdocvqa"
        # layoutlmv2 base : "hugginglaoda/layoutlmv2-base-uncased_finetuned_docvqa"
        # layoutlmv3 large : "microsoft/layoutlmv3-large"

        self.wandb_flag=True
        self.wandb_key=''
        self.wandb_project_desc=f'FineTuning_{self.model}'
        self.wandb_name=f'{os.path.basename(self.output_path)}_{self.learning_rate}_{self.batch_size}'
        self.wandb_model_desc=f'{self.model} - Fine Tuned on DocVQA, dataset: "Circulars_{os.path.basename(self.data_path)}'