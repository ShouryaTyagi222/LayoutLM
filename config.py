class Config:
    def __init__(self):
        self.model='LayoutLMv3'

        self.image_dir_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/Images'
        self.data_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data.json'
        self.output_path='/data/circulars/DATA/LayoutLM/Model_output'
        self.banned_txt_path='/data/circulars/DATA/LayoutLM/docvqa_dataset/banned_txt_files.txt'

        self.epochs=20
        self.batch_size=8
        self.learning_rate=5e-5
        self.data_split=0.2
        self.device=1

        self.init_checkpoint="rubentito/layoutlmv3-base-mpdocvqa"
        # layoutlmv3 base : rubentito/layoutlmv3-base-mpdocvqa
        # layoutlmv3 large : 