from utils_multiple_choice import SocialIQaQ2RelProcessor

processor = SocialIQaQ2RelProcessor()
data_dir = "data/socialiqa"
examples = processor.get_dev_examples(data_dir)
examples = processor.get_test_examples(data_dir)
examples = processor.get_train_examples(data_dir)

