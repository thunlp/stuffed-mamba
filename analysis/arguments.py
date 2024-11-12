from tap import Tap


class Args(Tap):
    train_len = 8 * 1024
    max_len = 32 * 1024
    model_path: str = '/path/to/model'
    tok_path: str = '/path/to/tokenizer'
    prompt_name = 'nextlines'
    bucket_size = 512
    device = 'cuda'
    file_ext = 'pdf'
    data_path: str = '/path/to/data'
