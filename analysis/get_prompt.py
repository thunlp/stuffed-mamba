import json
from typing import Optional


def get_long_prompt(prompt_name: str = 'nextlines', data_path: Optional[str] = None) -> str:
    if prompt_name == "nextlines":
        return '\n' * 300000
    if prompt_name == 'newlines':
        return '\n' * 300000
    if prompt_name == 'capital':
        return 'The capital of China is Beijing. The capital of USA is Washington. The capital of Norway is Oslo. ' * 2000  # 1200 -> 25K tokens
    if prompt_name == 'book3':
        prompt = ''
        assert data_path is not None
        path = data_path
        n_lines = 2
        with open(path) as f:
            for i, line in enumerate(f):
                if i == n_lines:
                    break
                text = json.loads(line)['text']
                prompt += '\n\n' + text
        return prompt
    if prompt_name == 'slimpj':
        prompt = ''
        assert data_path is not None
        path = data_path
        n_lines = 128
        with open(path) as f:
            for i, line in enumerate(f):
                if i == n_lines:
                    break
                text = json.loads(line)['text']
                prompt += text
        return prompt
    if prompt_name == 'redpj_16k':
        prompt = ''
        assert data_path is not None
        path = data_path
        n_lines = 64
        with open(path) as f:
            for i, line in enumerate(f):
                if i == n_lines:
                    break
                orig_content = json.loads(line)['original_content']
                data = json.loads(orig_content)
                text = data['raw_content']
                prompt += text
        return prompt
    raise ValueError

