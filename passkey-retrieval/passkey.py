from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Optional
import random


class PasskeyDataGenerator:
    '''
    A generator that generates the passkey promopt:

    <prefix>
    <context block>
    <context block>
    ...
    <context block>
    <passkey>
    <context block>
    ...
    <context block>
    <suffix>
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prefix: str = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it.\n\n",
        context_block: str = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n",
        suffix: str = "\nWhat is the pass key? The passkey is",
        passkey_block_template: str = "The passkey is {passkey}. Remember it. {passkey} is the passkey.\n",
        prompt_template: str = "{prefix}{context}{suffix}",
    ):
        self.tokenizer = tokenizer
        self.context_block = context_block
        self.prefix = prefix
        self.suffix = suffix
        self.passkey_block_template = passkey_block_template
        self.prompt_template = prompt_template

        self.context_block_len = len(self.tokenizer.encode(self.context_block))
        self.prefix_len = len(self.tokenizer.encode(self.prefix))
        self.suffix_len = len(self.tokenizer.encode(self.suffix))

        self.key_space = [f"{i:05d}" for i in range(100000)]

    def get_one(
        self,
        context_len: int,
        answer_pos: float,
        passkey: Optional[str] = None,
    ) -> tuple[str, str]:
        if passkey is None:
            passkey = random.choice(self.key_space)

        context_len -= self.prefix_len + self.suffix_len
        # answer_index = int(answer_pos * context_len)
        passkey_block = self.passkey_block_template.format(passkey=passkey)
        # Calculate the number of context blocks needed (approximately)
        num_blocks = context_len // self.context_block_len
        passkey_block_index = int(answer_pos * (num_blocks - 1))

        # Generate the context with the passkey inserted
        context = ""
        for i in range(num_blocks):
            if i == passkey_block_index:
                context += passkey_block
            else:
                context += self.context_block
        
        prompt = self.prompt_template.format(
            prefix=self.prefix,
            context=context,
            suffix=self.suffix,
        )
        return prompt, passkey
    


if __name__ == "__main__":
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("/Users/donny/donny/research/ckpts/rwkv6-1.6b", trust_remote_code=True)  # type: ignore
    passkey_data_generator = PasskeyDataGenerator(tokenizer)
    prompt, ans = passkey_data_generator.get_one(500, 1.0)
    print(prompt)
    print(ans)
    prompt, ans = passkey_data_generator.get_one(500, 0.0)
    print(prompt)
    print(ans)
