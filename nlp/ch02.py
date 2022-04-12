import torch
import torch.nn.functional as F

TEXT = "My name is Flash. I am very very fast!!!"


def char_tokenizer(text: str) -> list:
    tokens = list(text)
    return tokens


def word_tokenizer(text: str) -> list:
    tokens = text.split()
    return tokens


def get_token2idx(tokens):
    token2idx = {tkn: idx for idx, tkn in enumerate(sorted(set(tokens)))}
    return token2idx


if __name__ == "__main__":
    text = TEXT
    ## step-1: tokenization
    tokens = char_tokenizer(text)  ## getting the character-level lokens
    token2idx = get_token2idx(tokens)  ## get the token-to-idx mapping
    input_ids = [token2idx[tkn] for tkn in tokens]  ## numericalization
    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    print(one_hot_encodings.shape)
    print(len(tokens))
    print(len(token2idx))

    print(tokens[0])
    print(input_ids[0])
    print(one_hot_encodings[0])

    ## using word tokenizer
    word_tokens = word_tokenizer(text)
    print(word_tokens)
    token2idx = get_token2idx(word_tokens)
    print(token2idx)
    input_ids = [token2idx[tkn] for tkn in word_tokens]
    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    print(word_tokens[0])
    print(input_ids[0])
    print(one_hot_encodings[0])
