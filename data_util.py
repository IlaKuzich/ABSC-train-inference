from transformers import BertTokenizer
import numpy as np
import torch


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def preprocess_text(text, aspect, tokenizer, device):
    aspect = aspect.lower().strip()
    text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]

    text_indexes = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)

    text_len = np.sum(text_indexes != 0)
    aspect_len = np.sum(aspect != 0)
    concat_bert_indexes = tokenizer.text_to_sequence(
        '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    concat_segments_indexes = [0] * (text_len + 2) + [1] * (aspect_len + 1)
    concat_segments_indexes = pad_and_truncate(concat_segments_indexes, tokenizer.max_seq_len)

    text_bert_indexes = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    aspect_bert_indexes = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    inputs = [torch.tensor([data], device=device) for data in (concat_bert_indexes, concat_segments_indexes, text_bert_indexes, aspect_bert_indexes)]
    return inputs