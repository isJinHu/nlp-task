from torch.utils.data import DataLoader, Dataset
import jieba
import traceback
import numpy as np


class LanguageModelDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.vocab = get_vocab()
        self.vocab_size = len(self.vocab)
        self.lines = []
        with open(data, 'r', encoding='utf8') as f:
            self.lines += f.readlines()
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        item = {
            'input_ids': None,
            'labels': None
        }
        example = self.lines[index]
        label, content = example.split('\t')
        content = jieba.cut(content, cut_all=False)
        input_ids = [self.vocab[w] for w in content]
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        if len(input_ids) < self.max_seq_len:
            input_ids = input_ids + [0 for _ in range(self.max_seq_len - len(input_ids))]

        item['input_ids']=input_ids
        item['labels']=int(label)

        for k, v in item.items():
            item[k] = np.array(v)
        return item

    def __len__(self):
        return len(self.lines)


def get_vocab():
    lines = []
    with open('./data/train.txt', 'r', encoding='utf8') as f:
        lines += f.readlines()
    with open('./data/dev.txt', 'r', encoding='utf8') as f:
        lines += f.readlines()

    word_list = []
    for example in lines:
        label, content = example.split('\t')
        content = jieba.cut(content, cut_all=False)
        word_list += [w for w in content]
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    # print(len(word_list))
    return word_dict


if __name__ == '__main__':
    get_vocab()
