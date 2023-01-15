import traceback

import numpy as np
import os
import logging
from sklearn.metrics import classification_report
import jieba


def convert_exp(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '1	这是一条正样本',
                                                            '0	这是一条负样本',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    if tokenizer:
        tokenized_output = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for example in examples['text']:
            try:
                label, content = example.split('\t')
                encoded_inputs = tokenizer(
                    text=content,
                    truncation=True,
                    max_length=max_seq_len,
                    padding='max_length')
            except:
                print(f'"{example}" -> {traceback.format_exc()}')
                continue

            tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
            tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
            tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
            tokenized_output['labels'].append(int(label))

        for k, v in tokenized_output.items():
            tokenized_output[k] = np.array(v)

        return tokenized_output
    else:
        tokenized_output = {
            'input_ids': [],
            'labels': []
        }
        word_list = []
        for example in examples['text']:
            label, content = example.split('\t')
            content = jieba.cut(content, cut_all=False)
            word_list += [w for w in content]
        word_list = list(set(word_list))
        word_dict = {w: i for i, w in enumerate(word_list)}
        print(len(word_list))
        for example in examples['text']:
            try:
                label, content = example.split('\t')
                content = jieba.cut(content, cut_all=False)
                input_ids = [word_dict[w] for w in content]
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                if len(input_ids) < max_seq_len:
                    input_ids = input_ids + [0 for _ in range(max_seq_len - len(input_ids))]
            except:
                print(f'"{example}" -> {traceback.format_exc()}')
                continue
            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(int(label))

        for k, v in tokenized_output.items():
            tokenized_output[k] = np.array(v)
        return tokenized_output


def get_logger(save_path, log_file_name, name=__name__):
    # print(save_path, log_file_name)
    if not os.path.exists(save_path):  # logs
        os.makedirs(save_path)
    log_path = save_path + log_file_name
    print(f'log path: {log_path}')
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    consle_handler = logging.StreamHandler()
    consle_handler.setLevel(logging.INFO)
    consle_handler.setFormatter(formatter)

    logger.addHandler(consle_handler)
    logger.addHandler(file_handler)
    return logger


def cal_performance(predictions, labels):
    """
    Returns accuracy, f1, recall
    """
    report = classification_report(labels, predictions, zero_division=0, output_dict=True)
    return report['accuracy'], report['macro avg']['f1-score'], report['macro avg']['recall']
