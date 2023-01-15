import os
import time
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import default_data_collator
from utils import convert_exp, get_logger, cal_performance
import random
import numpy as np
from data import LanguageModelDataset


class TextCNN(nn.Module):
    def __init__(self, n_labels, vocab_size, embed_dim, kernel_sizes, chanel_num, kernel_num):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, kernel_num, (size, embed_dim)) for size in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.output = nn.Linear(len(kernel_sizes) * kernel_num, n_labels)

    def forward(self, x):
        # input_batch : [n_step, batch_size, embedding size]
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.output(x)
        # logits = F.log_softmax(self.output(x), dim=1)
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    predictions, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['input_ids'].to(args.device))
            loss = criterion(outputs, batch["labels"].to(torch.int64).to(args.device))
            total_loss += loss
            batch_predictions = torch.argmax(outputs, dim=-1).detach().cpu().tolist()
            predictions.extend(batch_predictions)
            labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader)
    acc, f1, recall = cal_performance(predictions, labels)
    model.train()
    return acc, f1, recall, avg_loss


def train(model, train_dataloader, eval_dataloader, criterion, optimizer, scheduler):
    # training

    logger.info(f" Start training for {args.n_epochs} epochs")

    best_f1 = 0

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()

        model.train()
        start_time = time.time()
        len_iter = len(train_dataloader)
        # print(len_iter)
        val_acces, val_f1s, val_recalls, val_losses = [], [], [], []  # plt

        for i, batch in enumerate(train_dataloader, start=1):
            outputs = model(batch['input_ids'].to(args.device))
            labels = batch['labels'].to(torch.int64).to(args.device)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if i % args.logging_steps == 0:
                # lr = optimizer.param_groups[0]["lr"]
                time_diff = time.time() - start_time
                ms_per_batch = time_diff * 1000 / args.logging_steps
                logger.info(f'| epoch {epoch:3d} | {i:5d}/{len_iter:5d} batches | '
                            # f'lr {lr:02.2f} |'
                            f'ms/batch {ms_per_batch:5.2f} | '
                            f'loss {loss.item():5.2f}')
                start_time = time.time()

            if i % args.valid_steps == 0:
                acc, f1, recall, avg_loss = evaluate_model(model, eval_dataloader, criterion)
                val_acces.append(acc)
                val_f1s.append(f1)
                val_recalls.append(recall)
                val_losses.append(avg_loss)

                logger.info("Valid | acc: %.5f, f1: %.5f, recall: %.5f, global optim: %.5f, loss: %.5f" % (
                    acc, f1, recall, max(val_f1s), avg_loss))

                if f1 > best_f1:
                    logger.info(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, MODEL_NAME+".bin")
                    logger.info('Model has save to %s' % save_dir)
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    torch.save(model.state_dict(),os.path.join(save_dir))

        elapsed = time.time() - epoch_start_time
        logger.info(f'| End of epoch {epoch:3d} | time: {elapsed:5.2f}s |')


def main():
    # loading data
    logger.info("Loading dataset from {} ...".format(args.train_path + ' ' + args.dev_path))
    train_dataset = LanguageModelDataset(args.train_path, args.max_seq_len)
    eval_dataset = LanguageModelDataset(args.dev_path, args.max_seq_len)
    # data loader
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset,  batch_size=args.batch_size)

    # model
    model = TextCNN(n_labels=args.n_labels, vocab_size=train_dataset.vocab_size, embed_dim=args.embed_dim,
                    kernel_sizes=[int(size) for size in args.kernel_sizes.split(',')],
                    chanel_num=args.chanel_num, kernel_num=args.kernel_num)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    model.to(args.device)

    train(model, train_dataloader, eval_dataloader, criterion, optimizer, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    MODEL_NAME = 'TextCNN'
    parser.add_argument("--train_path", default='data/train.txt', type=str)
    parser.add_argument("--dev_path", default='data/dev.txt', type=str)
    parser.add_argument("--save_dir", default="./checkpoints", type=str)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('--kernel_num', type=int, default=16)
    parser.add_argument('--chanel_num', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--n_labels", default=8, type=int)

    parser.add_argument("--valid_steps", default=5, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)
    args = parser.parse_args()
    print(f"args: {args}")
    print('-' * 88)

    logger = get_logger('./logs/', '/{}_{}.txt'.format(MODEL_NAME, args.seed), __name__)
    set_seed(args.seed)

    main()
