import os
import time
import argparse
from functools import partial
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, default_data_collator, \
    get_scheduler
from utils import convert_exp, get_logger, cal_performance
import random
import numpy as np


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
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device))
            loss = criterion(outputs.logits, batch["labels"].to(args.device))
            total_loss += loss
            batch_predictions = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
            predictions.extend(batch_predictions)
            labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader)
    acc, f1, recall = cal_performance(predictions, labels)
    model.train()
    return acc, f1, recall, avg_loss


def train(model, train_dataloader, eval_dataloader, criterion, optimizer, scheduler, tokenizer):
    # training

    logger.info(f" Start training for {args.n_epochs} epochs")

    best_f1 = 0

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()

        model.train()
        start_time = time.time()
        len_iter = len(train_dataloader)
        print(len_iter)
        val_acces, val_f1s, val_recalls, val_losses = [], [], [], []  # plt

        for i, batch in enumerate(train_dataloader):
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device))
            labels = batch['labels'].to(args.device)
            loss = criterion(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                    save_dir = os.path.join(args.save_dir, MODEL_NAME)
                    logger.info('Model has save to %s' % save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model.save_pretrained(os.path.join(save_dir))
                    tokenizer.save_pretrained(os.path.join(save_dir))

        elapsed = time.time() - epoch_start_time
        logger.info(f'| End of epoch {epoch:3d} | time: {elapsed:5.2f}s |')


def main():
    # model
    logger.info("loading model and tokenizer from {} ...".format(args.model))
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = args.n_labels  # 设置类别数
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # print(model)

    # loading data
    logger.info("Loading dataset from {} ...".format(args.train_path + ' ' + args.dev_path))
    dataset = load_dataset('text', data_files={'train': args.train_path, 'dev': args.dev_path})
    logger.info(dataset)
    # convert the text data into tensors
    convert_func = partial(convert_exp, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    # data loader
    train_dataset, eval_dataset = dataset["train"], dataset["dev"]
    print(type(train_dataset))
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.n_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.to(args.device)

    train(model, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    MODEL_NAME = 'BERT'
    parser.add_argument("--model", default='../pretrained_models/bert-base-chinese', type=str)
    parser.add_argument("--train_path", default='data/train.txt', type=str)
    parser.add_argument("--dev_path", default='data/dev.txt', type=str)
    parser.add_argument("--save_dir", default="./checkpoints", type=str)
    # parser.add_argument("--bert_path", default='', type=str)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Linear warmup over warmup_ratio * total_steps.")

    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--n_labels", default=8, type=int, help="Total classes of labels.")

    parser.add_argument("--valid_steps", default=50, type=int, required=False, help="evaluate frequecny.")
    parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
    args = parser.parse_args()
    print(f"args: {args}")
    print('-' * 88)

    logger = get_logger('./logs/', '/{}_{}.txt'.format(MODEL_NAME, args.seed), __name__)
    set_seed(args.seed)

    main()
