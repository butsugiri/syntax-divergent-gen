# -*- coding: utf-8 -*-
"""

"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from SeqCode import dataset
from SeqCode import net
from SeqCode import resource


def train_epoch(model, criterion, train_iter, optimizer, device):
    total_loss = 0.0
    total_tokens = 0
    model.train()
    for batch in tqdm(train_iter):
        pos, pos_len, src, src_len, trg_pos, trg_pos_len = batch['net_input']
        logits = model(pos.to(device), pos_len.to(device), src.to(device), src_len.to(device))
        loss = criterion(F.log_softmax(logits, dim=1), trg_pos.view(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_tokens = len(trg_pos.nonzero())
        total_loss += loss.data * n_tokens
        total_tokens += n_tokens

        nonzero_indices = trg_pos.view(-1).nonzero()
        nonzero_targets = trg_pos.view(-1)[nonzero_indices]
        nonzero_predicts = logits.argmax(1).cpu()[nonzero_indices]
        correct_predicts = int(torch.sum(nonzero_targets == nonzero_predicts))
    return total_loss / total_tokens, correct_predicts / total_tokens


def validate_epoch(model, criterion, valid_iter, device):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in tqdm(valid_iter):
        pos, pos_len, src, src_len, trg_pos, trg_pos_len = batch['net_input']
        logits = model(pos.to(device), pos_len.to(device), src.to(device), src_len.to(device))
        loss = criterion(F.log_softmax(logits, dim=1), trg_pos.view(-1).to(device))
        n_tokens = len(pos.nonzero())
        total_loss += loss.data * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def train(args):
    res = resource.Resource(args, train=True)
    res.save_config_file()
    logger = res.logger

    # TODO: support Tensorboard
    # TODO: add README
    # TODO: Report Accuracy

    train_dataset = dataset.CodeLearningDataset(
        spm_code_model=args.spm_code_model,
        spm_source_model=args.spm_source_model,
        code_data=args.train_code,
        source_data=args.train_source,
        size=args.size
    )
    valid_dataset = dataset.CodeLearningDataset(
        spm_code_model=args.spm_code_model,
        spm_source_model=args.spm_source_model,
        code_data=args.valid_code,
        source_data=args.valid_source
    )
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    valid_iter = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    model = net.EncoderDecoder(
        n_vocab_code=train_dataset.n_vocab_code,
        n_vocab_source=train_dataset.n_vocab_source,
        embed_size=args.embed_dim,
        hidden_size=args.hidden_dim,
        n_codebook=args.codebook,
        n_centroid=args.centroid
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == 'cuda':
        model.to(device)

    criterion = nn.NLLLoss(ignore_index=0)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    best_valid_loss = 0.0
    for epoch in range(args.epoch):
        logger.info('Training: epoch [{}]'.format(epoch))
        train_loss, train_acc = train_epoch(model, criterion, train_iter, optimizer, device)
        logger.info('Validation: epoch [{}]'.format(epoch))
        valid_loss = validate_epoch(model, criterion, valid_iter, device)
        logger.info('Complete epoch {}\tTrain loss: {}\tValid loss: {}'.format(epoch, train_loss, valid_loss))

        if epoch == 0 or valid_loss < best_valid_loss:
            model_path = os.path.join(res.output_dir, 'best_model.pt')

            logger.info('Saving model to [{}]'.format(model_path))
            torch.save(model.state_dict(), model_path)
            best_valid_loss = valid_loss


def get_args():
    parser = argparse.ArgumentParser(description='Sequence code learning script')
    parser.add_argument('--train_code', type=os.path.abspath, help='write here')
    parser.add_argument('--train_source', type=os.path.abspath, help='write here')
    parser.add_argument('--valid_code', type=os.path.abspath, help='write here')
    parser.add_argument('--valid_source', type=os.path.abspath, help='write here')
    parser.add_argument('--size', type=int, default=100, help='ratio of the dataset to use')
    parser.add_argument('--gpu', type=int, default=-1, help='write here')
    parser.add_argument('--spm_code_model', type=os.path.abspath, help='write here')
    parser.add_argument('--spm_source_model', type=os.path.abspath, help='write here')
    parser.add_argument('--epoch', default=30, type=int, help='write here')
    parser.add_argument('--batch_size', default=32, type=int, help='write here')
    parser.add_argument('--embed_dim', default=512, type=int, help='write here')
    parser.add_argument('--hidden_dim', default=512, type=int, help='write here')
    parser.add_argument('--optimizer', default='Adam', type=str, help='write here')
    parser.add_argument('--lr', default=0.001, type=float, help='write here')
    # codes
    parser.add_argument('--codebook', '-N', default=2, type=int, help='write here')
    parser.add_argument('--centroid', '-K', default=4, type=int, help='write here')
    # output
    parser.add_argument('--out', '-O', default='result', type=str, help='Output dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(args)
