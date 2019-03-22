# -*- coding: utf-8 -*-
"""

"""
import argparse
import os

import sentencepiece as spm
import torch
import torch.nn.functional as F
from logzero import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, code_data, source_data, spm_code_model, spm_source_model):
        self.spm_code_model = spm.SentencePieceProcessor()
        self.spm_code_model.Load(spm_code_model)

        self.spm_source_model = spm.SentencePieceProcessor()
        self.spm_source_model.Load(spm_source_model)

        self.n_vocab_code = len(self.spm_code_model)
        self.n_vocab_source = len(self.spm_source_model)

        self.code_data = [l.strip() for l in open(code_data, 'r')]
        self.source_data = [l.strip() for l in open(source_data, 'r')]
        self.logger = logger

    def __getitem__(self, idx):
        code_text = self.code_data[idx]
        source_text = self.source_data[idx]
        code_indices = torch.Tensor(self._encode_text(code_text, False, spm_model=self.spm_code_model))
        source_indices = torch.Tensor(self._encode_text(source_text, True, spm_model=self.spm_source_model))
        return code_indices, source_indices

    def _encode_text(self, text, add_special_symbol, spm_model):
        indices = spm_model.encode_as_ids(text)
        if add_special_symbol:
            indices = [spm_model.bos_id()] + indices + [spm_model.eos_id()]
        return indices

    def __len__(self):
        return len(self.source_data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.tensor(lengths)

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths


class RNNEncoder(nn.Module):
    def __init__(self, n_vocab, embed_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size)

    def forward(self, xs, xs_len):
        exs = self.embedding(xs)
        packed_exs = pack_padded_sequence(exs, xs_len, batch_first=True)
        packed_hs, (hs, cs) = self.lstm(packed_exs)
        return packed_hs, (hs, cs)


class RNNDecoder(nn.Module):
    def __init__(self, n_vocab, embed_size, hidden_size):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_vocab)

    def forward(self, xs, xs_len, enc_states):
        exs = self.embedding(xs)

        hs = enc_states[0][0]
        cs = enc_states[1][0]
        hs_list = []
        for ex in exs.unbind(dim=1):
            hs, cs = self.lstm(ex, (hs, cs))
            hs_list.append(hs)
        cat_hs = torch.cat(hs_list, dim=0)
        logits = self.linear(cat_hs)
        return logits


class EmbeddingCompressor(nn.Module):
    def __init__(self, n_codebooks, n_centroids, hidden_dim, tau):
        super(EmbeddingCompressor, self).__init__()
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        """
        self.M = n_codebooks
        self.K = n_centroids
        self.tau = tau

        M = self.M
        K = self.K

        self.l1 = nn.Linear(hidden_dim, M * K)
        # self.l2 = L.Linear(M * K // 2, M * K, initialW=u_init, initial_bias=u_init)
        uni_mat = torch.nn.init.uniform_(torch.empty(M * K, hidden_dim))
        self.codebook = nn.Parameter(uni_mat)

    def forward(self, hidden_vec):
        probs = F.gumbel_softmax(self.l1(hidden_vec).view(-1, self.M * self.K), self.tau)
        # probs ==> batchsize, M * K
        code_sum = torch.matmul(probs, self.codebook)
        return code_sum

    def predict(self, hidden_vec):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, n_vocab_code, n_vocab_source, embed_size, hidden_size, n_codebook, n_centroid):
        super(EncoderDecoder, self).__init__()
        self.code_encoder = RNNEncoder(n_vocab_code, embed_size, hidden_size)
        self.src_encoder = RNNEncoder(n_vocab_source, embed_size, hidden_size)
        self.decoder = RNNDecoder(n_vocab_code, embed_size, hidden_size)

        self.codes = EmbeddingCompressor(n_codebook, n_centroid, hidden_size, tau=1.0)

        self.N = n_codebook
        self.K = n_centroid

    def forward(self, pos, pos_len, src, src_len):
        _, (hs, cs) = self.code_encoder(pos, pos_len)
        # hs.size() ==> n_layer, batch_size, hidden_dim
        code_sum = self.codes(hs)

        # sort source sequence in decending order
        sorted_src_len, inds = torch.sort(src_len.clone().detach(), 0, descending=True)
        sorted_src = src[inds]
        _, (hs, cs) = self.src_encoder(sorted_src, sorted_src_len)
        # unsort sequence to original order
        hs = torch.zeros_like(hs).scatter_(1, inds[None, :, None].expand(1, hs.shape[1], hs.shape[2]), hs)

        dec_init_hs = hs + code_sum[None]
        cs = torch.zero_(torch.empty_like(cs))
        logits = self.decoder(pos, pos_len, (dec_init_hs, cs))
        return logits


def train_epoch(model, criterion, train_iter, optimizer, device):
    total_loss = 0.0
    total_tokens = 0
    model.train()
    for batch in train_iter:
        pos, pos_len, src, src_len = batch
        logits = model(pos.to(device), pos_len.to(device), src.to(device), src_len.to(device))
        loss = criterion(F.log_softmax(logits, dim=1), pos.view(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_tokens = len(pos.nonzero())
        total_loss += loss.data * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def validate_epoch(model, criterion, valid_iter, device):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in valid_iter:
        pos, pos_len, src, src_len = batch
        logits = model(pos.to(device), pos_len.to(device), src.to(device), src_len.to(device))
        loss = criterion(F.log_softmax(logits, dim=1), pos.view(-1).to(device))
        n_tokens = len(pos.nonzero())
        total_loss += loss.data * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def main(args):
    # TODO: GPUに対応させる
    # TODO: Tensorboard


    train_dataset = TranslationDataset(
        spm_code_model=args.spm_code_model,
        spm_source_model=args.spm_source_model,
        code_data=args.train_code,
        source_data=args.train_source
    )
    valid_dataset = TranslationDataset(
        spm_code_model=args.spm_code_model,
        spm_source_model=args.spm_source_model,
        code_data=args.valid_code,
        source_data=args.valid_source
    )
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    valid_iter = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = EncoderDecoder(
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise NotImplementedError

    for epoch in range(args.epoch):
        train_loss = train_epoch(model, criterion, train_iter, optimizer, device)
        valid_loss = validate_epoch(model, criterion, valid_iter, device)
        logger.info('Complete epoch {}: Train {}\tValid {}'.format(epoch, train_loss, valid_loss))


def get_args():
    parser = argparse.ArgumentParser(description='My sequence-to-sequence model')
    parser.add_argument('--train_code', type=os.path.abspath, help='write here')
    parser.add_argument('--train_source', type=os.path.abspath, help='write here')
    parser.add_argument('--valid_code', type=os.path.abspath, help='write here')
    parser.add_argument('--valid_source', type=os.path.abspath, help='write here')

    parser.add_argument('--gpu', type=int, default=-1, help='write here')

    parser.add_argument('--spm_code_model', type=os.path.abspath, help='write here')
    parser.add_argument('--spm_source_model', type=os.path.abspath, help='write here')

    parser.add_argument('--epoch', default=30, type=int, help='write here')
    parser.add_argument('--batch_size', default=32, type=int, help='write here')
    parser.add_argument('--embed_dim', default=128, type=int, help='write here')
    parser.add_argument('--hidden_dim', default=128, type=int, help='write here')
    parser.add_argument('--optimizer', default='Adam', type=str, help='write here')
    parser.add_argument('--lr', default=0.001, type=float, help='write here')

    # codes
    parser.add_argument('--codebook', '-N', default=2, type=int, help='write here')
    parser.add_argument('--centroid', '-K', default=4, type=int, help='write here')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
