# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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
        uni_mat = torch.nn.init.uniform_(torch.empty(M * K, hidden_dim))
        self.codebook = nn.Parameter(uni_mat)

    def forward(self, hidden_vec):
        hs = F.softplus(self.l1(hidden_vec))
        logit = torch.log(hs + 1e-08).view(-1, self.M, self.K).view(-1, self.K)
        probs = F.gumbel_softmax(logit, self.tau).view(-1, self.M * self.K)
        # probs ==> batchsize, M * K
        code_sum = torch.matmul(probs, self.codebook)
        return code_sum

    def predict(self, hidden_vec):
        hs = F.softplus(self.l1(hidden_vec))
        logit = torch.log(hs + 1e-08).view(-1, self.M, self.K)
        codes_batch = logit.argmax(dim=2)
        return codes_batch


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

    def predict(self, pos, pos_len):
        _, (hs, cs) = self.code_encoder(pos, pos_len)
        codes_batch = self.codes.predict(hs)
        return codes_batch
