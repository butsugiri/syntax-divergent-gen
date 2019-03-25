# -*- coding: utf-8 -*-
import sentencepiece as spm
import torch
from logzero import logger
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, code_data, source_data, spm_code_model, spm_source_model):
        self.spm_code_model = spm.SentencePieceProcessor()
        self.spm_code_model.Load(spm_code_model)

        self.spm_source_model = spm.SentencePieceProcessor()
        self.spm_source_model.Load(spm_source_model)

        self.n_vocab_code = len(self.spm_code_model)
        self.n_vocab_source = len(self.spm_source_model)

        logger.info('Loading code data from [{}]'.format(code_data))
        self.code_data = [l.strip() for l in open(code_data, 'r')]

        logger.info('Loading source data from [{}]'.format(source_data))
        self.source_data = [l.strip() for l in open(source_data, 'r')]

    def __getitem__(self, idx):
        code_text = self.code_data[idx]
        source_text = self.source_data[idx]
        code_indices = torch.LongTensor(self._encode_text(code_text, add_bos=True, add_eos=False, spm_model=self.spm_code_model))
        source_indices = torch.LongTensor(self._encode_text(source_text, add_bos=True, add_eos=True, spm_model=self.spm_source_model))
        trg_code_indices = torch.LongTensor(self._encode_text(code_text, add_bos=False, add_eos=True, spm_model=self.spm_code_model))
        return code_indices, source_indices, trg_code_indices

    def _encode_text(self, text, add_bos, add_eos, spm_model):
        indices = spm_model.encode_as_ids(text)
        if add_bos:
            indices = [spm_model.bos_id()] + indices
        if add_eos:
            indices = indices + [spm_model.eos_id()]
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
    code_seqs, src_seqs, trg_code_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    code_seqs, code_lengths = merge(code_seqs)
    src_seqs, src_lengths = merge(src_seqs)
    trg_code_seqs, trg_code_lengths = merge(trg_code_seqs)

    return code_seqs, code_lengths, src_seqs, src_lengths, trg_code_seqs, trg_code_lengths
