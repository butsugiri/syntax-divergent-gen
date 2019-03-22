# -*- coding: utf-8 -*-
"""

"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SeqCode import dataset
from SeqCode import net
from SeqCode import resource


def predict(model, test_iter, device):
    model.eval()
    for batch in tqdm(test_iter):
        pos, pos_len, src, src_len = batch
        codes_batch = model.predict(pos.to(device), pos_len.to(device))
        for codes in codes_batch.cpu().numpy().tolist():
            print(' '.join(['<c{}>'.format(c) for c in codes]))


def test(args):
    res = resource.Resource(args, train=False, log_filename='predict.log', output_dir=os.path.dirname(args.model))
    res.load_config()
    config = res.config
    logger = res.logger

    test_dataset = dataset.TranslationDataset(
        spm_code_model=config['spm_code_model'],
        spm_source_model=config['spm_source_model'],
        code_data=args.test_code,
        source_data=args.test_source
    )
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    model = net.EncoderDecoder(
        n_vocab_code=test_dataset.n_vocab_code,
        n_vocab_source=test_dataset.n_vocab_source,
        embed_size=config['embed_dim'],
        hidden_size=config['hidden_dim'],
        n_codebook=config['codebook'],
        n_centroid=config['centroid']
    )

    logger.info('Load model parameter from [{}]'.format(args.model))
    model.load_state_dict(torch.load(args.model))

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == 'cuda':
        model.to(device)
    predict(model, test_iter, device)


def get_args():
    parser = argparse.ArgumentParser(description='Sequence code learning script')
    parser.add_argument('--model', type=os.path.abspath, help='write here')
    parser.add_argument('--test_code', type=os.path.abspath, help='write here')
    parser.add_argument('--test_source', type=os.path.abspath, help='write here')
    parser.add_argument('--gpu', type=int, default=-1, help='write here')
    parser.add_argument('--batch_size', default=32, type=int, help='write here')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    test(args)
