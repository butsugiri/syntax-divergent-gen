# -*- coding: utf-8 -*-
"""

"""
import argparse
import os
from itertools import groupby

POS_TAGS_TO_KEEP = ('N', 'V', 'PRP', ',', '.')


def get_args():
    parser = argparse.ArgumentParser(description='pos tagging')
    parser.add_argument('--input', '-i', required=True, type=os.path.abspath, help='write here')
    args = parser.parse_args()
    return args


def remove_unnecessary_tags(tokens):
    return [t for t in tokens if t in POS_TAGS_TO_KEEP]


def extract_suffix(tokens):
    out = []
    for token in tokens:
        if token.startswith('N'):
            out.append('N')
        elif token.startswith('V'):
            out.append('V')
        else:
            out.append(token)
    return out


def remove_consecutive_tags(tokens):
    return [x[0] for x in groupby(tokens)]


def main(args):
    with open(args.input, 'r') as fi:
        for line in fi:
            tokens = line.strip().split()
            tokens = extract_suffix(tokens)
            tokens = remove_unnecessary_tags(tokens)
            tokens = remove_consecutive_tags(tokens)
            print(' '.join(tokens))


if __name__ == "__main__":
    args = get_args()
    main(args)
