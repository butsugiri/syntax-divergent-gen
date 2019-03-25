# -*- coding: utf-8 -*-
"""

"""
import argparse
import os

import stanfordnlp


def get_args():
    parser = argparse.ArgumentParser(description='pos tagging')
    parser.add_argument('--input', '-i', required=True, type=os.path.abspath, help='write here')
    parser.add_argument('--model', '-m', required=True, type=os.path.abspath, help='write here')
    args = parser.parse_args()
    return args


def main(args):
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos', models_dir=args.model, pos_batch_size=6000)

    with open(args.input, 'r') as fi:
        for line in fi:
            line = line.strip()
            doc = nlp(line)
            out = ' '.join(word.pos for sentence in doc.sentences for word in sentence.words)
            print(out)


if __name__ == "__main__":
    args = get_args()
    main(args)
