# -*- coding: utf-8 -*-
"""
sometimes corase pos is empty
"""
import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--length', default=30, type=int, help='write here')
    args = parser.parse_args()
    return args


def main(args):
    for line in sys.stdin:
        line = line.rstrip()
        l1, l2 = line.split('\t')
        if l1 == '':
            continue
        print(line)


if __name__ == "__main__":
    args = get_args()
    main(args)