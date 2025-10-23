from __future__ import annotations

import argparse
import numpy as np


def build_argparser():
    p = argparse.ArgumentParser(description='Evaluate PR-MAE checkpoint (placeholder)')
    p.add_argument('--ckpt', type=str, required=False)
    return p


def main():
    args = build_argparser().parse_args()
    print('Evaluation CLI placeholder. Use training script for now.')


if __name__ == '__main__':
    main()
