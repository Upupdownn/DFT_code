#!/usr/bin/env python3
"""
Transforms cfDNA End-Motif (EDM) frequencies into frequency-domain amplitude features
based on the DSP framework.

Input: EDM TSV file per sample (k-mer frequencies)
Output: Amp TSV file per sample
"""

import argparse
import logging
import pandas as pd
import numpy as np
from scipy.special import softmax


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Path to EDM file")
    parser.add_argument("output", help="Path to Amplitude file")
    return parser.parse_args()

def edm2amp(edm_file:str, amp_file:str):
    edm_df = pd.read_table(edm_file, header=0, index_col=0)

    # Row-wise Z-score Standardization
    row_means = edm_df.mean(axis=1)
    row_stds = edm_df.std(axis=1).replace(0, 1)         # Avoid division by zero
    z_scored = edm_df.subtract(row_means, axis=0).divide(row_stds, axis=0)
    
    # Row-wise Softmax Nonlinear Mapping
    sm_features = softmax(z_scored.values, axis=1)
    
    # fft
    fft_results = np.fft.fft(sm_features, axis=1)
    
    # Extract Magnitude Spectra [cite: 64]
    magnitudes = np.abs(fft_results)
    
    # Frequency Component Selection 
    N = edm_df.shape[1]
    amp_data = magnitudes[:, 1 : (N // 2) + 1]
    
    # Construct and Save the Output DataFrame
    num_cols = amp_data.shape[1]
    col_names = [f"{i}hz" for i in range(1, num_cols + 1)]
    
    amp_df = pd.DataFrame(amp_data, index=edm_df.index, columns=col_names)
    
    amp_df.to_csv(amp_file, sep='\t')

    pass


def main():
    args = parse_args()

    edm2amp(args.input, args.output)

if __name__ == "__main__":
    main()