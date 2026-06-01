#!/usr/bin/env python3
"""
Generate frequency-domain feature matrices from an input feature matrix.

Input:
    TSV file with samples as rows and features as columns.

Outputs:
    - fft_amplitude_full.tsv
    - fft_phase_full.tsv
    - fft_amplitude_processed.tsv
    - fft_phase_processed.tsv
    - dct_features.tsv
    - wavelet_features.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.frequency_utils import fft_transform, dct_features, wavelet_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FFT, DCT, and wavelet features from a feature matrix."
    )

    parser.add_argument(
        "input_file",
        help="Input TSV feature matrix. Rows are samples and columns are features.",
    )

    parser.add_argument(
        "output_dir",
        help="Directory to save transformed feature matrices.",
    )

    return parser.parse_args()


def make_feature_names(prefix, n_features):
    return [f"{prefix}_{i}" for i in range(n_features)]


def save_matrix(values, sample_ids, output_file, prefix, sep="\t"):
    values = np.asarray(values)

    df = pd.DataFrame(
        values,
        index=sample_ids,
        columns=make_feature_names(prefix, values.shape[1]),
    )

    df.index.name = "sample"
    df.to_csv(output_file, sep=sep)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_table(args.input_file, index_col=0)
    sample_ids = feature_df.index
    x = feature_df.to_numpy(dtype=float)

    # FFT full amplitude and phase
    fft_amp_full, fft_phase_full = fft_transform(x, axis=1)

    save_matrix(fft_amp_full, sample_ids, output_dir / "fft_amplitude_full.tsv", prefix="FFT_amp")

    save_matrix(fft_phase_full, sample_ids, output_dir / "fft_phase_full.tsv", prefix="FFT_phase")

    # FFT processed amplitude: half spectrum + DC removed
    fft_amp_processed, fft_phase_processed_raw = fft_transform(x, 1, True, True, True)

    save_matrix(fft_amp_processed, sample_ids, output_dir / "fft_amplitude_processed.tsv", prefix="FFT_amp")

    fft_phase_processed = np.concatenate(
        [
            np.sin(fft_phase_processed_raw),
            np.cos(fft_phase_processed_raw),
        ],
        axis=1,
    )

    save_matrix(fft_phase_processed, sample_ids, output_dir / "fft_phase_processed.tsv", prefix="FFT_phase")

    # DCT features
    dct_values = dct_features(x, axis=1)

    save_matrix(dct_values, sample_ids, output_dir / "dct_features.tsv", prefix="DCT")

    # Wavelet features
    wavelet_values = wavelet_features(x, axis=1)

    save_matrix(wavelet_values, sample_ids, output_dir / "wavelet_features.tsv", prefix="Wavelet")

    print(f"Input samples: {x.shape[0]}")
    print(f"Input features: {x.shape[1]}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()