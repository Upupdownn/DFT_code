'''
zscore_normalize()
softmax_normalize()
dft_transform()
get_amplitude_spectrum()
get_phase_spectrum()
'''

import numpy as np
from scipy import fft
from scipy.special import softmax
from scipy.stats import zscore
import pywt


def preprocess_edm(x, axis=1):
    """
    Apply row-wise z-score normalization followed by softmax.
    """
    x = np.asarray(x, dtype=float)

    x = zscore(x, axis=axis)
    x = softmax(x, axis=axis)

    return x


def fft_transform(
    x,
    axis=1,
    preprocess=True,
    remove_dc=False,
    half_spectrum=False,
):
    """
    Apply FFT along the specified axis.

    Parameters
    ----------
    x : array-like
        Input feature matrix.

    axis : int, default=1
        Axis along which FFT is applied.

    preprocess : bool, default=True
        Whether to apply EDM preprocessing before FFT.

    remove_dc : bool, default=False
        Whether to remove the DC component.

    half_spectrum : bool, default=False
        Whether to retain only the non-redundant half spectrum.

    Returns
    -------
    amplitude : ndarray
        FFT amplitude spectrum.

    phase : ndarray
        FFT phase spectrum in radians.
    """
    x = np.asarray(x, dtype=float)

    if preprocess:
        x = preprocess_edm(x, axis=axis)

    values = fft.fft(x, axis=axis)

    amplitude = np.abs(values)
    phase = np.angle(values)

    if half_spectrum:
        n = amplitude.shape[axis]
        keep = n // 2 + 1
        indices = np.arange(keep)

        amplitude = np.take(amplitude, indices=indices, axis=axis)
        phase = np.take(phase, indices=indices, axis=axis)

    if remove_dc:
        indices = np.arange(1, amplitude.shape[axis])

        amplitude = np.take(amplitude, indices=indices, axis=axis)
        phase = np.take(phase, indices=indices, axis=axis)

    return amplitude, phase


def dct_features(x, axis=1, preprocess=False, norm="ortho"):
    """
    Extract DCT-II features.

    Returns the full DCT coefficient matrix.
    """
    x = np.asarray(x, dtype=float)

    if preprocess:
        x = preprocess_edm(x, axis=axis)

    return fft.dct(
        x,
        type=2,
        norm=norm,
        axis=axis,
    )


def wavelet_features(
    x,
    axis=1,
    preprocess=False,
    wavelet="db4",
    level=None,
):
    """
    Extract discrete wavelet transform coefficients.
    """
    x = np.asarray(x, dtype=float)

    if preprocess:
        x = preprocess_edm(x, axis=axis)

    x = np.moveaxis(x, axis, -1)

    coeffs_all = []

    for row in x.reshape(-1, x.shape[-1]):
        coeffs = pywt.wavedec(
            row,
            wavelet=wavelet,
            level=level,
        )
        coeffs_all.append(np.concatenate(coeffs))

    coeffs_all = np.asarray(coeffs_all)

    return coeffs_all.reshape(
        x.shape[:-1] + (coeffs_all.shape[-1],)
    )