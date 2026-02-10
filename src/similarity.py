from numba import njit
import numpy as np
from scipy.optimize import linear_sum_assignment

@njit
def find_matches(spec1_mz: np.ndarray, spec2_mz: np.ndarray,
                 Da_tolerance: float = 0.01, ppm_tolerance = 20.0) -> list:
    """Faster search for matching peaks.
    Makes use of the fact that spec1 and spec2 contain ordered peak m/z (from
    low to high m/z).

    Parameters
    ----------
    spec1_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    spec2_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    Da_tolerance:
        Peaks will be considered a match when delta mz <= Da_tolerance appart.
    ppm_tolerance:
        Peaks will be considered a match when delta mz <= ppm_tolerance * mz appart.

    Returns
    -------
    matches
        List containing entries of type (idx1, idx2).

    """
    lowest_idx = 0
    matches = []
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        low_bound = min(mz - Da_tolerance, mz * (1-ppm_tolerance*1e-6))
        high_bound = max(mz + Da_tolerance, mz * (1+ppm_tolerance*1e-6))
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx]
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx + 1
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches

@njit
def collect_peak_pairs(spec1: np.ndarray, spec2: np.ndarray,
                       Da_tolerance: float = 0.01, ppm_tolerance = 20.0):
    # pylint: disable=too-many-arguments
    """Find matching pairs between two spectra.

    Args
    ----
    spec1:
        Spectrum peak mzs and intensities as numpy array (sorted by mzs).
    spec2:
        Spectrum peak mzs and intensities as numpy array (sorted by mzs).
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    Da_tolerance:
        Peaks will be considered a match when delta mz <= Da_tolerance appart.
    ppm_tolerance:
        Peaks will be considered a match when delta mz <= ppm_tolerance * mz appart.

    Returns
    -------
    matching_pairs : numpy array
        Array of found matching peaks.
    """
    matches = find_matches(spec1[:, 0], spec2[:, 0], Da_tolerance, ppm_tolerance)
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i1, i2 in zip(idx1,idx2):
        prod_spec1 = spec1[i1, 0] * spec1[i1, 1]
        prod_spec2 = spec2[i2, 0] * spec2[i2, 1]
        matching_pairs.append([i1, i2, prod_spec1 * prod_spec2])
    return np.array(matching_pairs.copy())

def cosine_hungarian_similarity(
        spec1: np.ndarray, spec2: np.ndarray,
        Da_tolerance: float = 0.1, ppm_tolerance = 20.0
    ) -> tuple[float, int]:
    """ Return cosine score and number of matched peaks between two spectra mzs/intensity.

    Args:
        spec1 (np.ndarray): in shape [peaks, 2(mzs,intensity)], 
        spec2 (np.ndarray): in shape [peaks, 2(mzs,intensity)], 
        Da_tolerance (float, optional): _description_. Defaults to 0.01.
        ppm_tolerance (float, optional): _description_. Defaults to 20.0.

    Returns:
        tuple[float, int]: cosine score and number of matched peaks.
    """
    # 1. sort spec1 and spec2 according to mzs
    spec1 = spec1[np.argsort(spec1[:,0]),:]
    spec2 = spec2[np.argsort(spec2[:,0]),:]
    matching_pairs = collect_peak_pairs(spec1, spec2, Da_tolerance=Da_tolerance, ppm_tolerance=ppm_tolerance)
    if matching_pairs is None:
        return 0., 0
    # 2. sort according to similarity score
    matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind='mergesort')[::-1], :]
    paired_peaks1 = list(set(matching_pairs[:, 0]))
    paired_peaks2 = list(set(matching_pairs[:, 1]))
    matrix_size = (len(paired_peaks1), len(paired_peaks2))
    # linear_sum_assignment optimize the cost, objective -1*similarity
    matching_pairs_matrix = np.zeros(matrix_size)
    for i in range(matching_pairs.shape[0]):
        matching_pairs_matrix[paired_peaks1.index(matching_pairs[i, 0]),
                              paired_peaks2.index(matching_pairs[i, 1])] = -1 * matching_pairs[i, 2]
    if matching_pairs_matrix is None:
        return 0., 0
    row_ind, col_ind = linear_sum_assignment(matching_pairs_matrix)
    score = -1 * matching_pairs_matrix[row_ind, col_ind].sum()
    used_matches = [(paired_peaks1[x], paired_peaks2[y]) for (x, y) in zip(row_ind, col_ind)]
    # Normalize score:
    spec1_norm = spec1[:, 0] * spec1[:, 1]
    spec2_norm = spec2[:, 0] * spec2[:, 1]
    score = score / (np.sqrt(np.sum(spec1_norm**2)) * np.sqrt(np.sum(spec2_norm**2)))
    return score, len(used_matches)



