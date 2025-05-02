import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def normalize_to_sum(xx: npt.NDArray) -> npt.NDArray:
    return xx / np.sum(xx)


@numba.njit
def normalize_to_max(xx: npt.NDArray) -> npt.NDArray:
    return xx / np.max(xx)
