'''
This code is from or is inspired by the SciPy implementation of periodic B-splines.
It is a modified version of the original code to work with periodic B-splines.
https://github.com/scipy/scipy/blob/main/scipy/interpolate/_bsplines.py
'''

import numpy as np
import operator

from math import prod
from scipy.interpolate import BSpline
from scipy._lib._util import normalize_axis_index
from scipy.interpolate import _dierckx

from woodbury_algorithm import *

# -----------------------

def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64
    
def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.

    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = np.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and not np.isfinite(x).all():
        raise ValueError("Array must not contain infs or nans.")
    return x

def periodic_knots(x, k):
    '''
    returns vector of nodes on circle
    '''
    xc = np.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = np.diff(xc)
        xc[1: -1] -= dx[:-1] / 2
    dx = np.diff(xc)
    t = np.zeros(n + 2 * k)
    t[k: -k] = xc
    for i in range(0, k):
        # filling first `k` elements in descending order
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        # filling last `k` elements in ascending order
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t


def print_full_cyclic_banded_matrix(A, ur, ll):
    """
    Assemble and print the full cyclic banded matrix from banded part A (banded, shape (num_bands, n))
    and wrap-around blocks ur, ll.
    """
    num_bands, n = A.shape
    kul = ur.shape[0]
    full = np.zeros((n, n))
    center = num_bands // 2

    # Fill the banded part
    for i in range(num_bands):
        offset = i - center
        diag = A[i]
        if offset >= 0:
            # Fill the main and upper diagonals
            np.fill_diagonal(full[:, offset:], diag[:n - offset])
        else:
            # Fill the lower diagonals
            np.fill_diagonal(full[-offset:, :], diag[-offset:])
            
    # Add wrap-around blocks if their shapes are compatible
    if kul > 0 and ur.shape == (kul, kul) and ll.shape == (kul, kul):
        full[:kul, -kul:] += ur
        full[-kul:, :kul] += ll

    # print("Full cyclic banded matrix:")
    print(full)


def make_interp_spline(x, y, show_linear_system = False):

    k = 3                   # default B-spline degree as 3 (cubic B-spline)
    axis=0                  # default axis for interpolation is 0 (first dimension)
    t=None                  # default knot vector is None, which means it will be generated automatically
    check_finite = True     # check for finite values in the input arrays

    # ---------------

    # assure that y is a numpy array
    y = np.asarray(y)

    # ensures that the axis parameter refers to a valid and unambiguous axis of the NumPy array y
    axis = normalize_axis_index(axis, y.ndim)   

    # convert the variables into numpy arrays of a specific floating-point type, and then check if they contain any non-finite values (like NaN or inf)
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)

    # now internally interp axis is zero
    y = np.moveaxis(y, axis, 0)

    # ensures that k is converted to a plain python integer type
    k = operator.index(k)

    # generate a knot vector suitable for periodic splines, ensuring the spline wraps around smoothly
    t = periodic_knots(x, k)

    # convert t to a numpy array of floats, ensuring it is contiguous in memory
    t = _as_float_array(t, check_finite)

    # ---------------

    n = y.shape[0]
    extradim = prod(y.shape[1:])            # calculates the product of the dimensions of y excluding the first dimension
    y_new = y.reshape(n, extradim)          # reshapes y to a 2D array with n rows and extradim columns
    c = np.zeros((n + k - 1, extradim))     # initializes a zero array for coefficients for the spline

    # n <= k case is solved with full matrix
    if n <= k:
        print("n <= k case, of which we need to solve with full matrix (not doing it now)")

    # number of coefficients needed to represent the spline. 
    nt = len(t) - k - 1

    # size of block elements
    kul = int(k / 2)

    # kl = ku = k
    ab = np.zeros((3 * k + 1, nt), dtype=np.float64, order='F')

    # upper right and lower left blocks
    ur = np.zeros((kul, kul))
    ll = np.zeros_like(ur)

    # `offset` is made to shift all the non-zero elements to the end of the
    # matrix
    # NB: 1. drop the last element of `x` because `x[0] = x[-1] + T` & `y[0] == y[-1]`
    #     2. pass ab.T to _coloc to make it C-ordered; below it'll be fed to banded
    #        LAPACK, which needs F-ordered arrays
    _dierckx._coloc(x[:-1], t, k, ab.T, k)

    # remove zeros before the matrix
    ab = ab[-k - (k + 1) % 2:, :]

    # The least elements in rows (except repetitions) are diagonals
    # of block matrices. Upper right matrix is an upper triangular
    # matrix while lower left is a lower triangular one.
    for i in range(kul):
        ur += np.diag(ab[-i - 1, i: kul], k=i)
        ll += np.diag(ab[i, -kul - (k % 2): n - 1 + 2 * kul - i], k=-i)

    # remove elements that occur in the last point
    # (first and last points are equivalent)
    A = ab[:, kul: -k + kul]

    # ---------------

    # print the cyclic banded matrix and the points of entry
    if(show_linear_system):
        print("----------")
        print("D = N @ P")
        print("\nMatriz N (funções base avaliadas nos parâmetros):")
        print_full_cyclic_banded_matrix(A, ur, ll)
        print("\nMatriz D (pontos de entrada):")
        print(y_new[:, :][:-1])

    # ---------------

    # Solve the cyclic banded linear system for each right-hand side (each column of y_new)
    for i in range(extradim):
        # Solve the system using the Woodbury algorithm for periodic/cyclic banded matrices
        cc = woodbury_algorithm(A, ur, ll, y_new[:, i][:-1], k)
        # Concatenate the periodic wrap-around coefficients to form the full coefficient vector
        c[:, i] = np.concatenate((cc[-kul:], cc, cc[:kul + k % 2]))

    # Reshape the coefficients to match the original y shape (except for the first dimension)
    c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))

    # Construct and return the periodic B-spline object
    return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)