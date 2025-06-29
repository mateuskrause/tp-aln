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

def _periodic_knots(x, k):
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

# -----------------------

def _make_periodic_spline(x, y, t, k, axis):
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

    # Print the linear system to be solved
    print("Linear system matrix A (shape {}):".format(A.shape))
    print(A)
    print("Right-hand side vectors (each column is a system):")
    print(y_new[:, :][:-1])

    for i in range(extradim):
        cc = woodbury_algorithm(A, ur, ll, y_new[:, i][:-1], k)
        c[:, i] = np.concatenate((cc[-kul:], cc, cc[:kul + k % 2]))
    c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True):
    
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
    t = _periodic_knots(x, k)

    # convert t to a numpy array of floats, ensuring it is contiguous in memory
    t = _as_float_array(t, check_finite)


    
    # x : 1-D array of independent variable values (the "knots" or sample points) at which the data y are provided.
    #     In the context of interpolation, `x` represents the positions along the domain where the function values are known.
    # y : Array of dependent variable values (the data to be interpolated) corresponding to each value in `x`.
    #     In the context of interpolation, `y` contains the function values at the positions specified by `x`.
    # k : Degree of the spline. Must be a non-negative integer. Default is 3 (cubic spline).
    #     In the context of interpolation, `k` determines the smoothness and flexibility of the resulting spline curve.
    # t : Knot vector. If None, a knot vector suitable for periodic splines is generated automatically.
    #     In the context of interpolation, `t` specifies the locations of the knots that define the piecewise polynomial segments of the spline.
    # axis : int, optional
    #     Axis along which the interpolation is performed. Default is 0.
    #     In the context of interpolation, `axis` specifies which axis of the `y` array corresponds to the data points to be interpolated."""
    

    print(f"x: {x}")
    print(f"y: {y}")
    print(f"k: {k}")
    print(f"t: {t}")
    print(f"axis: {axis}")
    # make the perioric spline using the helper function
    return _make_periodic_spline(x, y, t, k, axis)