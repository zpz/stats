"""Agresti-Coull confidence interval for the estimate of binomial proportion.

Refrence: Agresti and Coull, 1998
"Approximate is better than 'exact' for interval estimatin
of binomial proportions", The American Statistician.
Wikipedia entry: "Binomial proportion confidence interval".
"""

import math
import collections
import numpy as np


def _ac(n, x, z):
    if n <= 0:
        return None, None
    one_over_n = 1.0 / (n + z * z)
    p = one_over_n * (x + z * z / 2.0)
    # Equivalently,
    #   p = (x + 2.0) / (n + 4.0)
    # when z = 2.0

    d = z * math.sqrt(one_over_n * p * (1.0 - p))

    # (n, x): p,      d
    # (0, 0): 0.5,    0.5
    # (1, 0): 0.4,    0.4382
    # (2, 0): 0.3333, 0.3849
    # (3, 0): 0.2857, 0.3415
    # (1, 1): 0.6,    0.4382
    # (2, 2): 0.6667, 0.3849
    # (3, 3): 0.7143, 0.3415

    # The Agresti-Coull interval is [p - d, p + d].
    # The naiive estimator is x / n.
    return p, d


def agresti_coull(n, x, z=2.0):
    """Agresti-Coull

    Parameters
    ----------
    n: sample size; scalar or numpy array
    x: number of successes; scalar or numpy array compatible with 'n'

    Returns
    -------
    center and half-width of the interval.
    """

    if isinstance(n, collections.Iterable) or isinstance(x, collections.Iterable):
        if not isinstance(n, collections.Iterable):
            n = [n] * len(x)
        elif not isinstance(x, collections.Iterable):
            x = [x] * len(n)
        pp, dd = [], []
        for nn, xx in zip(n, x):
            p, d = _ac(nn, xx, z)
            pp.append(p)
            dd.append(d)
        if isinstance(n, np.ndarray) and isinstance(x, np.ndarray):
            pp = np.array(pp, float)
            dd = np.array(dd, float)
        return pp, dd
    return _ac(n, x, z)
