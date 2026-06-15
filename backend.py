# -*- coding: utf-8 -*-
import numpy as _np


def _gpu_enabled():
    try:
        import config
        return bool(getattr(config, 'use_gpu', True))
    except Exception:
        return True


try:
    if not _gpu_enabled():
        raise ImportError("GPU отключён в config.use_gpu")
    import cupy as xp
    import cupyx.scipy.sparse as _sp
    from cupyx.scipy.sparse.linalg import factorized
    xp.cuda.Device(0).use()
    GPU = True
except Exception:
    import numpy as xp
    import scipy.sparse as _sp
    from scipy.sparse.linalg import factorized
    GPU = False


def diags(diagonals, offsets, format="csc"):
    return _sp.diags(diagonals, offsets, format=format)


def to_cpu(arr):
    if GPU and hasattr(arr, 'get'):
        return arr.get()
    return _np.asarray(arr)


def to_gpu(arr):
    if GPU:
        return xp.asarray(arr)
    return _np.asarray(arr)
