'''
=======================================
Provides operations on vectors.
(adapted from perturbed_netwon repo
 see https://bitbucket.org/johannrudi/perturbed_newton.git)

Author:             Johann Rudi
=======================================
'''

import numpy as np

def _getDataShape(u):
    return u.vector().get_local().shape

def _getData(u):
    return u.vector().get_local()

def _setData(u, u_data):
    u.vector().set_local(u_data)

def _getDataShaped(u):
    u_data = _getData(u)
    dim = u.value_dimension(0)
    n_entries_per_dim = int(u_data.size/dim)
    return np.reshape(u_data, (n_entries_per_dim, dim))

def _setDataShaped(u, u_data):
    u_data_shaped = np.reshape(u_data, _getDataShape(u))
    _setData(u, u_data_shaped)

def getVertexValues(u, mesh=None):
    # get values
    if mesh is not None:
        vals = u.compute_vertex_values(mesh)
    else:
        vals = u.compute_vertex_values()
    # split values for higher-dimensional fields
    dim = u.value_dimension(0)
    if 1 < dim:
        return np.matrix(np.split(vals, dim)).T
    else:
        return vals

def setZero(u):
    u_data = _getData(u)
    u_data[:] = 0.0
    _setData(u, u_data)

def setValue(u, value):
    u_data = _getData(u)
    u_data[:] = value
    _setData(u, u_data)

def setVector(u, vector):
    u_data = _getData(u)
    vector_data = vector.get_local()
    u_data[:] = vector_data[:]
    _setData(u, u_data)

def scale(u, scalar):
    u_data = _getData(u)
    u_data[:] *= scalar
    _setData(u, u_data)

def multiplyArray(u, a):
    u_data = _getDataShaped(u)
    u_data = (u_data.T * a).T
    _setDataShaped(u, u_data)

def computeMagnitude(u, reg=0.0):
    u_data = _getDataShaped(u)
    return np.sqrt(np.sum(np.abs(u_data)**2, axis=-1) + reg*reg)

def _normalize_array(a, reg=0.0):
    norm = np.sqrt(np.sum(np.abs(a)**2) + reg*reg)
    if np.isfinite(1.0/norm):
        return a/norm
    else:
        return np.zeros_like(a)

def normalizeMagnitude(u, reg=0.0):
    u_data = _getDataShaped(u)
    u_data = np.apply_along_axis(func1d=_normalize_array, axis=1, arr=u_data, reg=reg)
    _setDataShaped(u, u_data)

def _boundmax_array(a, max_value=1.0):
    norm = np.linalg.norm(a)
    if max_value < norm:
        return a*max_value/norm
    else:
        return a

def boundMaxMagnitude(u, max_value=1.0):
    u_data = _getDataShaped(u)
    u_data = np.apply_along_axis(_boundmax_array, 1, u_data)
    _setDataShaped(u, u_data)

def addNoiseRandUniform(u, noiseSeed=1):
    # get data
    u_data = _getData(u)

    # add random noise
    np.random.seed(seed=noiseSeed)
    u_data[:] += np.random.rand(u_data.size)

    # set data
    _setData(u, u_data)

def addNoiseRandNormal(u, noiseSeed=1, noiseStdDev=1.0):
    # get data
    u_data = _getData(u)

    # add random noise
    np.random.seed(seed=noiseSeed)
    u_data[:] += noiseStdDev * np.random.randn(u_data.size)

    # set data
    _setData(u, u_data)

#======================================

import inspect, pprint

def printMeta(u):
    print(type(u))
    pprint.pprint(inspect.getmembers(type(u)))
