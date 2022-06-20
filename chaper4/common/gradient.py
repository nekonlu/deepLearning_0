# coding: utf-8
import numpy as np


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

# def numerical_gradient(f, x):
#     h = 1e-4
#
#     x1 = x.copy()
#     x2 = x.copy()
#
#     x1 += h
#     x2 -= h
#
#     func = np.vectorize(f)
#
#     x1 = func(x1)
#     x2 = func(x2)
#
#     grad = (x1 - x2) / (2 * h)
#
#     return grad
