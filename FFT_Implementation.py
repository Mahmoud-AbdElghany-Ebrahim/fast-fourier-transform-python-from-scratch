import numpy as np


def DFT(x):
    n = np.arange(len(x))
    M = np.exp(-2j * np.pi * n.reshape((len(x), 1)) * n / len(x))
    return np.dot(M, x)


def FFT(arr):
    N = len(arr)
    m = int (N/2)

    if N <= 32:
        return DFT(arr)
    else:
        even = FFT(arr[::2])
        odd = FFT(arr[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + factor[:m] * odd, even + factor[m:] * odd])


X = np.random.random(512)
# built-in function
print(np.fft.fft(X))
# my function
print(FFT(X))

# compare between two method if equal print true else false
print(np.allclose(FFT(X), np.fft.fft(X)))
