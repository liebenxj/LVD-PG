using PyCall

py"""
from scipy.fft import dct, idct
import numpy as np

def dct_uniform_quantization(x, max_nbins = 256, minval = None, maxval = None):
    if minval is None:
        minval = np.min(x, axis = 0, keepdims = True)
    if maxval is None:
        maxval = np.max(x, axis = 0, keepdims = True)
    n_bins = np.ones([1, x.shape[1]], dtype = np.int32) * max_nbins
    n_bins = np.minimum(np.floor(maxval - minval).astype(np.int32), n_bins)
    
    x_quantized = np.floor((x - minval) / (maxval - minval + 1e-6) * n_bins).astype(np.uint32) + 1
    x_quantized = np.maximum(x_quantized, 1)
    x_quantized = np.minimum(x_quantized, max_nbins)
    
    return x_quantized, minval, maxval, n_bins

def dct_uniform_dequantization(x_quantized, minval, maxval, n_bins):
    x_quantized = x_quantized.reshape(x_quantized.shape[0], -1)
    x = x_quantized / n_bins.reshape(1,-1) * (maxval.reshape(1,-1) - minval.reshape(1,-1)) + minval.reshape(1,-1)
    return x

def img_dct(x, type = 2, nbins = 256, metadata = None):
    new_imgs = dct(dct(x, type = type, axis = 2), type = type, axis = 3)
    new_imgs = new_imgs.reshape(x.shape[0], -1)

    if metadata is None:
        x, minval, maxval, nbins = dct_uniform_quantization(new_imgs, max_nbins = nbins)
        metadata = (minval, maxval, nbins)
    else:
        x, _, _, _ = dct_uniform_quantization(new_imgs, max_nbins = nbins, minval = metadata[0], maxval = metadata[1])

    return x, metadata

def img_idct(x, img_size, metadata, type = 2):
    x = x.reshape(x.shape[0], -1)
    x = dct_uniform_dequantization(x, metadata[0], metadata[1], metadata[2])
    x = x.reshape(x.shape[0], 3, img_size[0], img_size[1])
    return idct(idct(x, type = type, axis = 3), type = type, axis = 2)
"""

function dct_transform(data::Array; precision = 8, metadata = nothing)
    py"img_dct"(data; nbins = 2^precision, metadata)
end

function idct_transform(data::Array; img_size, metadata = nothing)
    py"img_idct"(data; img_size, metadata)
end