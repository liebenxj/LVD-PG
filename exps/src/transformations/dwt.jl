using Random: randperm
using PyCall

py"""
import pywt
import numpy as np
import cv2

def flatten_wavedec(coeffs):
    num_samples = coeffs[0].shape[0]
    flattened_x = np.zeros([num_samples, 0])
    data_format = []
    for scale_idx in range(len(coeffs)):
        if scale_idx == 0:
            data_format.append((coeffs[scale_idx].shape[1:], 1))
            flattened_x = np.concatenate(
                (flattened_x, coeffs[scale_idx].reshape(num_samples, -1)),
                axis = 1
            )
        else:
            data_format.append((coeffs[scale_idx][0].shape[1:], len(coeffs[scale_idx])))
            for i in range(len(coeffs[scale_idx])):
                flattened_x = np.concatenate(
                    (flattened_x, coeffs[scale_idx][i].reshape(num_samples, -1)),
                    axis = 1
                )
    return flattened_x, data_format

def unflatten_wavedec(flattened_x, data_format):
    num_samples = flattened_x.shape[0]
    idx_start = 0
    coeffs = []
    for scale_idx in range(len(data_format)):
        data_size = data_format[scale_idx][0]
        data = []
        for idx in range(data_format[scale_idx][1]):
            idx_end = idx_start + data_size[0] * data_size[1]
            data.append(flattened_x[:,idx_start:idx_end].reshape(num_samples, data_size[0], data_size[1]))
            idx_start = idx_end
        if len(data) == 1:
            coeffs.append(data[0])
        else:
            coeffs.append(data)
    return coeffs

def uniform_quantization(x, max_nbins = 256, minval = None, maxval = None):
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

def uniform_dequantization(x_quantized, minval, maxval, n_bins):
    x_quantized = x_quantized.reshape(x_quantized.shape[0], -1, 3)
    x = x_quantized / n_bins.reshape(1,-1,1) * (maxval.reshape(1,-1,1) - minval.reshape(1,-1,1)) + minval.reshape(1,-1,1)
    return x

def dwt_transform(imgs, wavelet = "bior6.8"):
    assert len(imgs.shape) == 4
    assert imgs.shape[1] == 3
    s = imgs.shape
    imgs = imgs.reshape(s[0] * 3, s[2], s[3])
    
    x_dwt = pywt.wavedec2(imgs, wavelet = wavelet, axes = (1, 2))
    x_dwt, data_format = flatten_wavedec(x_dwt)
    x_dwt = x_dwt.reshape(s[0], -1)
    
    return x_dwt, data_format

def lossy_dwt_transform(imgs, wavelet = "haar", metadata = None, precision = 8):
    imgs = imgs.transpose(0, 2, 3, 1)
    assert len(imgs.shape) == 4
    assert imgs.shape[3] == 3
    s = imgs.shape
    imgs = imgs.transpose(3, 0, 1, 2).reshape(3 * s[0], s[1], s[2])

    x_dwt = pywt.wavedec2(imgs, wavelet = wavelet, axes = (1, 2))
    x_dwt, data_format = flatten_wavedec(x_dwt)
    
    if metadata is None:
        x_quantized, minval, maxval, n_bins = uniform_quantization(x_dwt, max_nbins = 2**precision)
    else:
        x_quantized, minval, maxval, n_bins = uniform_quantization(
            x_dwt, max_nbins = 2**precision, minval = metadata[1], maxval = metadata[2]
        )
    x_quantized = x_quantized.reshape(3, s[0], -1).transpose(1, 2, 0).reshape(s[0], -1)

    return x_quantized, (data_format, minval, maxval, n_bins)

def idwt_transform(x_dwt, data_format, imgshape, wavelet = "bior6.8"):
    s = x_dwt.shape
    x_dwt = x_dwt.reshape(s[0] * 3, -1)
    x_dwt = unflatten_wavedec(x_dwt, data_format)
    x_recon = pywt.waverec2(x_dwt, wavelet = wavelet, axes = (1, 2))
    x_recon = x_recon.reshape(s[0], 3, imgshape[0], imgshape[1])
    return x_recon

def lossy_idwt_transform(x_quantized, imgshape, wavelet = "haar", metadata = None):
    data_format, minval, maxval, n_bins = metadata
    x_dwt = uniform_dequantization(x_quantized, minval, maxval, n_bins)
    s = x_dwt.shape
    x_dwt = x_dwt.reshape(s[0], -1, 3).transpose(2, 0, 1).reshape(3*s[0], -1)
    x_dwt = unflatten_wavedec(x_dwt, data_format)
    x_recon = pywt.waverec2(x_dwt, wavelet = wavelet, axes = (1, 2))
    x_recon = x_recon.reshape(3, -1, imgshape[0] * imgshape[1]).transpose(
        1, 2, 0).reshape(-1, imgshape[0], imgshape[1], 3).transpose(0, 3, 1, 2)
    return x_recon
"""


function dwt_transform(data::Array; wavelet = "bior6.8", precision = 8)
    dwt_img, data_format = py"dwt_transform"(data; wavelet)
    dwt_img = clamp.(UInt32.(round.(dwt_img)), zero(UInt32), UInt32(2^precision-1))
    
    dwt_img, data_format
end


function idwt_transform(data::Array, data_format; img_size, wavelet = "bior6.8")
    img = py"idwt_transform"(Float64.(data), data_format, (img_size[1], img_size[2]); wavelet)

    UInt8.(clamp.(round.(img), zero(Int32), Int32(255)))
end


function lossy_dwt_transform(data::Array; wavelet = "haar", precision = 8, metadata = nothing)
    dwt_img, metadata = py"lossy_dwt_transform"(data; wavelet, metadata, precision)
    dwt_img = clamp.(UInt32.(round.(dwt_img)), zero(UInt32), UInt32(2^precision-1))

    dwt_img, metadata
end


function lossy_idwt_transform(data::Array; img_size, metadata, wavelet = "haar")
    img = py"lossy_idwt_transform"(data; imgshape = (img_size[1], img_size[2]), wavelet, metadata)

    UInt8.(clamp.(round.(img), zero(Int32), Int32(255)))
end


function get_lossless_wavelets()
    lossless_wavelets = ["bior2.8", "bior3.9", "bior6.8", "coif3", "coif4", "coif5", "coif6", "coif7", "coif8", "coif9", 
        "coif10", "coif11", "coif12", "coif13", "coif14", "coif15", "coif16", "coif17", "db9", "db10", "db11", "db12", 
        "db13", "db14", "db15", "db16", "db17", "db18", "db19", "db20", "db21", "db22", "db23", "db24", "db25", "db26", 
        "db27", "db28", "db29", "db30", "db31", "db32", "db33", "db34", "db35", "db36", "db37", "db38", "dmey", "rbio2.8", 
        "rbio3.9", "rbio6.8", "sym9", "sym10", "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20"]
    m = randperm(length(lossless_wavelets))
    lossless_wavelets[m]
end


function sample_lossless_wavelet()
    lossless_wavelets = get_lossless_wavelets()
    m = rand(1:length(lossless_wavelets))
    lossless_wavelets[m]
end


function get_lossy_wavelets()
    lossy_wavelets = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior3.1", "bior3.3", "bior3.5", 
        "bior3.7", "bior4.4", "bior5.5", "coif1", "coif2", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "haar", 
        "rbio1.1", "rbio1.3", "rbio1.5", "rbio2.2", "rbio2.4", "rbio2.6", "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", 
        "rbio4.4", "rbio5.5", "sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8"]
    m = randperm(length(lossy_wavelets))
    lossy_wavelets[m]
end


function sample_lossy_wavelet()
    lossy_wavelets = get_lossy_wavelets()
    m = rand(1:length(lossy_wavelets))
    lossy_wavelets[m]
end