import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib import patches
from math import floor


# Note:
# Given a couple of coordinates x,y we can get the corresponding flat address in 2 different ways:
#  -> 1) i1 = np.ravel_multi_index((y, x), (y_dim, x_dim)) == x + y * x_dim
#        i.e. (y, x) = np.unravel_index(i1, (y_dim, x_dim))
#  -> 2) i2 = np.ravel_multi_index((x, y), (x_dim, y_dim)) == y + x * y_dim
#        i.e. (x, y) = np.unravel_index(i2, (x_dim, y_dim))
# where i1 is different from i2.

# --------------------------------------------- Symmetric Gaussian kernel ----------------------------------------------
def gauss2d(j, sigma, space_dim, threshold=None):
    """Summary: function that calculates elongated gaussian 2D kernel.

    Args:
        j (int): postsynaptic index.
        sigma (float): sigma (SD) of kernel.
        space_dim (tuple or int): number of rows, cols in 2d array of neurons.

    Returns:
        kernel (float 1d-array): value of kernel that can be set as a weight
    """

    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    i = range(space_dim[0] * space_dim[1])
    (iy, ix) = np.unravel_index(i, (space_dim[1], space_dim[0]))
    (jy, jx) = np.unravel_index(j, (space_dim[1], space_dim[0]))
    # i.e. j = jx + jy * space_dim[0]  (the same for i, ix, iy)
    x = ix - jx
    y = iy - jy
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    if threshold:
        kernel[kernel < threshold] = 0
        if threshold >= kernel.max() - 0.15:
            warn('\nAttention: the threshold value %s chosen for the kernel is very high...\n' % threshold)
    return kernel


def show_gauss2d_kernel(space_dim, sigma=15, threshold=None):
    """Show Gaussian kernel in the central spot of the input space."""
    # Define kernel centered in the central point of the pixel array (having squared shape with dimension space_dim)
    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    x_center, y_center = int(space_dim[0] / 2), int(space_dim[1] / 2)
    target = np.ravel_multi_index((y_center, x_center), (space_dim[1], space_dim[0]))
    # i.e. target = x_center + y_center * space_dim[0]
    kernel = gauss2d(target, sigma, space_dim, threshold=threshold)
    kernel = kernel.reshape(space_dim)
    # Show it
    plt.pcolormesh(kernel)
    plt.show()


def compute_dim_gauss2d(space_dim, sigma=15, threshold=None):
    # Define kernel centered in the central point of the pixel array (having squared shape with dimension space_dim)
    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    x_center, y_center = int(space_dim[0] / 2), int(space_dim[1] / 2)
    target = np.ravel_multi_index((y_center, x_center), (space_dim[1], space_dim[0]))
    # i.e. target = x_center + y_center * space_dim[0]
    kernel = gauss2d(target, sigma, space_dim, threshold=threshold)
    kernel = kernel.reshape(space_dim)
    dim_kernel = max(len(kernel[kernel[y_center, :] > 0]), len(kernel[kernel[:, x_center] > 0]))
    return dim_kernel


# --------------------------------------------- Elongated Gaussian kernel ----------------------------------------------
def gauss2d_elong(j, sigma, p, teta, space_dim, threshold=None):
    """Summary: function that calculates elongated gaussian 2D kernel.

    Args:
        j (int): postsynaptic index.
        sigma (float): sigma (SD) of kernel.
        p (float): y_sigma/x_sigma
        teta (float): angle in degrees from vertical
        space_dim (tuple or int): number of rows, cols in 2d array of neurons.

    Returns:
        kernel (float 1d-array): value of kernel that can be set as a weight
    """

    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    i = range(space_dim[0] * space_dim[1])
    (iy, ix) = np.unravel_index(i, (space_dim[1], space_dim[0]))
    (jy, jx) = np.unravel_index(j, (space_dim[1], space_dim[0]))
    x = ix - jx
    y = iy - jy
    x_teta = x*np.cos(np.deg2rad(teta))+y*np.sin(np.deg2rad(teta))
    y_teta = -x*np.sin(np.deg2rad(teta))+y*np.cos(np.deg2rad(teta))
    kernel = np.exp(-((x_teta/p)**2 + y_teta**2) / (2 * sigma**2))
    if threshold:
        kernel[kernel < threshold] = 0
        if threshold >= kernel.max() - 0.15:
            warn('\nAttention: the threshold value %s chosen for the kernel is very high...\n' % threshold)
    return kernel


def show_gauss2d_elong_kernel(space_dim, sigma=5, p=2., teta=45, threshold=None):
    """Show Gaussian kernel in the central spot of the input space."""
    # Define kernel centered in the central point of the pixel array (having squared shape with dimension space_dim)
    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    x_center, y_center = int(space_dim[0] / 2), int(space_dim[1] / 2)
    target = np.ravel_multi_index((y_center, x_center), (space_dim[1], space_dim[0]))
    # i.e. target = x_center + y_center * space_dim[0]
    kernel = gauss2d_elong(target, sigma, p, teta, space_dim, threshold=threshold)
    kernel = kernel.reshape(space_dim)
    # Show it
    plt.pcolormesh(kernel)
    plt.show()


def compute_dim_gauss2d_elong(space_dim, sigma=15, p=2., threshold=None):
    # Define kernel centered in the central point of the pixel array (having squared shape with dimension space_dim)
    # and with a 90° orientation
    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    x_center, y_center = int(space_dim[0] / 2), int(space_dim[1] / 2)
    target = np.ravel_multi_index((y_center, x_center), (space_dim[1], space_dim[0]))
    # i.e. target = x_center + y_center * space_dim[0]
    kernel = gauss2d_elong(target, sigma, p, 90, space_dim, threshold=threshold)
    kernel = kernel.reshape(space_dim)
    dim_kernel = len(kernel[kernel[y_center, :] > 0])
    return dim_kernel


# ------------------------------------------ Elongated ON-OFF Gaussian kernel ------------------------------------------
def bigauss2d_elong(j, sigma, p, teta, space_dim, threshold=2e-1):
    """Summary: function that calculates elongated gaussian 2D kernel.

    Args:
        j (int): postsynaptic index.
        sigma (float): sigma (SD) of kernel.
        p (float): x_sigma/y_sigma
        teta (float): angle in degrees from vertical
        space_dim (tuple or int): number of rows, cols in 2d array of neurons.

    Returns:
        float: value of kernel that can be set as a weight
    """

    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    i = range(space_dim[0] * space_dim[1])
    (iy, ix) = np.unravel_index(i, (space_dim[1], space_dim[0]))
    (jy, jx) = np.unravel_index(j, (space_dim[1], space_dim[0]))
    x = ix - jx
    y = iy - jy
    x_teta = x * np.cos(np.deg2rad(teta)) - y * np.sin(np.deg2rad(teta))
    y_teta = x * np.sin(np.deg2rad(teta)) + y * np.cos(np.deg2rad(teta))
    exponent1 = -((x_teta + sigma) ** 2 + (y_teta / p) ** 2) / (2 * sigma ** 2)
    exponent2 = -((x_teta - sigma) ** 2 + (y_teta / p) ** 2) / (2 * sigma ** 2)
    ex = np.exp(exponent1) - np.exp(exponent2)
    source_ind = np.where(np.logical_or(ex > threshold, ex < -threshold))[0]
    res = np.zeros(space_dim[0] * space_dim[1])
    res[source_ind] = ex[source_ind]
    neg_i = np.where(res < 0)[0]
    neg_w = res[neg_i]
    res[neg_i] = 0
    if teta < 90:
        res[neg_i - 1] = neg_w
    else:
        neg_w = neg_w[neg_i < space_dim[1] * space_dim[0] - space_dim[1] - 1]
        neg_i = neg_i[neg_i < space_dim[1] * space_dim[0] - space_dim[1] - 1]
        res[neg_i + space_dim[0]] = neg_w
    return res


# -------------------------------------------- Elongated Mexican-Hat kernel --------------------------------------------
def mexican2d_elong(j, sigma, p, teta, space_dim, threshold=2e-1):
    """Summary: function that calculates mexican-hat kernel.

    Args:
        j (int): postsynaptic index.
        sigma (float): sigma (SD) of kernel.
        p (float): x_sigma/y_sigma
        teta (float): angle in degrees from vertical
        space_dim (tuple or int): number of rows, cols in 2d array of neurons.

    Returns:
        float: value of kernel that can be set as a weight
    """

    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    i = range(space_dim[0] * space_dim[1])
    (iy, ix) = np.unravel_index(i, (space_dim[1], space_dim[0]))
    (jy, jx) = np.unravel_index(j, (space_dim[1], space_dim[0]))
    x = ix - jx
    y = iy - jy
    sigma_x_2 = sigma * 2
    sigma_x_1 = sigma
    sigma_y_2 = p * sigma_x_2
    sigma_y_1 = p * sigma_x_2
    x_teta = x * np.cos(np.deg2rad(teta)) - y * np.sin(np.deg2rad(teta))
    y_teta = x * np.sin(np.deg2rad(teta)) + y * np.cos(np.deg2rad(teta))
    exponent1 = -(x_teta ** 2 / (2 * sigma_x_1 ** 2) + y_teta ** 2 / (2 * sigma_y_1 ** 2))
    exponent2 = -(x_teta ** 2 / (2 * sigma_x_2 ** 2) + y_teta ** 2 / (2 * sigma_y_2 ** 2))
    ex1 = 1 / np.sqrt(2 * np.pi * sigma_x_1 * sigma_y_1) * np.exp(exponent1)
    ex2 = 1 / np.sqrt(2 * np.pi * sigma_x_2 * sigma_y_2) * np.exp(exponent2)
    ex = ex1 - ex2
    s_pos = np.where(ex > 0)[0]
    ex[s_pos] = ex[s_pos] / max(ex[s_pos])
    s_neg = np.where(ex < 0)[0]
    ex[s_neg] = - 0.7 * ex[s_neg] / min(ex[s_neg])
    kernel = np.zeros(space_dim[0] * space_dim[1])
    source_ind = np.where(np.abs(ex) > threshold)
    kernel[source_ind] = ex[source_ind]
    return kernel


# ----------------------------------------- 2 Distant Elongated Gaussian kernel ----------------------------------------
def distant_gauss2d_elong(j, sigma, d, teta, space_dim, threshold=None):
    """Summary: function that calculates 2 elongated gaussian 2D kernels at distance 'd' and sums them up.

    Args:
        j (int): postsynaptic index.
        sigma (float): sigma (SD) of kernel.
        p (float): x_sigma/y_sigma
        teta (float): angle in degrees from vertical
        space_dim (tuple or int): number of rows, cols in 2d array of neurons.

    Returns:
        float: value of kernel that can be set as a weight
    """

    if type(space_dim) is not tuple:
        assert type(space_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        space_dim = (space_dim, space_dim)
    i = range(space_dim[0] * space_dim[1])
    (iy, ix) = np.unravel_index(i, (space_dim[1], space_dim[0]))
    (jy, jx) = np.unravel_index(j, (space_dim[1], space_dim[0]))
    x = ix - jx
    y = iy - jy
    x_teta = x*np.cos(np.deg2rad(teta))+y*np.sin(np.deg2rad(teta))
    y_teta = -x*np.sin(np.deg2rad(teta))+y*np.cos(np.deg2rad(teta))
    exponent1 = -(x_teta**2 + (y_teta+d)**2) / (2 * sigma**2)
    exponent2 = -(x_teta**2 + (y_teta-d)**2) / (2 * sigma**2)
    kernel = np.exp(exponent1) + np.exp(exponent2)
    if threshold:
        kernel[kernel < threshold] = 0
        if threshold >= kernel.max() - 0.15:
            warn('\nAttention: the threshold value %s chosen for the kernel is very high...\n' % threshold)
    return kernel


# ------------------------------------------- Out space Convolutional layer --------------------------------------------
def convolution_out_shape(space_dim, kernel_size=1, stride=1, padding=0, dilation=1, return_error=True):
    """Utility function for computing output of convolutions:
    it takes a tuple of (height, weight) relative to the input space (together with kernel and convolutional parameters)
    and it returns a tuple of (height, weight) relative to the output space.
    Note that in case of squared input a single integer can be assigned to space_dim and a single integer will be
    returned as out_space_dim.
    Also note that, if 'return_error' flag is set to True, the error of convolution operation (i.e. the remainder from
    the number of times the kernel, with relative stride, can slide on the given input space) is also returned."""

    if type(space_dim) is not tuple and type(kernel_size) is not tuple:
        assert all([type(k) == int for k in [space_dim, kernel_size, stride, padding, dilation]]),\
            'Input space dimension, kernel size, stride, padding and dilation should all be integers.'
        out_shape_exact = ((space_dim + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        out_shape = floor(out_shape_exact)
        error = abs(out_shape - out_shape_exact)
    else:
        if type(space_dim) is not tuple:
            space_dim = (space_dim, space_dim)
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        assert all([type(k) == int for k in
                    [space_dim[0], space_dim[1], kernel_size[0], kernel_size[1], stride, padding, dilation]]),\
            '\nInput space dimension, kernel size, stride, padding and dilation should all be integers.\n'
        out_shape_exact = (((space_dim[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1,
                           ((space_dim[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        out_shape = (floor(out_shape_exact[0]), floor(out_shape_exact[1]))
        error = (abs(out_shape[0] - out_shape_exact[0]), abs(out_shape[1] - out_shape_exact[1]))
    if return_error:
        return out_shape, error
    else:
        return out_shape


def compute_sliding_gauss2d(in_dim, out_dim, kernel_sigma, stride, threshold_kernel=None, return_centers=False):
    if type(in_dim) is not tuple:
        assert type(in_dim) == int, '\nInput space dimension should be an integer or tuple of integers.\n'
        in_dim = (in_dim, in_dim)
    if type(out_dim) is not tuple:
        assert type(out_dim) == int, '\nOutput space dimension should be an integer or tuple of integers.\n'
        out_dim = (out_dim, out_dim)
    if type(stride) is not tuple:
        assert type(stride) == int, '\nStride should be an integer or tuple of integers.\n'
        stride = (stride, stride)
    all_kernels = np.zeros((in_dim[0] * in_dim[1], out_dim[0], out_dim[1]))
    dim_kernel = compute_dim_gauss2d(in_dim, kernel_sigma, threshold_kernel)
    centers = []
    for y, x in np.ndindex(out_dim[1], out_dim[0]):
        y_center, x_center = y * stride[1] + int(round((dim_kernel - 1) / 2)),\
                             x * stride[0] + int(round((dim_kernel - 1) / 2))
        target = np.ravel_multi_index((y_center, x_center), (in_dim[1], in_dim[0]))
        # i.e. target =  x_center + y_center * in_dim[0]
        centers.append([x_center, y_center])
        # TODO: manage different kernel parameters for different kernel types
        all_kernels[:, y, x] = gauss2d(target, kernel_sigma, in_dim, threshold=threshold_kernel)
    if return_centers:
        return all_kernels.reshape(in_dim[0], in_dim[1], out_dim[1], out_dim[0]), centers
    else:
        return all_kernels.reshape(in_dim[0], in_dim[1], out_dim[1], out_dim[0])


def plot_moving_convolution_kernel(kernels, slow_motion=False):
    """Function that takes an ndarray 'kernels' having dimensions (in_size_x, in_size_y, out_size_x, out_size_y) and
    plots the moving kernel while sliding on the input space."""
    if slow_motion:
        interval = 0.8
    else:
        interval = 0.2
    for y, x in np.ndindex(kernels.shape[3], kernels.shape[2]):
        plt.imshow(kernels[:, :, y, x])
        plt.pause(interval)
    plt.close()


def plot_moving_convolution_kernel_on_image(image, kernel_size, centers, slow_motion=False):
    if slow_motion:
        interval = 0.8
    else:
        interval = 0.2
    fig, ax = plt.subplots(1)
    ax.set_title("DVS events accumulator")
    plt.axis('off')
    ax.imshow(image, cmap='gray')
    for center in centers:
        circle = patches.Circle(center, (kernel_size - 1) / 2, color='r', fill=False, linewidth=3)
        ax.add_patch(circle)
        plt.pause(interval)
    plt.close()


def convolve2d(image_in, kernel, stride=1, padding=0):
    dim_in = image_in.shape
    dim_kern = kernel.shape
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))
    # Shape of Output Convolution
    dim_out = (int(((dim_in[0] - dim_kern[0] + 2 * padding) / stride) + 1),
               int(((dim_in[1] - dim_kern[1] + 2 * padding) / stride) + 1))
    image_out = np.zeros(dim_out)
    # Apply Equal Padding to All Sides
    if padding != 0:
        image_padded = np.zeros((dim_in[0] + padding*2, dim_in[1] + padding*2))
        image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image_in
    else:
        image_padded = image_in
    # Iterate through image
    for y in range(dim_in[1]):
        if y > dim_in[1] - dim_kern[1]:
            break
        if y % stride == 0:
            for x in range(dim_in[1]):
                if x > dim_in[0] - dim_kern[0]:
                    break
                try:
                    if x % stride == 0:
                        image_out[x, y] = (kernel * image_padded[x: x + dim_kern[0], y: y + dim_kern[1]]).sum()
                except:
                    break
    return image_out
