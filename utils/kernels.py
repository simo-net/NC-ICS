import numpy as np


# --------------------------------------------- Symmetric Gaussian kernel ----------------------------------------------
def gauss2d(space: (int, int), center: (int, int), sigma: float,
            threshold: float or None = None) -> np.ndarray:
    """Summary: function that builds a gaussian 2D kernel.

    Args:
        space (int, int): x and y dimensions (respectively) on which the kernel is defined and acts.
        center (int, int): x and y coordinates (respectively) defining the center of the kernel on the space.
        sigma (float): sigma (SD) of the kernel.
        threshold (float): minimum value under which all kernel values should be set to zero.

    Returns:
        kernel (float 1d-array): kernel values as 1D array.
    """
    assert isinstance(space, tuple), 'Input space dimension should be a tuple of integers.'
    (y, x) = np.unravel_index(range(np.prod(space)), (space[1], space[0]))
    kernel = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
    if threshold:
        kernel[kernel < threshold] = 0
    return kernel.reshape(space)


def dim_gauss2d(space: (int, int), sigma: float = 15,
                threshold: float or None = None) -> int:
    assert isinstance(space, tuple), 'Input space dimension should be a tuple of integers.'
    center = (int(space[0] / 2), int(space[1] / 2))
    kernel = gauss2d(space, center, sigma, threshold=threshold)
    dim_kernel = max(sum(kernel[center[1], :] > 0), sum(kernel[:, center[0]] > 0))
    return dim_kernel


# --------------------------------------------- Elongated Gaussian kernel ----------------------------------------------
def gauss2d_elong(space: (int, int), center: (int, int), sigma: float, p: float, teta: float,
                  threshold: float or None = None) -> np.ndarray:
    """Summary: function that builds an elongated gaussian 2D kernel.

    Args:
        space (int, int): x and y dimensions (respectively) on which the kernel is defined and acts.
        center (int, int): x and y coordinates (respectively) defining the center of the kernel on the space.
        sigma (float): sigma (SD) of the kernel.
        p (float): defined as y_sigma/x_sigma.
        teta (float): angle in degrees from vertical.
        threshold (float): minimum value under which all kernel values should be set to zero.

    Returns:
        kernel (float 1d-array): kernel values as 1D array.
    """
    assert isinstance(space, tuple), 'Input space dimension should be a tuple of integers.'
    (y, x) = np.unravel_index(range(np.prod(space)), (space[1], space[0]))
    teta_rad = np.deg2rad(teta)
    x_teta = (x - center[0]) * np.cos(teta_rad) + (y - center[1]) * np.sin(teta_rad)
    y_teta = -(x - center[0]) * np.sin(teta_rad) + (y - center[1]) * np.cos(teta_rad)
    kernel = np.exp(-((x_teta / p)**2 + y_teta**2) / (2 * sigma**2))
    if threshold:
        kernel[kernel < threshold] = 0
    return kernel.reshape(space)


def dim_gauss2d_elong(space: (int, int), sigma: float = 15, p: float = 2,
                      threshold: float or None = None) -> (int, int):
    assert isinstance(space, tuple), 'Input space dimension should be a tuple of integers.'
    center = int(space[0] / 2), int(space[1] / 2)
    kernel = gauss2d_elong(space, center, sigma, p, 90, threshold=threshold)
    dim_kernel = (sum(kernel[center[1], :] > 0), sum(kernel[:, center[0]] > 0))
    return dim_kernel


# ------------------------------------------- Out space Convolutional layer --------------------------------------------
def conv2d_out_shape(space: (int, int), kernel_shape: (int, int),
                     stride: (int, int) = (1, 1), padding: (int, int) = (0, 0), dilation: (float, float) = (1, 1),
                     return_error: bool = False) -> (int, int):
    """Utility function for computing output of convolutions:
    it takes a tuple of (x, y) dimensions relative to the input space (together with kernel and convolution parameters)
    and it returns a tuple of (x, y) dimensions relative to the output space.
    Note that, if 'return_error' flag is set to True, the error of convolution operation (i.e. the remainder from
    the number of times the kernel, with relative stride, can slide on the given input space) is also returned."""
    assert len(space) == len(kernel_shape) == len(stride) == len(padding) == len(dilation)
    out_shape, error = np.array([conv1d_out_shape(space[k], kernel_shape[k], stride[k], padding[k], dilation[k], True)
                                 for k in range(len(space))]).swapaxes(0, 1)
    out_shape = tuple(out_shape.astype(int))
    if return_error:
        return out_shape, tuple(error)
    return out_shape


def conv1d_out_shape(space: int, kernel_size: int,
                     stride: int, padding: int, dilation: float,
                     return_error: bool = False) -> int or (int, float):
    out_size_exact = ((space + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    out_size = int(np.floor(out_size_exact))
    if return_error:
        err = abs(out_size - out_size_exact)
        return out_size, err
    return out_size
