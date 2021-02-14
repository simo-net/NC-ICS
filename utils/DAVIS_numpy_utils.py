import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import wraps


# ---------------------------------- utilities for DVS data ----------------------------------
def dvs_control(n_info):
    """Control DVS input is a numpy array with correct shape."""

    def inner_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], np.ndarray) or args[0].shape[1] != n_info:
                info = {6: 'timestamps, x,y addresses, polarities and p_1, p_2 info',
                        4: 'timestamps, x,y addresses and polarities',
                        3: 'timestamps, flatten addresses and polarities'}.get(n_info)
                raise TypeError('DVS input must be a numpy array with shape Nx%d where N is the number of events at '
                                'all %s.' % (n_info, info))
            return func(*args, **kwargs)

        return wrapper

    return inner_function


def dvs_refractory_filter(address, timestamp, polarity=None, refractory=100, return_filtered=False):
    """Given the pixels' addresses and timestamps (and eventually the polarity) of events outputted by the sensor,
     this function applies a refractory filter, removing all those events following the previous one (that is emitted
     from the same pixel) by less than a given refractory period (in microseconds).
     This is useful for avoiding errors when running simulation, in a clock-driven fashion, having a time-step bigger
     than the inter-spike-interval (ISI) of some pixels (i.e. to avoid that some neurons in the network will spike more
     than once during a single time-step of the simulation).
     Args:
        refractory (int, optional): refractory period each pixel should have, in microseconds.
        address (1d-np.array, required): flattened addresses of the spiking neurons.
        timestamp (1d-np.array, required): timestamps of the events.
        polarity (1d-np.array, optional): polarity of the events.
        [Note: all these 3 arrays must have same dimension!]
    Returns:
        The filtered version of all input arrays: if polarity array is specified, outputs are 3, else 2."""
    # Sort events by their pixel address (the input events are supposed to be sorted by their timestamp)
    idx_sort = np.argsort(address, kind='mergesort')
    addr_sort = address[idx_sort]
    ts_sort = timestamp[idx_sort]
    # Count the number of events for each pixel and remove, from sorted arrays, all pixels spiking at most once
    addr_counts, idx_start, counts = np.unique(addr_sort, return_counts=True, return_index=True)
    idx_start_cut = idx_start[counts < 2]
    addr_sort = np.delete(addr_sort, idx_start_cut)
    idx_sort = np.delete(idx_sort, idx_start_cut)
    ts_sort = np.delete(ts_sort, idx_start_cut)
    # Compute ISI of same pixels and save indices of the events following the previous one by less than dt (refractory)
    ts_diff = np.diff(ts_sort)
    i_risk = np.where(abs(ts_diff) <= refractory)[0]
    idx_remove = idx_sort[[i + 1 for i in i_risk if addr_sort[i] == addr_sort[i + 1]]]
    # Remove (filter out) such events, if any, and return the filtered arrays
    address_filt = np.delete(address, idx_remove)
    timestamp_filt = np.delete(timestamp, idx_remove)
    if polarity:
        polarity_filt = np.delete(polarity, idx_remove)
        if return_filtered:
            return timestamp_filt, address_filt, polarity_filt, idx_remove
        return timestamp_filt, address_filt, polarity_filt
    if return_filtered:
        return timestamp_filt, address_filt, idx_remove
    return timestamp_filt, address_filt


def dvs_timereset(dvsnpy):
    """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
    timestamps, this function returns the same array but resetting all timestamps according to the first event.
    Args:
        dvsnpy (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
            represents the timestamps.
    Returns:
        dvs_reset (np.ndarray): events array as dvsnpy but resetting all timestamps according to the first one (so that
            the timestamp of the first event is 0).
    """
    dvsnpy[:, 0] -= dvsnpy[0, 0]
    return dvsnpy


def dvs_cut_timewindow(dvsnpy, start, stop):
    """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
    timestamps, this function returns the same array but without the last events, having timestamps greater than the
    given recording duration.
    Args:
        dvsnpy (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
            represents the timestamps.
        duration (int, required): wanted duration (in us) for the recorded events.
    Returns:
        dvs_cut (np.ndarray): events array as dvsnpy but without events generated after the desired duration.
    """
    idx_keep = np.where(np.logical_and(dvsnpy[:, 0] >= start + dvsnpy[0, 0], dvsnpy[:, 0] <= stop + dvsnpy[0, 0]))[0]
    return dvsnpy[idx_keep]


def dvs_cut_duration(dvsnpy, duration):
    """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
    timestamps, this function returns the same array but without the last events, having timestamps greater than the
    given recording duration.
    Args:
        dvsnpy (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
            represents the timestamps.
        duration (int, required): wanted duration (in us) for the recorded events.
    Returns:
        dvs_cut (np.ndarray): events array as dvsnpy but without events generated after the desired duration.
    """
    return dvsnpy[dvsnpy[:, 0] <= duration + dvsnpy[0, 0]]


def dvs_compute_rateonoff(dvsnpy):
    rate_on = (len(dvsnpy[dvsnpy[:, -1] == 1])) / dvsnpy.shape[0]
    rate_off = 1 - rate_on
    return rate_on, rate_off


def dvs_compute_FRonoff(dvsnpy, N_x, N_y):
    pol_on = (dvsnpy[:, -1] == 1)
    pol_off = np.logical_not(pol_on)
    fr_on = (len(dvsnpy[pol_on])) / (N_x * N_y * (dvsnpy[-1, 0] - dvsnpy[0, 0]) * 10 ** -6)
    fr_off = (len(dvsnpy[pol_off])) / (N_x * N_y * (dvsnpy[-1, 0] - dvsnpy[0, 0]) * 10 ** -6)
    return fr_on, fr_off


def dvs_select_polarity(dvsnpy, polarity='on'):
    pol_on = (dvsnpy[:, -1] == 1)
    pol_off = np.logical_not(pol_on)
    if polarity == 'on':
        return dvs_timereset(dvsnpy[pol_on])
    elif polarity == 'off':
        return dvs_timereset(dvsnpy[pol_off])
    else:
        raise ValueError("The polarity to select can be either 'on' or 'off'.")


@dvs_control(4)
def dvs_cut_pixelarray(dvsnpy, dim, N_x, N_y):
    """Given an array of DVS events, with N rows (number of events) and 4 columns (timestamps, x, y address and
     polarity), this function returns the same array but cutting off the events where x >= dim and y >= dim.
     Args:
         dvsnpy (np.ndarray, required): events array with N rows (number of events) and 4 columns (timestamps, x, y
            coordinates and polarity).
         dim (int, required): number of pixels along the x and y dimensions of the new squared pixel array.
     Returns:
         dvs_cut (np.ndarray): events array as dvsnpy but with no event having x and y >= dim.
    """
    dvs_cut = dvsnpy[np.logical_and(dvsnpy[:, 1] >= int((N_x - dim) / 2), dvsnpy[:, 1] < int((N_x + dim) / 2))]
    dvs_cut = dvs_cut[np.logical_and(dvs_cut[:, 2] >= int((N_y - dim) / 2), dvs_cut[:, 2] < int((N_y + dim) / 2))]
    dvs_cut[:, 1] -= int((N_x - dim) / 2)
    dvs_cut[:, 2] -= int((N_y - dim) / 2)
    if dvs_cut[0, 0]:  # reset to 0 the timestamp of the first event (if necessary)
        dvs_cut[:, 0] -= dvs_cut[0, 0]
    return dvs_cut


@dvs_control(4)
def dvs_flattenaddress(dvsnpy, x_dim, y_dim):
    """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x, y address and
     polarity) it returns an array with same number of rows but 3 columns where each pair of x, y (pixel) coordinates
     is converted to an index that corresponds to a flattened pixel array.
     Args:
         dvsnpy (np.ndarray, required): events array with N rows (number of events) and 4 columns (timestamps, x, y
            coordinates and polarity).
         x_dim (int, required): number of pixels along the x dimension of the pixel array.
         y_dim (int, required): number of pixels along the y dimension of the pixel array.
     Returns:
         dvs_flatten (np.ndarray): events array with converted index (i.e. flattened pixel address).
    """
    dvs_flatten = np.delete(dvsnpy, 2, axis=1)
    dvs_flatten[:, 1] = np.ravel_multi_index((dvsnpy[:, 2], dvsnpy[:, 1]), (y_dim, x_dim))
    # The last line is equivalent to: dvs_flatten[:, 1] = dvsnpy[:, 1] + dvsnpy[:, 2] * x_dim
    return dvs_flatten


def adapt_dtype(array):
    if array.max() < 2 ** 8:
        return array.astype(np.uint8)
    elif array.max() < 2 ** 16:
        return array.astype(np.uint16)
    elif array.max() < 2 ** 32:
        return array.astype(np.uint32)
    else:
        return array.astype(np.uint64)


@dvs_control(3)
def dvs_return_id_and_ts(dvsnpy):
    ts, id = dvsnpy[:, 0], dvsnpy[:, 1]
    return adapt_dtype(ts), adapt_dtype(id)


@dvs_control(3)
def dvs_computevideofreq(dvsnpy, x_dim, y_dim, rate=20, duration=0):
    """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten pixel
    address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
    reconstructed video. Each 2D array (frame) has y_dim rows and x_dim columns where each element represents the
    firing rate of the corresponding pixel during a 1/rate time-window.
    Note: dvsnpy must be time-reset before giving it as argument to this function! Otherwise results won't make sense.
    Args:
        dvsnpy (np.ndarray, required): events array with N rows (number of events) and 3 columns (timestamps, flatten
            pixel address and polarity).
        x_dim (int, required): number of pixels along the x dimension of the pixel array.
        y_dim (int, required): number of pixels along the y dimension of the pixel array.
        rate (float, required): the wanted frame rate (in Hz) for the resulting video.
        duration (float, optional): if it is specified, the entered value is used to compute the firing rate, otherwise
            the duration is computed from dvsnpy.
    Returns:
        video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
            time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
            dimensions of the pixel array).
    """
    if not duration:
        duration = (dvsnpy[-1, 0] - dvsnpy[0, 0]) * 10 ** -6  # seconds
    n_frames = int(round(rate * duration))
    dur_frame = int(round(duration * 10 ** 6 / n_frames))  # micro-seconds
    video = np.zeros((n_frames, y_dim * x_dim))
    for k in range(n_frames):
        id_dvs = dvsnpy[np.logical_and(dvsnpy[:, 0] >= k * dur_frame, dvsnpy[:, 0] < (k + 1) * dur_frame), 1]
        unique, counts = np.unique(id_dvs, return_counts=True)
        fr = dict(zip(unique, counts))
        video[k, list(fr.keys())] = list(fr.values())
    video /= (dur_frame * 10 ** -6)
    # Reshape the video to size (n_frames, y_dim, x_dim)
    return video.reshape((n_frames, y_dim, x_dim))


@dvs_control(3)
def dvs_compute_start(dvs_flat, x_dim, y_dim, duration):
  dvsvid = dvs_computevideofreq(dvs_flat, x_dim, y_dim, rate=50)
  dvsvid = dvsvid.reshape(dvsvid.shape[0], x_dim * y_dim)
  inst_fr = []
  for k in range(dvsvid.shape[0]):
    inst_fr.append(np.sum(dvsvid[k, :]))
  start = dvs_flat[np.argmax(inst_fr)][0]
  if start > 30:
    start -= 30
  return start


# TODO: change this function definition.
@dvs_control(3)
def dvs_computevideofreq_onoff(dvsnpy, x_dim, y_dim, rate):
    """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten pixel
    address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
    reconstructed video. Each 2D array (frame) has y_dim rows and x_dim columns where each element represents the
    firing rate of the corresponding pixel during a 1/rate time-window.
    Args:
        dvsnpy (np.ndarray, required): events array with N rows (number of events) and 3 columns (timestamps, flatten
            pixel address and polarity).
        x_dim (int, required): number of pixels along the x dimension of the pixel array.
        y_dim (int, required): number of pixels along the y dimension of the pixel array.
        rate (float, required): the wanted frame rate (in Hz) for the resulting video.
    Returns:
        video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
            time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
            dimensions of the pixel array).
    """
    dvsnpy = dvs_timereset(dvsnpy)
    duration = dvsnpy[-1, 0] * 10 ** -6  # seconds
    on = dvsnpy[dvsnpy[:, 2] == 1]
    off = dvsnpy[dvsnpy[:, 2] == 0]
    video_on = dvs_computevideofreq(on, x_dim, y_dim, rate, duration=duration)
    video_off = dvs_computevideofreq(off, x_dim, y_dim, rate, duration=duration)
    return video_on, video_off
