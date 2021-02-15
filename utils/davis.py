import numpy as np


def adapt_dtype(array):
    if array.max() < 2 ** 8:
        return array.astype(np.uint8)
    elif array.max() < 2 ** 16:
        return array.astype(np.uint16)
    elif array.max() < 2 ** 32:
        return array.astype(np.uint32)
    else:
        return array.astype(np.uint64)


class dvs_handler:

    def __init__(self, data, shape):
        self.N_x, self.N_y = shape
        self.data = self.timereset(data)
        self.flat_data = self.flattenaddress()

    @property
    def ts(self):
        return adapt_dtype(self.flat_data[:, 0])

    @property
    def id(self):
        return adapt_dtype(self.flat_data[:, 1])

    @property
    def x(self):
        return adapt_dtype(self.data[:, 1])

    @property
    def y(self):
        return adapt_dtype(self.data[:, 2])

    @property
    def pol(self):
        return adapt_dtype(self.flat_data[:, 2])

    def timereset(self, data=None) -> np.ndarray:
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but resetting all timestamps according to the first event.
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
                represents the timestamps.
        Returns:
            data (np.ndarray): events array as input data but resetting all timestamps according to the first one (so
                that the timestamp of the first event is 0).
        """
        if data is not None:
            data[:, 0] -= data[0, 0]
            return data
        else:
            self.data[:, 0] -= self.data[0, 0]
            self.flat_data[:, 0] -= self.flat_data[0, 0]

    def flattenaddress(self):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x, y address
        and polarity) it returns an array with same number of rows but 3 columns where each pair of x, y (pixel)
        coordinates is converted to an index that corresponds to a flattened pixel array.
        """
        dvs_flatten = np.delete(self.data, 2, axis=1)
        dvs_flatten[:, 1] = np.ravel_multi_index((self.data[:, 2], self.data[:, 1]), (self.N_y, self.N_x))
        # The last line is equivalent to: dvs_flatten[:, 1] = dvsnpy[:, 1] + dvsnpy[:, 2] * x_dim
        return dvs_flatten

    def select_polarity(self, polarity='on'):
        pol_on = (self.data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        if polarity == 'on':
            self.data = self.data[pol_on]
            self.flat_data = self.flat_data[pol_on]
            self.timereset()
        elif polarity == 'off':
            self.data = self.data[pol_off]
            self.flat_data = self.flat_data[pol_off]
            self.timereset()
        else:
            raise ValueError("The polarity to select can be either 'on' or 'off'.")

    def cut_pixelarray(self, dim: int):
        """Given an array of DVS events, with N rows (number of events) and 4 columns (timestamps, x, y address and
        polarity), this function returns the same array but cutting off the events where x >= dim and y >= dim.
        Args:
            dim (int, required): number of pixels along the x and y dimensions of the new squared pixel array.
        """
        dvs_cut = self.data[np.logical_and(self.data[:, 1] >= int((self.N_x - dim) / 2),
                                           self.data[:, 1] < int((self.N_x + dim) / 2))]
        dvs_cut = dvs_cut[np.logical_and(dvs_cut[:, 2] >= int((self.N_y - dim) / 2),
                                         dvs_cut[:, 2] < int((self.N_y + dim) / 2))]
        dvs_cut[:, 1] -= int((self.N_x - dim) / 2)
        dvs_cut[:, 2] -= int((self.N_y - dim) / 2)
        if dvs_cut[0, 0]:  # reset to 0 the timestamp of the first event (if necessary)
            dvs_cut[:, 0] -= dvs_cut[0, 0]
        self.N_x, self.N_y = dim, dim
        self.data = dvs_cut
        self.flat_data = self.flattenaddress()
        self.timereset()

    def cut_timewindow(self, start: int, stop: int):
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but without the last events, having timestamps greater than the
        given recording duration.
        Args:
            start (int, required): first moment (in us) for the recorded events.
            stop (int, required): last moment (in us) for the recorded events.
        """
        idx_keep = np.where(np.logical_and(self.data[:, 0] >= start + self.data[0, 0],
                                           self.data[:, 0] <= stop + self.data[0, 0]))[0]
        self.data = self.data[idx_keep]
        self.flat_data = self.flat_data[idx_keep]
        self.timereset()

    def cut_duration(self, duration: int):
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but without the last events, having timestamps greater than the
        given recording duration.
        Args:
            duration (int, required): wanted duration (in us) for the recorded events.
        """
        self.data = self.data[self.data[:, 0] <= duration + self.data[0, 0]]
        self.flat_data = self.flat_data[self.flat_data[:, 0] <= duration + self.flat_data[0, 0]]

    def rate_onoff(self) -> (float, float):
        rate_on = (len(self.data[self.data[:, -1] == 1])) / self.data.shape[0]
        rate_off = 1 - rate_on
        return rate_on, rate_off

    def FR_onoff(self) -> (float, float):
        pol_on = (self.data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        fr_on = (len(self.data[pol_on])) / (self.N_x * self.N_y * (self.data[-1, 0] - self.data[0, 0]) * 10 ** -6)
        fr_off = (len(self.data[pol_off])) / (self.N_x * self.N_y * (self.data[-1, 0] - self.data[0, 0]) * 10 ** -6)
        return fr_on, fr_off

    def start_stimulus(self):
        dvsvid = self.video_freq(self.flat_data, rate=60)
        dvsvid = dvsvid.reshape(dvsvid.shape[0], self.N_x * self.N_y)
        inst_fr = [dvsvid[k, :].sum() for k in range(dvsvid.shape[0])]
        start = self.flat_data[np.argmax(inst_fr)][0]
        if start > 30:
            start -= 30
        return start

    def video_freq(self, data=None, rate=50, duration=0):
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten pixel
        address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
        reconstructed video. Each 2D array (frame) has y_dim rows and x_dim columns where each element represents the
        firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            data (np.ndarray, optional): events array with N rows (number of events) and 3 columns (timestamps, flatten
                pixel address and polarity).
            rate (float, optional): the wanted frame rate (in Hz) for the resulting video.
            duration (float, optional): if it is specified, the entered value is used to compute the firing rate, otherwise
                the duration is computed from dvsnpy.
        Returns:
            video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
                time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
                dimensions of the pixel array).
        """
        if data is None:
            data = self.flat_data
        if not duration:
            duration = (data[-1, 0] - data[0, 0]) * 10 ** -6  # seconds
        n_frames = int(round(rate * duration))
        dur_frame = int(round(duration * 10 ** 6 / n_frames))  # micro-seconds
        video = np.zeros((n_frames, self.N_y * self.N_x))
        for k in range(n_frames):
            id_dvs = data[np.logical_and(data[:, 0] >= k * dur_frame, data[:, 0] < (k + 1) * dur_frame), 1]
            unique, counts = np.unique(id_dvs, return_counts=True)
            fr = dict(zip(unique, counts))
            video[k, list(fr.keys())] = list(fr.values())
        video /= (dur_frame * 10 ** -6)
        # Reshape the video to size (n_frames, y_dim, x_dim)
        return video.reshape((n_frames, self.N_y, self.N_x))

    def video_freq_onoff(self, rate=50):
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten
        pixel address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
        reconstructed video. Each 2D array (frame) has y_dim rows and x_dim columns where each element represents the
        firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            rate (float, required): the wanted frame rate (in Hz) for the resulting video.
        Returns:
            video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
                time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
                dimensions of the pixel array).
        """
        data = self.timereset(self.flat_data)
        duration = data[-1, 0] * 10 ** -6  # seconds
        on = data[data[:, 2] == 1]
        off = data[data[:, 2] == 0]
        video_on = self.video_freq(data=on, rate=rate, duration=duration)
        video_off = self.video_freq(data=off, rate=rate, duration=duration)
        return video_on, video_off

    def refractory_filter(self, refractory=100, return_filtered=False):
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
        address, timestamp = self.id, self.ts
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
        self.data = np.delete(self.data, idx_remove, axis=0)
        self.flat_data = np.delete(self.flat_data, idx_remove, axis=0)
        self.timereset()
        if return_filtered:
            return idx_remove
