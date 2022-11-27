import numpy as np
import matplotlib.pyplot as plt
import cv2


# ======================================= Process event-based data ======================================
class dvs_handler:

    def __init__(self, data: np.ndarray or None, shape: (int, int) = (260, 346), reset_timestamps: bool = True):
        """
        This class helps to handle   event-based data. A set of methods for transforming, taking info and visualizing
        event-based data is available.
        Inputs:
            data (np.ndarray): Events in the form of a Nx4 np.ndarray where N is the total number of events and the
                  columns specify the timestamp, the x and y pixel address, and the polarity of all such events.
            shape ((int, int), optional): The shape of the full pixel array in the form of (height, width). If it is not
                  given, the shape of the DAVIS sensor is taken as default (260,346).
        Protected Attributes:
            N_x, N_y (int, int): Width and height of the camera pixel array.
            data (np.ndarray): Array of events with shape Nx4.
            dt (int): time-step or refractory period (us).
        """
        self.reset_timestamps = reset_timestamps
        self._N_y, self._N_x = shape if shape is not None else (None, None)
        if data is not None:
            self._data = data
            if self.reset_timestamps:
                self.timereset()
        self._dt = None

    @property
    def data(self):
        return self._data

    @property
    def shape_pixel_array(self):
        return self._N_y, self._N_x

    @property
    def ts(self):  # in micro-seconds (us)
        return adapt_dtype(self._data[:, 0])

    @property
    def x(self):
        return adapt_dtype(self._data[:, 1])

    @property
    def y(self):
        return adapt_dtype(self._data[:, 2])

    @property
    def id(self):
        return adapt_dtype(self.flatten_id(y=self._data[:, 2], x=self._data[:, 1]))

    @property
    def pol(self):
        return adapt_dtype(self._data[:, 3])

    @property
    def num_events(self):
        return self._data.shape[0]

    @property
    def duration(self):  # in micro-seconds (us)
        return self._data[-1, 0] - self._data[0, 0]

    @property
    def dt(self):  # in micro-seconds (us)
        if self._dt:
            return self._dt
        else:
            address, timestamp = self.id, self.ts
            # Sort events by their pixel address (the input events are supposed to be sorted by their timestamp)
            idx_sort = np.argsort(address, kind='stable')
            addr_sort, ts_sort = address[idx_sort], timestamp[idx_sort]
            # Find out the pixels firing more than once
            idx_risk = np.where(np.diff(addr_sort) == 0)[0]
            return (ts_sort[idx_risk + 1] - ts_sort[idx_risk]).min()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # ---------------------------------- Basic Utilities -----------------------------------
    def timereset(self, data: np.ndarray or None = None, reference_timestamp: int or None = None) -> np.ndarray:
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but resetting all timestamps according to the first event or
        to a given reference timestamp (if reference_timestamp is not None).
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
                represents the timestamps.
            reference_timestamp (int, optional): the timestamp of a given reference phenomenon by which to reset all the
                timestamps of events.
        Returns:
            (np.ndarray): events array as input data but resetting all timestamps according to the first one (so that
                the timestamp of the first event is 0 if reference_timestamp is None, else it depends by such value).
        """
        if data is not None:
            data_reset = np.copy(data)
            if reference_timestamp is None:
                reference_timestamp = data[0, 0]
            data_reset[:, 0] -= reference_timestamp
            return data_reset
        else:
            if reference_timestamp is None:
                reference_timestamp = self._data[0, 0]
            self._data[:, 0] -= reference_timestamp

    def flatten_id(self, y: np.ndarray or int, x: np.ndarray or int):
        return np.ravel_multi_index((y, x), (self._N_y, self._N_x))  # i.e. x + y * self._N_x

    def expand_id(self, id: np.ndarray or int):
        return np.unravel_index(id, (self._N_y, self._N_x))

    def flatten_data(self):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x, y address
        and polarity) it returns an array with same number of rows but 3 columns where each pair of x, y (pixel)
        coordinates is converted to an index that corresponds to a flattened pixel array.
        """
        flat_data = np.delete(self._data, 2, axis=1)
        try:
            flat_data[:, 1] = self.flatten_id(y=self._data[:, 2], x=self._data[:, 1])
            return flat_data
        except:
            raise ValueError('There is an issue in flattening data, you probably set a wrong shape of the pixel array.')

    # ----------------------------------- Transformations ----------------------------------
    def crop(self, shape: (int, int)):
        """Given an array of DVS events, with N rows (number of events) and 4 columns (timestamps, x, y address and
        polarity), this function returns the same array but cutting off the events where x >= dim and y >= dim.
        Args:
            shape ((int, int), required): number of pixels along the y and x dimensions (h, w) of the cropped array.
        """
        if shape[1] > self._N_x or shape[0] > self._N_y:
            raise ValueError('Cannot crop data with the given dimension.')
        x_start, y_start = (self._N_x - shape[1]) // 2, (self._N_y - shape[0]) // 2
        x_end, y_end = x_start + shape[1], y_start + shape[0]
        mask = np.logical_and(np.logical_and(self._data[:, 1] >= x_start, self._data[:, 1] < x_end),
                              np.logical_and(self._data[:, 2] >= y_start, self._data[:, 2] < y_end))
        self._data = self._data[mask]
        self._data[:, 1] -= x_start
        self._data[:, 2] -= y_start
        self._N_y, self._N_x = shape
        if self.reset_timestamps:
            self.timereset()

    def cut_duration(self, duration: float):
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but without the last events, having timestamps greater than the
        given recording duration.
        Args:
            duration (float, required): wanted duration (in us) for the recorded events.
        """
        self._data = self._data[self._data[:, 0] <= duration + self._data[0, 0]]

    def cut_timewindow(self, start: float, stop: float):
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but without the last events, having timestamps greater than the
        given recording duration.
        Args:
            start (float, required): first moment (in us) for the recorded events.
            stop (float, required): last moment (in us) for the recorded events.
        """
        idx_keep = np.where(np.logical_and(self._data[:, 0] >= start,
                                           self._data[:, 0] <= stop))[0]
        self._data = self._data[idx_keep]
        if self.reset_timestamps:
            self.timereset()

    def select_polarity(self, polarity: str = 'on'):
        """This function selects only events of a specific polarity and removes all the others."""
        pol = {'on': (self.pol == 1), 'off': (self.pol == 0)}.get(polarity)
        if pol is None:
            raise ValueError('The polarity to select can only be "on" or "off"!')
        self._data = self._data[pol]
        if self.reset_timestamps:
            self.timereset()

    # -------------------------------------- Filters ---------------------------------------
    def refractory_filter(self, refractory: float = 100, return_filtered: bool = False):
        """This function applies a refractory filter, removing all events, from the same pixel, coming after the
        previous event by less than a given refractory period (in micro-seconds).
        This is useful for avoiding errors when running simulation, in a clock-driven fashion, having a time-step
        bigger than the inter-spike-interval (ISI) of some pixels (i.e. to avoid that some neurons in the network will
        spike more than once during a single time-step of the simulation).
        Args:
            refractory (float, optional): refractory period each pixel should have, in microseconds (us).
            return_filtered (bool, optional): if True, the indices of filtered events are returned.
        """
        address, timestamp = self.id, self.ts
        # Sort events by their pixel address (events in data array are supposed to be sorted by their timestamp)
        idx_sort = np.argsort(address, kind='stable')
        addr_sort, ts_sort = address[idx_sort], timestamp[idx_sort]
        # Find out the pixels firing more than once
        idx_risk = np.where(np.diff(addr_sort) == 0)[0]
        # Find out the indices of the events to remove (because occurred within the refractory period)
        idx_remove = idx_sort[idx_risk[(ts_sort[idx_risk + 1] - ts_sort[idx_risk]) <= refractory] + 1]
        # Remove (filter out) such events, if any
        if len(idx_remove):
            self._data = np.delete(self._data, idx_remove, axis=0)
            if self.reset_timestamps:
                self.timereset()
        self._dt = refractory
        if return_filtered:
            return idx_remove

    # ------------------------------------ Firing Rates ------------------------------------
    def fraction_onoff(self) -> (float, float):
        """Compute the fraction of ON and OFF events wrt the total number of events."""
        rate_on = (len(self._data[self._data[:, -1] == 1])) / self._data.shape[0]
        rate_off = 1 - rate_on
        return rate_on, rate_off

    def mean_firing_rate(self) -> (float, float):
        """Compute the firing-rate of all events in the full pixel array (mean over the population of pixels)."""
        return self._data.shape[0] / (self._N_x * self._N_y * self.duration * 1e-6)

    def mean_firing_rate_onoff(self) -> (float, float):
        """Compute the firing-rate of ON and OFF events in the full pixel array (mean over the population of pixels)."""
        pol_on = (self._data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        fr_on = (len(self._data[pol_on])) / (self._N_x * self._N_y * self.duration * 1e-6)
        fr_off = (len(self._data[pol_off])) / (self._N_x * self._N_y * self.duration * 1e-6)
        return fr_on, fr_off

    # ------------------------------------- View Spikes ------------------------------------
    def spike_train(self, neuron_id: (int, int) or int, return_id: bool = False) ->\
            (np.ndarray, np.ndarray) or (np.ndarray, np.ndarray, np.ndarray):
        """Return the spike train of a given pixel/neuron."""
        if isinstance(neuron_id, tuple):
            idx = np.logical_and(self.x == neuron_id[1], self.y == neuron_id[0])
        elif np.issubdtype(type(neuron_id), np.integer):
            idx = (self.id == neuron_id)
        else:
            raise TypeError(
                'The given neuron position is not valid, it should be a tuple ([y,x] coordinates) or an integer '
                '(flat index).')
        if return_id:
            return self.ts[idx], self.pol[idx], self.id[idx]
        return self.ts[idx], self.pol[idx]

    def show_spiketrain(self, timestamps: np.ndarray or list or None = None,
                        polarities: np.ndarray or list or None = None,
                        duration: int or None = None, title: str = 'Spike Train',
                        show: bool or float = True, figsize: (int, int) = (12, 2)):
        """Show the spike train of a given pixel/neuron."""
        if timestamps is None:
            timestamps = self.ts
            polarities = self.pol
        if not duration:
            duration = self.duration
        pol_on = (polarities == 1)
        pol_off = np.logical_not(pol_on)
        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plot1, = plt.plot(timestamps[pol_on] * 1e-3, np.ones_like(timestamps[pol_on]), label='ON',
                          marker='|', markersize=20, color='g', linestyle='None')
        plot2, = plt.plot(timestamps[pol_off] * 1e-3, np.ones_like(timestamps[pol_off]), label='OFF',
                          marker='|', markersize=20, color='r', linestyle='None')
        plt.legend(handles=[plt.plot([], ls='-', color=plot1.get_color())[0],
                            plt.plot([], ls='-', color=plot2.get_color())[0]],
                   labels=[plot1.get_label(), plot2.get_label()])
        plt.title(title)
        plt.xlabel('Time (ms)')
        plt.xlim(timestamps[0], timestamps[0] + duration * 1e-3)
        plt.ylim(0.8, 1.2)
        plt.yticks([])
        if isinstance(show, bool):
            if show:
                plt.show()
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_rasterplot(self, data: np.ndarray or None = None, show: bool or float = True,
                        figsize: (int, int) or None = None):
        """Show the raster-plot of the whole activity: the spike train of each pixel."""
        if data is None:
            t, i, p = self.ts, self.id, self.pol
        elif data.shape[1] == 3:
            t, i, p = data[:, 0], data[:, 1], data[:, 2]
        else:
            raise ValueError('Data has wrong shape! It should be Nx3, in the form: [timestamp, flat index, polarity].')
        pol_on = (p == 1)
        pol_off = np.logical_not(pol_on)
        plt.figure(figsize=figsize)
        plt.plot(t[pol_on] * 1e-3, i[pol_on], '|g', markersize=2, label='ON')
        plt.plot(t[pol_off] * 1e-3, i[pol_off], '|r', markersize=2, label='OFF')
        plt.legend(loc='upper right', fontsize=14, markerscale=2.5)
        plt.title('Raster-plot of events')
        plt.xlabel('Time (ms)')
        plt.ylabel('Pixel ID')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    # --------------------------------------- Frames ---------------------------------------
    def frame(self, data: np.ndarray or None = None, clip_value: float = 0.5) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address, and polarity) it returns a 2D array: the reconstructed frame, where each element represents the
        firing rate of the corresponding pixel during the whole duration of recording.
        Returns:
            img (np.ndarray): frame reconstructed from DVS info by accumulating events.
        """
        if data is None:
            data = self._data
        pol_on = (data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        img_on, _, _ = np.histogram2d(data[pol_on, 2], data[pol_on, 1],
                                      bins=(self._N_y, self._N_x), range=[(0, self._N_y), (0, self._N_x)])
        img_off, _, _ = np.histogram2d(data[pol_off, 2], data[pol_off, 1],
                                       bins=(self._N_y, self._N_x), range=[(0, self._N_y), (0, self._N_x)])
        if clip_value:
            img = np.clip((img_on - img_off), -clip_value, clip_value) + clip_value
        else:
            img = (img_on - img_off)
        return img

    def show_frame(self, clip_value: float = 0.5, title: str = "DVS events accumulator",
                   show: bool or float = True, figsize: (int, int) = (5, 4)):
        """Show the reconstructed frame of events obtained accumulating all DVS events in time.
        Args:
            clip_value (float, optional): clip value of normalized array.
            title (str, optional): title of the plot.
            show (bool or float, optional): if bool it determines whether to show or not the window, else it represents
                the time (in seconds) during which the image should be displayed.
            figsize (tuple, optional): size of the window.
        """
        img = self.frame(clip_value=clip_value)
        if clip_value:
            img /= float(clip_value * 2)
        plt.figure(figsize=(figsize[0] + 1.2, figsize[1]))  # + 1.2 along x for colorbar
        plt.imshow(img, cmap='gray')
        plt.title(title)
        cb = plt.colorbar()
        cb.set_label('normalized FR')
        plt.axis('off')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def frame_firingrate(self, data: np.ndarray or None = None, duration: float = 0) -> np.ndarray:
        """This function is the 2D version of the firing_rate_1d() function.
        Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns a 2D array with y_dim rows and x_dim columns where each element
        represents the firing rate (in Hz) of the corresponding pixel during the whole recording.
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and 4 columns (timestamps, x and y
                pixel address and polarity).
            duration (float, optional): in seconds. If it is specified, the entered value is used to compute the firing
                rate, otherwise the duration is computed from data.
        Returns:
            frame (np.ndarray): single frame (with shape (y_dim)x(x_dim)) obtained accumulating all DVS events in time.
        """
        if data is None:
            data = self._data
        if not duration:
            duration = (data[-1, 0] - data[0, 0]) * 1e-6  # seconds
            if duration == 0:
                return np.zeros((self._N_y, self._N_x))
        frame, _, _ = np.histogram2d(data[:, 2], data[:, 1], bins=(self._N_y, self._N_x),
                                     range=[(0, self._N_y), (0, self._N_x)])
        frame = frame / duration
        return frame

    def frame_firingrate_onoff(self, duration: float = 0) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns two (ON/OFF) 2D arrays with N_y rows and N_x columns where each element
        represents the firing rate (in Hz) of the corresponding pixel during the whole recording. If specified, input
        duration must be expressed in seconds.
        Returns:
            frame_on, frame_off (np.ndarray, np.ndarray): two frames (with shape (y_dim)x(x_dim)) obtained accumulating
                all DVS ON/OFF events in time.
        """
        if not duration:
            duration = self.duration * 1e-6  # seconds
            if duration == 0:
                return np.zeros((self._N_y, self._N_x)), np.zeros((self._N_y, self._N_x))
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        frame_on = self.frame_firingrate(self._data[pol_on], duration=duration)
        frame_off = self.frame_firingrate(self._data[pol_off], duration=duration)
        return frame_on, frame_off

    def show_frame_firingrate(self, title: str = "DVS events accumulator",
                              show: bool or float = True, figsize: (int, int) = (6, 4)):
        """Show the single frame obtained accumulating all ON/OFF DVS events in time."""
        frame = self.frame_firingrate()
        plt.figure(figsize=figsize)
        plt.imshow(frame, cmap='gray')
        plt.title(title)
        cb = plt.colorbar()
        cb.set_label('FR (Hz)')
        plt.axis('off')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_frame_firingrate_onoff(self, title: str = "On and Off DVS events accumulator",
                                    show: bool or float = True, figsize: (int, int) = (10, 5)):
        """Show the single frame obtained accumulating all DVS events in time."""
        frame_on, frame_off = self.frame_firingrate_onoff()
        frame_rgb = np.zeros((*frame_on.shape, 3), dtype=int)
        minval, maxval = min(frame_on.min(), frame_off.min()), max(frame_on.max(), frame_off.max())
        frame_rgb[..., 1] = np.uint8(np.interp(frame_on, (minval, maxval), (0, 255)))  # green = ON
        frame_rgb[..., 0] = np.uint8(np.interp(frame_off, (minval, maxval), (0, 255)))  # red = OFF
        plt.figure(figsize=figsize)
        plt.suptitle(title)
        plt.imshow(frame_rgb)
        plt.axis('off')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    # --------------------------------------- Videos ---------------------------------------
    def video(self, refresh: float = 50, clip_value: float = 0.5) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address, and polarity) it returns a 2D array: an array composed of N 2D arrays where N is the number of
        frames in the reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element
        represents the firing rate of the corresponding pixel during a 1/rate time-window.
        """
        duration = self.duration * 1e-6  # seconds
        if duration == 0:
            return np.zeros((1, self._N_y, self._N_x))
        n_frames = int(round(refresh * duration))
        dur_frame = int(round(1e6 / refresh))  # micro-seconds
        video = np.zeros((n_frames, self._N_y, self._N_x))
        for i in range(n_frames):
            events = self._data[np.logical_and(self._data[:, 0] >= i * dur_frame + self._data[0, 0],
                                               self._data[:, 0] < (i + 1) * dur_frame + self._data[0, 0]), :]
            video[i, :, :] = self.frame(data=events, clip_value=clip_value)
        return video

    def show_video(self, refresh: float = 50, clip_value: float = 0.5, title: str = "DVS reconstructed video",
                   position: (int, int) = (0, 0), zoom: float = 1):
        """Show the reconstructed video of events obtained accumulating all DVS events in time-windows of 1/rate.
        Args:
            refresh (int, optional): refresh rate of the video.
            clip_value (float, optional): clip value of normalized array.
            title (str, optional): title of the plot.
            position (tuple, optional): position of the cv2 window.
            zoom (float, optional): how much to zoom the window.
        """
        dt = 0 if not refresh else int(round(1e3 / refresh))
        dvsvid = self.video(refresh=refresh, clip_value=clip_value)
        if clip_value:
            dvsvid /= float(clip_value * 2)
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)
        cv2.destroyAllWindows()

    def video_firingrate(self, data: np.ndarray = None, refresh: float = 50, duration: float = 0) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
        reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element represents the
        firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            data (np.ndarray, optional): events array with N rows (number of events) and 4 columns (timestamps, x and y
                pixel address and polarity). Note: timestamps are should be reset according to the first event!
            refresh (float, optional): the wanted frame rate (in Hz) for the resulting video.
            duration (float, optional): in seconds. If it is specified, the entered value is used to compute the firing
                rate, otherwise the duration is computed from data.
        Returns:
            video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
                time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
                dimensions of the pixel array).
        """
        if data is None:
            data = self.timereset(self._data)
        if not duration:
            duration = data[-1, 0] * 1e-6  # seconds
            if duration == 0:
                return np.zeros((1, self._N_y, self._N_x))
        n_frames = int(round(refresh * duration))
        dur_frame = int(round(1e6 / refresh))  # micro-seconds
        video = np.zeros((n_frames, self._N_y, self._N_x))
        for k in range(n_frames):
            events = data[np.logical_and(data[:, 0] >= k * dur_frame,
                                         data[:, 0] < (k + 1) * dur_frame)]
            video[k, ...] = self.frame_firingrate(data=events, duration=1)  # return spike counts by setting duration=1
        video /= (dur_frame * 1e-6)
        return video

    def video_firingrate_onoff(self, refresh: float = 50) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns two (ON/OFF) arrays composed of N 2D arrays where N is the number of
        frames in the reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element
        represents the firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            refresh (float, required): the wanted frame rate (in Hz) for the resulting video.
        Returns:
            video_on, video_off (np.ndarray, np.ndarray): two videos (with shape Nx(y_dim)x(x_dim), where N is the
                number of frames) obtained accumulating all DVS ON/OFF events in time.
        """
        duration = self.duration * 1e-6  # seconds
        if duration == 0:
            return np.zeros((1, self._N_y, self._N_x)), np.zeros((1, self._N_y, self._N_x))
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        data = self.timereset(self._data)
        video_on = self.video_firingrate(data=data[pol_on], refresh=refresh, duration=duration)
        video_off = self.video_firingrate(data=data[pol_off], refresh=refresh, duration=duration)
        return video_on, video_off

    def show_video_firingrate(self, refresh: float = 50, title: str = 'DVS reconstructed video',
                              position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of DVS-reconstructed frames (obtained accumulating events in a time-window).
        """
        dt = 0 if not refresh else int(round(1e3 / refresh))
        dvsvid = self.video_firingrate(refresh=refresh)
        # Convert each pixel's firing rate to a grayscale value (in range [0, 255])
        dvsvid = np.uint8(np.interp(dvsvid, (dvsvid.min(), dvsvid.max()), (0, 255)))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    def show_video_firingrate_onoff(self, refresh: float = 50, title: str = 'On-Off DVS reconstructed video',
                                    position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of DVS-reconstructed frames (obtained accumulating events in a time-window).
        """
        dt = 0 if not refresh else int(round(1e3 / refresh))
        dvsvid_on, dvsvid_off = self.video_firingrate_onoff(refresh=refresh)
        # Convert each pixel's firing rate to a grayscale value (in range [0, 255]) and create an ON-OFF BGR video
        dvsvid = np.zeros((*dvsvid_on.shape, 3))  # BGR video
        minval, maxval = min(dvsvid_on.min(), dvsvid_off.min()), max(dvsvid_on.max(), dvsvid_off.max())
        dvsvid[..., 1] = np.uint8(np.interp(dvsvid_on, (minval, maxval), (0, 255)))  # green = ON
        dvsvid[..., 2] = np.uint8(np.interp(dvsvid_off, (minval, maxval), (0, 255)))  # red = OFF
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    # ------------------------------------- Utilities --------------------------------------

    def start_stimulus(self) -> int:
        dvsvid = self.video_firingrate(self._data, refresh=60)
        dvsvid = dvsvid.reshape(dvsvid.shape[0], self._N_x * self._N_y)
        inst_fr = [dvsvid[k, :].sum() for k in range(dvsvid.shape[0])]
        return self._data[np.argmax(inst_fr)][0]


def adapt_dtype(array: np.ndarray) -> np.ndarray:
    if array.max() < 2 ** 8:
        return array.astype(np.uint8)
    elif array.max() < 2 ** 16:
        return array.astype(np.uint16)
    elif array.max() < 2 ** 32:
        return array.astype(np.uint32)
    else:
        return array.astype(np.uint64)
