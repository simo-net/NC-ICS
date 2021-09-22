import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import cv2
import time


# ======================================= Process frame-based data ======================================
class aps_handler:

    def __init__(self, data: np.ndarray or None):
        """
        This class helps handling frame-based data. A set of methods for transforming, taking info and visualizing
        single frame or video is available.
        Inputs:
            data (np.ndarray): Video in the form of a N*Nx*Ny np.ndarray where N is the total number of frames and the
                other dimensions match the shape of the pixel array.
        Attributes:
            N_x, N_y (int, int): Width and height of the camera pixel array.
            data (np.ndarray): Array of frames with shape N*Nx*Ny.
            camera_mtx (np.ndarray): 2D array with shape (N_y,N_x) keeping the intrinsic parameters of the camera.
            distortion_coeffs (np.ndarray): 1D array keeping the distortion coefficients given by lens' optical effects.
        """
        if data is not None:
            self.load_array(data)
        self.camera_mtx = None
        self.distortion_coeffs = None

    @property
    def num_frames(self):
        return self.data.shape[0]

    @property
    def duration(self):
        # Underestimated duration of recording (40 Hz is the maximum frame rate)
        return self.data.shape[0] / 40

    # ------------------------------------ Update data -------------------------------------
    def load_file(self, file: str):
        """Load (and update) data from a NPY file.
        Args:
            file (str, required): full path of the file where events were recorded.
        """
        assert isinstance(file, str) and file is not None,\
            'You must specify a string as full path for loading event-based data.'
        extension = file.split('.')[-1]
        if extension == 'npy':
            self.data = np.load(file)
        else:
            raise ValueError('Type of file not supported. It must be an aedat4 or a npy file.')
        self.N_y, self.N_x = self.data[0].shape[:2]

    def load_array(self, data: np.ndarray):
        """Load (and update) data from a numpy Nd-array.
        Args:
            data (np.ndarray): Video in the form of a N*Nx*Ny np.ndarray where N is the total number of frames and the
                other dimensions match the shape of the pixel array.
        """
        self.data = data
        self.N_y, self.N_x = self.data[0].shape[:2]

    # ---------------------------------- Undistort data ------------------------------------
    def load_xml_calibration(self, file: str, sensor_model: str = 'DAVIS346', serial: str = '00000336'):
        """Load open-cv xml calibration file of the neuromorphic camera, where the camera matrix and distortion
        coefficients are stored. The camera matrix keeps the intrinsic parameters of the camera, thus once calculated
        it can be stored for future purposes. It includes information like focal length (f_x, f_y), optical
        centers (c_x, c_y), etc (it takes the form of a 3x3 matrix like [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).
        Distortion coefficients are 4, 5, or 8 elements (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]); if the vector is
        NULL/empty, zero distortion coefficients are assumed.
        Args:
            file (str): the full path of the xml file.
            sensor_model (str): the model of the neuromorphic sensor.
            serial (str): the serial number of the sensor.
        """
        file_read = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
        cam_tree = file_read.getNode(sensor_model + '_' + serial)
        self.camera_mtx = np.array(cam_tree.getNode('camera_matrix').mat())
        self.distortion_coeffs = np.array(cam_tree.getNode('distortion_coefficients').mat()).T
        file_read.release()

    def _map_undistortion(self, shape: (int, int) = (260, 346)) -> (np.ndarray, np.ndarray):
        """Compute the undistortion maps."""
        if self.camera_mtx is None or self.distortion_coeffs is None:
            raise ValueError('If you wish to undistort the frame-based data, you must call the method '
                             'load_xml_calibration() first, specifying the file where calibration info are stored '
                             '(camera matrix and lens distortion coefficients).')
        rotation = np.eye(3, dtype=int)
        map1, map2 = cv2.initUndistortRectifyMap(self.camera_mtx, self.distortion_coeffs, R=rotation,
                                                 newCameraMatrix=self.camera_mtx, size=shape[::-1],
                                                 m1type=cv2.CV_32FC1)
        return map1, map2

    def _remap(self, map1: np.ndarray, map2: np.ndarray, interpolate: bool = False):
        """Remap frames for removing radial and tangential lens distortion. It takes as input the undistortion maps and
        updates frames. We remove info falling on those pixels for which there is no correspondence in the
        original pixel array."""
        if interpolate:
            dst_video = np.zeros_like(self.data)
            for k, frame in enumerate(self.data):
                dst_video[k] = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            self.data = dst_video
        else:
            self.data = self.data[:, np.round(map2).astype(int), np.round(map1).astype(int)]

    def undistort(self, shape: (int, int) = (260, 346), interpolate: bool = False):
        """Compute undistortion maps and remap frames accordingly in order to correct lens effects. Note that this
        method should be called BEFORE cropping.
        Args:
             shape (tuple, optional): number of pixels along the y and x dimensions of the new pixel array.
             interpolate (bool, optional): whether to smoothen the result through interpolation.
        """
        self._remap(*self._map_undistortion(shape=shape), interpolate=interpolate)

    # ------------------------------------- Cut data ---------------------------------------
    def crop(self, shape: (int, int)):
        """Given an array of APS frames (each frame having number of rows y_dim and number of columns x_dim), this
        method returns the same array but with each frame having the given shape.
        Args:
            shape (tuple, required): number of pixels along the y and x dimensions of the new pixel array.
        """
        if shape[1] > self.N_x or shape[0] > self.N_y:
            raise ValueError('Cannot crop data with the given shape.')
        x_start, y_start = int((self.N_x - shape[1]) / 2), int((self.N_y - shape[0]) / 2)
        x_end, y_end = int((self.N_x + shape[1]) / 2), int((self.N_y + shape[0]) / 2)
        self.data = self.data[:, y_start:y_end, x_start:x_end]
        self.N_y, self.N_x = shape

    def crop_square(self, dim: int):
        """Given an array of APS frames (each frame having number of rows y_dim and number of columns x_dim), this
        method returns the same array but with each frame having squared (cut) size (number of rows and columns equals
        to dim).
        Args:
            dim (int, required): number of pixels along the x and y dimensions of the new squared pixel array.
        """
        if dim > self.N_x or dim > self.N_y or dim < 0:
            raise ValueError('Cannot crop data with the given dimension.')
        x_start, y_start = int((self.N_x - dim) / 2), int((self.N_y - dim) / 2)
        x_end, y_end = int((self.N_x + dim) / 2), int((self.N_y + dim) / 2)
        self.data = self.data[:, y_start:y_end, x_start:x_end]
        self.N_x, self.N_y = dim, dim

    def crop_region(self, start: (int, int), end: (int, int)):
        """Given a region of interest of the pixel array and an array of APS frames (each frame having number of rows
        y_dim and number of columns x_dim), this method returns the same array but cutting off info outside of the
        given region.
        Args:
             start (tuple, required): starting point (x, y) of the selected region.
             end (tuple, required): ending point (x, y) of the selected region.
        """
        x_start, x_end = start[0], end[0]
        # Note: when defining the y coordinate of the starting point, consider that y=0 is on top,
        # while y=N_y-1 is at the bottom. For this reason we do the following:
        y_start, y_end = self.N_y - end[1], self.N_y - start[1]
        self.data = self.data[:, y_start:y_end, x_start:x_end]
        self.N_x, self.N_y = (x_end - x_start), (y_end - y_start)

    def cut_duration(self, duration: int, refresh: float = 40.):
        """Given an array of APS frames and the relative timestamps, this function returns the same arrays but without the
        last frames (and timestamps), having timestamps greater than the given recording duration.
        Args:
            duration (int, required): wanted duration (in us) for the recorded frames.
            refresh (float, optional): refresh rate of the sensor.
        """
        dt = 10 ** 6 / refresh
        timestamps = np.linspace(0, dt * self.data.shape[0], self.data.shape[0], dtype=int)
        self.data = self.data[timestamps <= duration, ...]

    # ------------------------------------- Show frames ------------------------------------
    def show_frame(self, num_frame: int = 0, title: str or None = None,
                   show: bool = True, figsize: (int, int) = (5, 4)):
        """Show a single frame.
        Args:
            num_frame (int, optional): the frame to show.
            title (str, optional): title of the plot.
            show (bool, optional): whether to show or not the window.
            figsize (tuple, optional): size of the window.
        """
        if not title:
            title = 'APS frame ' + str(num_frame)
        plt.figure(figsize=figsize)
        plt.imshow(self.data[num_frame], cmap='gray')
        plt.title(title)
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

    def show_video(self, refresh: float = 40., title: str = 'APS video',
                   position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of APS frames.
        Args:
            refresh (int, optional): refresh rate of the video.
            title (str, optional): title of the plot.
            position (tuple, optional): position of the cv2 window.
            zoom (float, optional): how much to zoom the window.
        """
        dt = 0 if not refresh else int(round(10 ** 3 / refresh))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self.N_x * zoom), int(self.N_y * zoom))
        for frame in self.data:
            frame = cv2.resize(frame, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, frame)
            cv2.waitKey(dt)
        cv2.destroyAllWindows()


# ======================================= Process event-based data ======================================
class dvs_handler:

    def __init__(self, data: np.ndarray or None, shape: (int, int) = (260, 346), reset_timestamps: bool = True):
        """
        This class helps handling event-based data. A set of methods for transforming, taking info and visualizing
        event-based data is available.
        Inputs:
            data (np.ndarray): Events in the form of a Nx4 np.ndarray where N is the total number of events and the
                  columns specify the timestamp, the x and y pixel address, and the polarity of all such events.
            shape ((int, int), optional): The shape of the full pixel array in the form of (height, width). If it is not
                  given, the shape of the DAVIS sensor is taken as default (260,346).
        Protected Attributes:
            N_x, N_y (int, int): Width and height of the camera pixel array.
            data (np.ndarray): Array of events with shape Nx4.
            camera_mtx (np.ndarray): 2D array with shape (N_y,N_x) keeping the intrinsic parameters of the camera.
            distortion_coeffs (np.ndarray): 1D array keeping the distortion coefficients given by lens' optical effects.
            dt (int): time-step or refractory period (us).
        """
        self.reset_timestamps = reset_timestamps
        self._N_y, self._N_x = shape if shape is not None else (None, None)
        if data is not None:
            self._data = data
            if self.reset_timestamps:
                self.timereset()
        self._camera_mtx = None
        self._distortion_coeffs = None
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

    # ------------------------------------ Update data -------------------------------------
    def load_file(self, file: str, shape: (int, int) = (260, 346)):
        """Load (and update) data from a NPY file.
        Args:
            file (str): full path of the file where events were recorded.
            shape (int, int): The shape of the full pixel array in the form of (height, width).
                Note that y-dimension must be specified first!
        """
        self.N_y, self.N_x = shape
        self._dt = None
        self.flat_data = None
        assert isinstance(file, str) and file is not None,\
            'You must specify a string as full path for loading event-based data.'
        extension = file.split('.')[-1]
        if extension == 'npy':
            self.data = self.timereset(np.load(file))
        else:
            raise ValueError('Type of file not supported. It must be an aedat4 or a npy file.')
        self.__flatten_data()

    def load_array(self, data: np.ndarray, shape: (int, int) = (260, 346)):
        """Load (and update) data from a numpy Nd-array.
        Args:
            data (np.ndarray): Events in the form of a Nx4 np.ndarray where N is the total number of events and the
                columns specify the timestamp, the x and y pixel address, and the polarity of all such events.
            shape (int, int): The shape of the full pixel array in the form of (height, width).
                Note that y-dimension must be specified first!
        """
        self.N_y, self.N_x = shape
        self._dt = None
        self.flat_data = None
        self.data = self.timereset(data)
        self.__flatten_data()

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

    def reformat_data(self, polarity: str or None = None) -> np.ndarray or (np.ndarray, np.ndarray):
        """This function returns 2 arrays (data_time and data_pol) if polarity is None, else only one array (data_time).
        Such arrays will have same shape as the pixel matrix (N_y, N_x) and will contain, at each pixel location, some
        information of the spikes emitted by such pixel. Specifically, the data_time array has, in position (y,x), a
        list of spike-times relative to the pixel in position y,x of the pixel array. Similarly, the data_pol array
        (only returned if polarity is None) has, in position (y,x), a list of polarities of the events generated by such
        pixel (these polarities are ordered as the spike times in the previous array)."""
        pol = {None: np.ones(self._data.shape[0], dtype=bool),
               'on': (self.pol == 1), 'off': (self.pol == 0)}.get(polarity)
        if pol is None:
            raise ValueError('The polarity parameter can only be "on", "off" or None!')
        # Sort events by their pixel address (the input events are supposed to be sorted by their timestamp)
        address = self.id[pol]
        idx_sort = np.argsort(address, kind='stable')
        addr_sort, ts_sort, pol_sort = address[idx_sort], self.ts[pol][idx_sort], self.pol[pol][idx_sort]
        # Find which pixels fire at least once, how many times they do and where such info is stored in the sorted array
        addr_unique, index, count = np.unique(addr_sort, return_index=True, return_counts=True)
        # For each pixel location, put a list of its spike times. If a pixel never fires, its value will be None.
        data_time = np.empty(self._N_y * self._N_x, dtype=list)
        data_time[addr_unique] = list(map(lambda tup: list(ts_sort[tup[0]: tup[0] + tup[1]]), zip(index, count)))
        if polarity is None:
            # For each pixel location, put a list of its spikes' polarities. If a pixel never fires, there will be None.
            data_pol = np.empty(self._N_y * self._N_x, dtype=list)
            data_pol[addr_unique] = list(map(lambda tup: list(pol_sort[tup[0]: tup[0] + tup[1]]), zip(index, count)))
            return data_time.reshape(self._N_y, self._N_x), data_pol.reshape(self._N_y, self._N_x)
        else:
            return data_time.reshape(self._N_y, self._N_x)

    # -------------------------------- Camera Undistortion ---------------------------------
    def load_xml_calibration(self, file: str, sensor_model: str = 'DAVIS346', serial: str = '00000336'):
        """Load open-cv xml calibration file of the neuromorphic camera, where the camera matrix and distortion
        coefficients are stored. The camera matrix keeps the intrinsic parameters of the camera, thus once calculated
        it can be stored for future purposes. It includes information like focal length (f_x, f_y), optical
        centers (c_x, c_y), etc (it takes the form of a 3x3 matrix like [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).
        Distortion coefficients are 4, 5, or 8 elements (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]); if the vector is
        NULL/empty, zero distortion coefficients are assumed.
        Args:
            file (str, required): the full path of the xml file.
            sensor_model (str, optional): the model of the neuromorphic sensor.
            serial (str, optional): the serial number of the sensor.
        """
        assert os.path.isfile(file), 'The given path is not a file.'
        assert file[-3:] == 'xml', 'Type of calibration file must be xml.'
        file_read = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
        cam_tree = file_read.getNode(sensor_model + '_' + serial)
        self._camera_mtx = np.array(cam_tree.getNode('camera_matrix').mat())
        self._distortion_coeffs = np.array(cam_tree.getNode('distortion_coefficients').mat()).T
        file_read.release()

    def _undistortion_maps(self) -> (np.ndarray, np.ndarray):
        """Compute the undistortion maps."""
        if self._camera_mtx is None or self._distortion_coeffs is None:
            raise ValueError('If you wish to undistort the event-based data, you must call the method '
                             'load_xml_calibration() first, specifying the file where calibration info are stored '
                             '(camera matrix and lens distortion coefficients).')
        map_1, map_2 = cv2.initUndistortRectifyMap(self._camera_mtx, self._distortion_coeffs, R=np.eye(3, dtype=int),
                                                   newCameraMatrix=self._camera_mtx, size=(self._N_x, self._N_y),
                                                   m1type=cv2.CV_32FC1)
        return np.round(map_1).astype(int), np.round(map_2).astype(int)

    def _remap(self, map1: np.ndarray, map2: np.ndarray):
        """Remap events for removing radial and tangential lens distortion. It takes as input the undistortion maps and
        updates events by means of the 2D (N_y x N_x) numpy array having a list of spikes as single element."""
        # Compute 2d array of events and undistort it
        data_time, data_pol = self.reformat_data(polarity=None)
        data_time_undistort, data_pol_undistort = data_time[map2, map1], data_pol[map2, map1]
        # Find unique flat address (undistorted) of active pixels, with their timestamps and polarities
        id_unique = np.ravel_multi_index(np.where(data_time_undistort != None), (self._N_y, self._N_x))
        ts_unique = data_time_undistort.reshape(self._N_y * self._N_x)[id_unique].tolist()
        pol_unique = data_pol_undistort.reshape(self._N_y * self._N_x)[id_unique].tolist()
        # Compute full undistorted address, timestamps and polarities
        id_undistort = np.repeat(id_unique, repeats=[len(k) for k in ts_unique])
        y_undistort, x_undistort = np.unravel_index(id_undistort, (self._N_y, self._N_x))
        ts_undistort = np.array(list(chain.from_iterable(ts_unique)))
        pol_undistort = np.array(list(chain.from_iterable(pol_unique)))
        # Pack them in new undistorted data afters sorting by events' timestamps
        idx_sort = np.argsort(ts_undistort, kind='stable')
        self._data = np.stack((ts_undistort[idx_sort],
                               x_undistort[idx_sort], y_undistort[idx_sort],
                               pol_undistort[idx_sort])).T
        if self.reset_timestamps:
            self.timereset()

    def undistort(self):
        """Compute undistortion maps and remap events accordingly in order to correct lens effects.
        Note that this method should be always called BEFORE cropping!!
        """
        self._remap(*self._undistortion_maps())

    # -------------------------------------- Filters ---------------------------------------
    def refractory_filter(self, refractory: float = 100, return_filtered: bool = False, verbose: bool = False):
        """This function applies a refractory filter, removing all events, from the same pixel, coming after the
        previous event by less than a given refractory period (in micro-seconds).
        This is useful for avoiding errors when running simulation, in a clock-driven fashion, having a time-step
        bigger than the inter-spike-interval (ISI) of some pixels (i.e. to avoid that some neurons in the network will
        spike more than once during a single time-step of the simulation).
        Args:
            refractory (float, optional): refractory period each pixel should have, in microseconds (us).
            return_filtered (bool, optional): if True, the indices of filtered events are returned.
            verbose (bool, optional): if True, some information on the filtering are printed out.
        """
        t0 = time.time()
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
        if verbose:
            print('\nRefractory-period filtering took', round(time.time() - t0, 2), 'seconds.')
            print('How many events were filtered out?', len(idx_remove))
            dt_removed, sameid_removed = [], []
            for k in idx_remove:
                t_removed, i_removed = timestamp[k], address[k]
                where_id = np.where(address == i_removed)[0]
                where_pre = where_id[where_id < k][-1]
                dt_removed.append(t_removed - timestamp[where_pre])
                sameid_removed.append(i_removed == address[where_pre])
            correct_id, correct_dt = all(sameid_removed), all(np.asarray(dt_removed) <= refractory)
            print('Are all events removed correct?', all([correct_dt, correct_id]))
            print('    - is dt of removed event wrt the previous one always <',
                  round(refractory * 1e-3), 'ms?', correct_dt)
            print('    - is the previous event emitted by the same pixel as the removed event?', correct_id, '\n')

    def denoising_filter(self, time_window: float = 1e4):
        """Drops events that are 'not sufficiently connected to other events' in the recording.
        In practise that means that an event is dropped if no other event occurred within a spatial neighbourhood
        of 1 pixel and a temporal neighbourhood of time_window (us). Useful to filter noisy recordings.
        Args:
            time_window (float, optional): The maximum temporal distance (in micro-seconds) to the next event,
                otherwise it is dropped. Lower values will therefore mean more events will be filtered out.
            # space_window (int, optional): number of pixels defining the radius of the spatial neighbourhood.
        """
        # X, Y = np.meshgrid(range(self._N_y), range(self._N_x))
        data_filt = np.zeros(self._data.shape, dtype=self._data.dtype)
        timestamp_memory = np.zeros((self._N_y, self._N_x), dtype=self._data.dtype) + time_window
        idx = 0
        for event in self._data:
            x, y, t = int(event[1]), int(event[2]), event[0]

            # # TODO: try adding a space window
            # # Remove next lines
            # neighbors_x, neighbors_y = np.where(np.round(np.sqrt((Y - y) ** 2 + (X - x) ** 2)) <= space_window)
            # neighbour_events = timestamp_memory[neighbors_y, neighbors_x]
            # if neighbour_events.size:  # apply filtering only at pixels far from the border of the sensor array
            #     if np.median(neighbour_events) > t:
            #         data_filt[idx] = event
            #         idx += 1
            # else:
            #     data_filt[idx] = event
            #     idx += 1

            # Keep from here
            if ((x > 0 and timestamp_memory[y, x - 1] > t) or (y > 0 and timestamp_memory[y - 1, x] > t)
                    or (x < self._N_x - 1 and timestamp_memory[y, x + 1] > t)
                    or (y < self._N_y - 1 and timestamp_memory[y + 1, x] > t)):
                data_filt[idx] = event
                idx += 1

            timestamp_memory[y, x] = t + time_window
        self._data = data_filt[:idx]
        if self.reset_timestamps:
            self.timereset()

    def find_hot_pixels(self, space_window: int = 3, polarity: str or None = None):
        """Find the most active pixels (risky hot pixels, such that their spike count is higher than mean+4*std on whole
        pixel array) and then check the activity of their neighbours in a given space window (default is 3x3).
        If the activity of the neighbours is also very high (higher than mean+std), then discard such pixels as hot
        pixels, else confirm that they are actually hot pixels and return their y,x address. Hot pixels can also be
        discriminated by the polarity of the events they generate.
        Note: Remember that the space window parameter should be an odd integer and do not set a high value otherwise it
        will be difficult to find hot pixels (5 is a good value, anyhow better not higher than 11)."""
        pixels_activity = {None: self.frame_firingrate(duration=1), 'on': self.frame_firingrate_onoff(duration=1)[0],
                           'off': self.frame_firingrate_onoff(duration=1)[1]}.get(polarity)
        if pixels_activity is None:
            raise ValueError('The polarity parameter can only be "on", "off" or None!')
        pixels_activity = pixels_activity.astype(int)
        pixels_activity_pad = np.pad(pixels_activity, pad_width=((space_window // 2, space_window // 2),
                                                                 (space_window // 2, space_window // 2)),
                                     mode='constant')
        risky_pixels_y, risky_pixels_x = np.where(pixels_activity >
                                                  pixels_activity.mean() + 4*pixels_activity.std())
        max_act = pixels_activity.mean() + 2 * pixels_activity.std()
        hot_pixel_x, hot_pixel_y = [], []
        for y, x in zip(risky_pixels_y, risky_pixels_x):
            neighbour_activity = pixels_activity_pad[y: y + space_window,
                                                     x: x + space_window]
            neighbour_activity[space_window // 2, space_window // 2] = 0
            mean_activity = np.sum(neighbour_activity) / (space_window ** 2 - 1)
            if mean_activity <= max_act:
                hot_pixel_x.append(x)
                hot_pixel_y.append(y)
        return np.array(hot_pixel_y), np.array(hot_pixel_x)

    def hot_pixels_filter(self, hot_pixels: (np.ndarray, np.ndarray) or (int, int) = None, polarity: str = None,
                          space_window: int = 3):
        """Remove events generated by hot pixels, identified by their y,x coordinates. You can also discriminate between
        ON and OFF events, thus filtering out only ON (or OFF) hot pixels (i.e. pixels generating too many ON/OFF events
        wrt their neighbourhood)."""
        if hot_pixels is None:
            hot_pixels = self.find_hot_pixels(polarity=polarity, space_window=space_window)
        pol = {None: np.ones(self._data.shape[0], dtype=bool),
               'on': (self.pol == 1), 'off': (self.pol == 0)}.get(polarity)
        # Find indices of events generated by hot pixels
        if len(hot_pixels[0]) == 0:
            return None
        elif len(hot_pixels[0]) == 1:
            mask = np.logical_and(self._data[pol, 1] == hot_pixels[1], self._data[pol, 2] == hot_pixels[0])
        else:
            find_hot = lambda loc: np.logical_and(self._data[pol, 1] == loc[0], self._data[pol, 2] == loc[1])
            mask = np.sum(np.array(list(map(find_hot, zip(hot_pixels[1], hot_pixels[0])))), axis=0, dtype=bool)
        # Remove them
        if polarity is None:
            self._data = self._data[~mask]
        else:
            self._data = np.vstack((self._data[pol][~mask], self._data[~pol]))
            self._data = self._data[self._data[:, 0].argsort(kind='stable')]
        if self.reset_timestamps:
            self.timereset()
           
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

    def crop_region(self, start: (int, int), end: (int, int)):
        """Given a region of interest of the pixel array and an array of DVS events, with N rows (number of events) and
        4 columns (timestamps, x, y address and polarity), this function returns the same array but cutting off the
        events outside of the given region.
        Args:
             start (tuple, required): starting point (x, y) of the selected region.
             end (tuple, required): ending point (x, y) of the selected region.
        """
        x_start, x_end = start[0], end[0]
        # Note: when defining the y coordinate of the starting point, consider that y=0 is on top,
        # while y=N_y-1 is at the bottom. For this reason we do the following:
        y_start, y_end = start[1], end[1]  # self._N_y - end[1], self._N_y - start[1]
        self._data = self._data[np.logical_and(self._data[:, 1] >= x_start,
                                               self._data[:, 1] < x_end)]
        self._data = self._data[np.logical_and(self._data[:, 2] >= y_start,
                                               self._data[:, 2] < y_end)]
        self._data[:, 1] -= x_start
        self._data[:, 2] -= y_start
        self._N_x, self._N_y = (x_end - x_start), (y_end - y_start)
        if self.reset_timestamps:
            self.timereset()

    def show_cropping(self, shape: (int, int)):
        x_start, y_start = int((self._N_x - shape[1]) / 2), int((self._N_y - shape[0]) / 2)
        _, ax = plt.subplots(1)
        plt.imshow(self.frame_firingrate(), cmap='gray')
        rect = plt.Rectangle((x_start, y_start), shape[1], shape[0], color='r', fill=False, linewidth=1, linestyle='-')
        ax.add_patch(rect)
        plt.show()

    def show_cropping_region(self, start: (int, int), end: (int, int)):
        x_start, y_start = start
        width, height = end[0] - start[0], end[1] - start[1]
        _, ax = plt.subplots(1)
        plt.imshow(self.frame_firingrate(), cmap='gray')
        rect = plt.Rectangle((x_start, y_start), width, height, color='r', fill=False, linewidth=1, linestyle='-')
        ax.add_patch(rect)
        plt.show()

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

    # -------------------------------- Other Transformations -------------------------------
    def flip_polarity(self):
        """Flip the polarity of events. ON becomes OFF and vice versa."""
        self._data[:, 3] = np.logical_not(self._data[:, 3]).astype(int)

    def merge_polarities(self):
        """Merge both polarities into ON events only."""
        self._data[self._data[:, 3] == 0, 3] = 1

    def reverse_time(self):
        """Reverse the timing of events."""
        self._data[:, 0] = np.abs(self._data[:, 0] - self._data[-1, 0])

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

    def firing_rate_pop_distribution(self, firing_rate: bool = False) -> np.ndarray:
        """Return the firing rate of each neuron in the pixel-array (as a 1d vector where the position represents the
        flatten id of each pixel), i.e. compute the firing rate distribution over the population of neurons/pixels."""
        hist, _ = np.histogram(self.id, bins=self._N_y * self._N_x, range=(0, self._N_y * self._N_x))
        if firing_rate:
            return hist / (self.duration * 1e-6)
        return hist

    def show_firing_rate_pop_distribution(self, firing_rate: bool = False, fit: bool = False,
                                          title: str = 'Activity of each pixel', show: bool = True):
        """Show the firing rate distribution over the population of neurons/pixels."""
        hist = self.firing_rate_pop_distribution(firing_rate=firing_rate)
        if fit:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(self.id)
            density.covariance_factor = lambda: .05
            density._compute_covariance()
            fitted = density(range(0, len(hist)))
            fitted = (fitted - fitted.min()) / (fitted.max() - fitted.min()) * (hist.max() - hist.min()) + hist.min()
        plt.figure()
        plt.plot(hist)
        if fit:
            plt.plot(fitted)
        plt.title(title)
        plt.xlabel('Pixel ID')
        plt.ylabel({True: 'Firing Rate (Hz)', False: 'Spike Count'}.get(firing_rate))
        if show:
            plt.show()

    def activity_distribution(self, firing_rate: bool = False) -> (np.ndarray, np.ndarray):
        """Compute the the number of neurons with the same spike count (or firing rate) and return 2 arrays (with same
        shape) representing the number of spikes (or firing rate) and the corresponding number of neurons."""
        hist = self.firing_rate_pop_distribution(firing_rate=firing_rate)
        num_spikes, num_neurons = np.unique(hist, return_counts=True)
        return num_spikes, num_neurons

    def show_activity_distribution(self, firing_rate: bool = False, show: bool = True,
                                   title: str = 'Activity Distribution:\nfraction of pixels with different activities'):
        """Plot the number of neurons sharing the same firing activity (in terms of spike count or firing rate)."""
        num_spikes, num_neurons = self.activity_distribution(firing_rate=firing_rate)
        plt.figure()
        plt.plot(num_spikes, num_neurons / (self._N_y * self._N_x) * 100)
        plt.title(title)
        plt.ylabel('# of Pixels (%)')
        plt.xlabel({True: 'Firing Rate (Hz)', False: 'Spike Count'}.get(firing_rate))
        if show:
            plt.show()

    # --------------------- Instantaneous Firing Rates (single pixel) ----------------------
    def ifr_tot(self, timestamps: np.ndarray or None = None, num_neurons: int = 0,
                    duration: float = 0, bin_size: float = 10e3, smooth: bool = True) -> np.ndarray:
        """This function computes the instantaneous firing rate of a single neuron or entire population of neurons (in
        such case you should specify the total number of neurons) taking as input its spike times.
        By default (no spike times given), the function computes the IFR from all spikes in the entire population.
        The temporal window for averaging the spiking activity is also taken as input (in us). If the bool smooth
        parameter is True, a gaussian window function will be used, else a rectangular function."""
        if timestamps is None:
            timestamps = self.timereset(self._data)[:, 0]
        if not duration:
            duration = self.duration
        if not num_neurons:
            num_neurons = self._N_x * self._N_y
        timestamps = timestamps[timestamps <= duration]
        n_steps = int(round(duration/bin_size))
        ifr, _ = np.histogram(timestamps, bins=n_steps, range=(0, duration))
        if smooth:
            gauss = np.exp(-np.arange(-n_steps//2+1, n_steps//2+1) ** 2 / 2)
            return convolve(ifr, gauss, mode='constant')
        return ifr / (bin_size * 1e-6 * num_neurons)

    def ifr_tot_onoff(self, duration: float = 0, bin_size: float = 10e3,
                      smooth: bool = True) -> (np.ndarray, np.ndarray):
        """This function computes the instantaneous firing rate of all spikes in the population of pixels/neurons. The
        temporal window (or bin size) for averaging the spiking activity is taken as input (in us). If the bool smooth
        parameter is True, a gaussian window function will be used, else a rectangular function."""
        if not duration:
            duration = self.duration
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        ts = self.timereset(self._data)[:, 0]
        ifr_on = self.ifr_tot(timestamps=ts[pol_on], duration=duration, bin_size=bin_size, smooth=smooth)
        ifr_off = self.ifr_tot(timestamps=ts[pol_off], duration=duration, bin_size=bin_size, smooth=smooth)
        return ifr_on, ifr_off

    def show_ifr_tot(self, ifr: np.ndarray or None = None, duration: float or None = None, bin_size: float = 10e3,
                     smooth: bool = True, show: bool or int = True, figsize: (int, int) or None = None):
        """Visualise a given instantaneous firing rate."""
        if not duration:
            duration = self.duration
        start = 0
        if ifr is None:
            ifr = self.ifr_tot(duration=duration, bin_size=bin_size, smooth=smooth)
            start = self._data[0, 0]
        time = np.linspace(0, duration, len(ifr)) + start
        plt.figure(figsize=figsize)
        plt.plot(time * 1e-3, ifr)
        plt.title('Instantaneous Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('IFR (Hz)')
        if isinstance(show, bool):
            if show:
                plt.show()
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_ifr_tot_onoff(self, duration: float or None = None, bin_size: float = 10e3, smooth: bool = True,
                           show: bool or int = True, figsize: (int, int) or None = None):
        """Visualise a given instantaneous firing rate."""
        if not duration:
            duration = self.duration
        ifr_on, ifr_off = self.ifr_tot_onoff(duration=duration, bin_size=bin_size, smooth=smooth)
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        time_on = np.linspace(0, duration, len(ifr_on)) + self._data[pol_on, 0][0]
        time_off = np.linspace(0, duration, len(ifr_off)) + self._data[pol_off, 0][0]
        plt.figure(figsize=figsize)
        plt.plot(time_on * 1e-3, ifr_on, label='ON')
        plt.plot(time_off * 1e-3, ifr_off, label='OFF')
        plt.legend()
        plt.title('Instantaneous Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('IFR (Hz)')
        if isinstance(show, bool):
            if show:
                plt.show()
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    # ----------------------- Instantaneous Firing Rates (all pixels) ----------------------
    def ifr_video(self, data: np.ndarray = None, duration: float = 0, bin_size: float = 10e3,
                  smooth: bool = True) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
        reconstructed IFR video. Each 2D array (frame) has N_y rows and N_x columns where each element represents the
        firing rate of the corresponding pixel during a time window of bin_size.
        Args:
            data (np.ndarray, optional): events array with N rows (number of events) and 4 columns (timestamps, x and y
                pixel address and polarity). Note: timestamps are should be reset according to the first event!
            bin_size (float, optional): the size of each time bin in micro-seconds.
            duration (float, optional): in seconds. If it is specified, the entered value is used to compute the firing
                rate, otherwise the duration is computed from data.
            smooth (bool, optional): whether to smoothen the resulting values in time with a gaussian kernel having an
                std of bin_size.
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
        n_frames = int(round(duration / bin_size)) if bin_size < duration else 1
        video = np.zeros((n_frames, self._N_y, self._N_x))
        for k in range(n_frames):
            events = data[np.logical_and(data[:, 0] >= k * bin_size,
                                         data[:, 0] < (k + 1) * bin_size)]
            video[k, ...] = self.frame_firingrate(data=events, duration=1)  # return spike counts by setting duration=1
            # # Note: this is equivalent as doing the following
            # pixel_location = self.flat_data[np.logical_and(self.flat_data[:,0]>=k*bin_size,
            #                                                self.flat_data[:,0]<(k+1)*bin_size),1]
            # idx, counts = np.unique(pixel_location, return_counts=True)
            # video[k, idx] = counts
        # video = video.reshape((n_frames, self.N_y, self.N_x))
        video /= (bin_size * 1e-6)
        if smooth:
            gauss = np.exp(-np.arange(-n_frames//2+1, n_frames//2+1) ** 2 / 2)
            return convolve(video, gauss[:, None, None], mode='constant')
        return video

    def ifr_video_onoff(self, duration: float = 0, bin_size: float = 50,
                        smooth: bool = True) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns two (ON/OFF) arrays composed of N 2D arrays where N is the number of
        frames in the reconstructed IFR video. Each 2D array (frame) has N_y rows and N_x columns where each element
        represents the firing rate of the corresponding pixel during a time window of bin_size.
        Args:
            bin_size (float, optional): the size of each time bin in micro-seconds.
            duration (float, optional): in seconds. If it is specified, the entered value is used to compute the firing
                rate, otherwise the duration is computed from data.
            smooth (bool, optional): whether to smoothen the resulting values in time with a gaussian kernel having an
                std of bin_size.
        Returns:
            video_on, video_off (np.ndarray, np.ndarray): two videos (with shape Nx(y_dim)x(x_dim), where N is the
                number of frames) obtained accumulating all DVS ON/OFF events in time.
        """
        if not duration:
            duration = self.duration * 1e-6  # seconds
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        data = self.timereset(self._data)
        video_on = self.ifr_video(data=data[pol_on], duration=duration, bin_size=bin_size, smooth=smooth)
        video_off = self.ifr_video(data=data[pol_off], duration=duration, bin_size=bin_size, smooth=smooth)
        return video_on, video_off

    def show_ifr_video(self, duration: float or None = None, bin_size: float = 10e3, smooth: bool = True,
                       title: str = 'Video of IFR', position: (int, int) = (0, 0), zoom: float = 1,
                       refresh: float = 50):
        """Visualise the instantaneous firing rate video."""
        if not duration:
            duration = self.duration
        dt = 0 if not refresh else int(round(1e3 / refresh))
        ifr = self.ifr_video(duration=duration, bin_size=bin_size, smooth=smooth)
        # Convert each pixel's firing rate to a grayscale value (in range [0, 255])
        dvsvid = np.uint8(np.interp(ifr, (ifr.min(), ifr.max()), (0, 255)))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for frame in dvsvid:
            frame = cv2.resize(frame, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, frame)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    def show_ifr_video_onoff(self, duration: float or None = None, bin_size: float = 10e3, smooth: bool = True,
                             title: str = 'Video of On-Off IFR', position: (int, int) = (0, 0), zoom: float = 1,
                             refresh: float = 50):
        """Visualise ON and OFF instantaneous firing rate videos superimposed (green=ON, red=OFF)."""
        if not duration:
            duration = self.duration
        dt = 0 if not refresh else int(round(1e3 / refresh))
        ifr_on, ifr_off = self.ifr_video_onoff(duration=duration, bin_size=bin_size, smooth=smooth)
        # Convert each pixel's firing rate to a grayscale value (in range [0, 255]) and create an ON-OFF BGR video
        dvsvid = np.zeros((*ifr_on.shape, 3))  # BGR video
        minval, maxval = min(ifr_on.min(), ifr_off.min()), max(ifr_on.max(), ifr_off.max())
        dvsvid[..., 1] = np.uint8(np.interp(ifr_on, (minval, maxval), (0, 255)))  # green = ON
        dvsvid[..., 2] = np.uint8(np.interp(ifr_off, (minval, maxval), (0, 255)))  # red = OFF
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    # -------------------------------- Inter Spike Intervals -------------------------------
    def isi_frame(self, polarity: str = None, duration: float = 0) -> np.ndarray:
        """Reconstruct frame of inter-spike intervals obtained taking the mean ISI of each pixel firing more than once.
        The ISI value of all other pixels is set to the whole duration of DVS recording.
        Args:
            polarity (str, optional): The polarity of the data to select, either 'on' or 'off' (if None, all polarities
                are considered).
            duration (float, optional): in micro-seconds. If it is specified, the entered value is used, otherwise the
                duration is computed from data.
        Returns:
            frame (np.ndarray, np.ndarray): ISI frame (with shape (y_dim)x(x_dim)).
        """
        pol = {None: np.ones(self._data.shape[0], dtype=bool),
               'on': (self.pol == 1), 'off': (self.pol == 0)}.get(polarity)
        if pol is None:
            raise ValueError('The polarity parameter can only be "on", "off" or None!')
        if not duration:
            duration = self._data[pol][-1, 0] - self._data[pol][0, 0]
        # Sort events by their pixel address (the input events are supposed to be sorted by their timestamp)
        idx_sort = np.argsort(self.id[pol], kind='stable')
        addr_sort, ts_sort, pol_sort = self.id[pol][idx_sort], self.ts[pol][idx_sort], self.pol[pol][idx_sort]
        # Find which pixels fire at least once, how many times they do and where such info is stored in the sorted array
        addr_unique, index, count = np.unique(addr_sort, return_index=True, return_counts=True)
        # For each pixel location, put its mean inter-spike interval. If a pixel fires less than twice, its value will
        # be equal to the entire duration of recording.
        frame_isi = np.ones(self._N_y * self._N_x, dtype=int) * duration
        frame_isi[addr_unique[count > 1]] = list(map(lambda tup:
                                                     np.round(np.mean(np.diff((ts_sort[tup[0]: tup[0] + tup[1]])))),
                                                     zip(index[count > 1], count[count > 1])))
        return frame_isi.reshape((self._N_y, self._N_x))

    def isi_frame_onoff(self, duration: float = 0) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address and polarity) it returns two (ON/OFF) 2D arrays with N_y rows and N_x columns where each element
        represents the firing rate (in Hz) of the corresponding pixel during the whole recording. If specified, input
        duration must be expressed in seconds.
        Args:
            duration (float, optional): in micro-seconds. If it is specified, the entered value is used, otherwise the
                duration is computed from data.
        Returns:
            frame_on, frame_off (np.ndarray, np.ndarray): two (ON/OFF) ISI frames (with shape (y_dim)x(x_dim)).
        """
        if not duration:
            duration = self.duration
            if duration == 0:
                return np.zeros((self._N_y, self._N_x)), np.zeros((self._N_y, self._N_x))
        frame_on = self.isi_frame(polarity='on', duration=duration)
        frame_off = self.isi_frame(polarity='off', duration=duration)
        return frame_on, frame_off

    def show_isi_frame(self, title: str = "Mean Inter-Spike Interval of DVS pixels",
                       show: bool = True, figsize: (int, int) = (5, 4)):
        """Show the reconstructed frame of ISI events obtained taking the mean inter-spike interval of each pixel firing
        more than once. The ISI value of all other pixels is set to the whole duration of DVS recording.
        Args:
            title (str, optional): title of the plot.
            show (bool, optional): whether to show or not the window.
            figsize (tuple, optional): size of the window.
        """
        plt.figure(figsize=(figsize[0] + 1.2, figsize[1]))  # + 1.2 along x for colorbar
        plt.imshow(self.isi_frame() * 1e-3, cmap='gray')
        plt.title(title)
        cb = plt.colorbar()
        cb.set_label('ISI value (ms)')
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

    def show_isi_frame_onoff(self, title: str = "Mean Inter-Spike Interval of DVS pixels",
                             show: bool or float = True, figsize: (int, int) = (10, 5)):
        """Show the reconstructed frame of ISI events obtained taking the mean inter-spike interval of each pixel firing
        an ON/OFF spike more than once. The ISI value of all other pixels is set to the whole duration of DVS recording.
        Args:
            title (str, optional): title of the plot.
            show (bool or float, optional): if bool it determines whether to show or not the window, else it represents
                the time (in seconds) during which the image should be displayed.
            figsize (tuple, optional): size of the window.
        """
        img_on, img_off = self.isi_frame_onoff()
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)
        im0 = ax[0].imshow(img_on * 1e-3, cmap='gray')
        cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.04)
        cb0.set_label('ISI value (ms)', labelpad=-2)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        im1 = ax[1].imshow(img_off * 1e-3, cmap='gray')
        cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.04)
        cb1.set_label('ISI value (ms)', labelpad=-2)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def isi_histogram(self, bin_size: float = 1e3, polarity: str = None,
                      duration: float = 0) -> (np.ndarray, np.ndarray):
        """Compute the inter-spike-interval histogram, i.e. the number of times each ISI value occurs.
        Args:
            bin_size (float, optional): The bin size for the resulting histogram (in micro-seconds).
            polarity (str, optional): The polarity of the data to select, either 'on' or 'off' (if None, all polarities
                are considered).
            duration (float, optional): in micro-seconds. If it is specified, the entered value is used, otherwise the
                duration is computed from data.
        Returns:
            bins (np.ndarray, np.ndarray): ISI values (in micro-seconds).
            hist (np.ndarray, np.ndarray): occurrence of such ISI values (counts).
        """
        pol = {None: np.ones(self._data.shape[0], dtype=bool),
               'on': (self.pol == 1), 'off': (self.pol == 0)}.get(polarity)
        if pol is None:
            raise ValueError('The polarity parameter can only be "on", "off" or None!')
        if not duration:
            duration = self._data[pol][-1, 0] - self._data[pol][0, 0]
        # Sort events by their pixel address (the input events are supposed to be sorted by their timestamp)
        idx_sort = np.argsort(self.id[pol], kind='stable')
        addr_sort, ts_sort, pol_sort = self.id[pol][idx_sort], self.ts[pol][idx_sort], self.pol[pol][idx_sort]
        # Compute ISI of sorted events (remove ISI of events generated by different pixels)
        idx_samepix = (np.diff(addr_sort) == 0)
        # bins, hist = np.unique(np.diff(ts_sort)[idx_samepix], return_counts=True)  # TODO: check this
        hist, bins = np.histogram(np.diff(ts_sort)[idx_samepix],
                                  bins=int(duration/bin_size), range=(0, duration))  # 1kHz sampling
        return bins[:-1], hist / bin_size

    def isi_histogram_onoff(self, bin_size: float = 1e3, duration: float = 0) -> (np.ndarray, np.ndarray, np.ndarray):
        """Compute the inter-spike-interval histogram of ON/OFF events, i.e. the number of times each ISI value occurs
        for ON (and OFF) events.
        Args:
            bin_size (float, optional): The bin size for the resulting histogram (in micro-seconds).
            duration (float, optional): in micro-seconds. If it is specified, the entered value is used, otherwise the
                duration is computed from data.
        Returns:
            bins (np.ndarray, np.ndarray): ISI values (in micro-seconds).
            hist (np.ndarray, np.ndarray): occurrence of such ISI values (counts).
        """
        if not duration:
            duration = self.duration
            if duration == 0:
                return np.zeros((self._N_y, self._N_x)), np.zeros((self._N_y, self._N_x))
        _, hist_on = self.isi_histogram(polarity='on', duration=duration, bin_size=bin_size)
        isi, hist_off = self.isi_histogram(polarity='off', duration=duration, bin_size=bin_size)
        return isi, hist_on, hist_off

    def show_isi_histogram(self, bin_size: float = 1e3, title: str = "ISI histogram",
                           show: bool = True, figsize: (int, int) = (5, 4)):
        """Show the inter-spike-interval histogram, i.e. the number of times each ISI value occurs."""
        isi, hist = self.isi_histogram(bin_size=bin_size)
        plt.figure(figsize=figsize)
        plt.plot(isi * 1e-3, hist)
        plt.title(title)
        plt.ylabel('# of occurrences')
        plt.xlabel('ISI (ms)')
        if show:
            plt.show()

    def show_isi_histogram_onoff(self, bin_size: float = 1e3, title: str = "ISI histogram",
                                 show: bool = True, figsize: (int, int) = (5, 4)):
        """Show the inter-spike-interval histogram of ON/OFF events, i.e. the number of times each ISI value occurs
        for ON (and OFF) events."""
        isi, hist_on, hist_off = self.isi_histogram_onoff(bin_size=bin_size)
        plt.figure(figsize=figsize)
        plt.plot(isi * 1e-3, hist_on, label='ON')
        plt.plot(isi * 1e-3, hist_off, label='OFF')
        plt.legend()
        plt.title(title)
        plt.ylabel('# of occurrences')
        plt.xlabel('ISI (ms)')
        if show:
            plt.show()

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

    # ------------------------------------ Time Surfaces -----------------------------------
    def time_surface(self, data: np.ndarray or None = None, dt: float or None = None, duration: float = 0,
                     verbose: bool = False) -> np.ndarray:
        """This function returns a 3D matrix of binary values. The first dimension is equal to the number of time-steps
        (dt) that are present in the whole duration; second and third dimensions are equal to the resolution of the
        sensor (number of pixels along y and x respectively). The matrix is made of all zeros except for ones in
        correspondence to the time-step and the location of an event.
        Args:
            data (np.ndarray): array of events with shape (N, 4) where N is the number of events. Note: timestamps
                should have been reset according to the first event!
            dt (float, required): the time-step in micro-seconds (you should consider applying the refractory filter
                with such refractory value before calling this method). Default value: 10000 us.
            duration (float, optional): the duration of recording/simulation.
            verbose (bool, optional): whether to print out some information.
        Return:
            (np.ndarray): the time surface made of binary values.
        """
        if data is None:
            data = self.timereset(self._data)
        if not dt:
            # If a refractory filter was applied, the time-step is set to the refractory period, else to 10ms
            dt = self._dt if self._dt else 1e4
        if not duration:
            duration = data[-1, 0]
        else:
            data = data[data[:, 0] <= duration]
        N_t = int(np.ceil(duration / dt))
        res = np.zeros((N_t, self._N_y, self._N_x), dtype=bool)
        for t in range(N_t):
            idx = np.where(np.logical_and(data[:, 0] >= t * dt,
                                          data[:, 0] < (t+1) * dt))[0]
            if not len(idx):
                continue
            res[t, data[idx, 2], data[idx, 1]] = 1
        if verbose:
            n_evts_surf = len(np.where(res > 0)[0])
            correct = {True: 'exact', False: 'approximate (%d events have been removed)'
                                             % (data.shape[0] - n_evts_surf)}.get(n_evts_surf == data.shape[0])
            print('\nThe time surface reconstructed from the events is {}.\n'.format(correct))
        return res

    def time_surface_onoff(self, dt: float or None = None, duration: float = 0) -> (np.ndarray, np.ndarray):
        """This function returns two 3D matrix of binary values for ON and OFF events respectively. The first dimension
        is equal to the number of time-steps (dt) that are present in the whole duration; second and third dimensions
        are equal to the resolution of the sensor (number of pixels along y and x respectively). The matrices are made
        of all zeros except for ones in correspondence to the time-step and the location of an ON/OFF event.
        Args:
            dt (float, required): the time-step in micro-seconds (you should consider applying the refractory filter
                with such refractory value before calling this method). Default value: 10000 us.
            duration (float, optional): the duration of recording/simulation.
        Return:
            (np.ndarray, np.ndarray): the time surfaces (made of binary values) of ON and OFF events separately.
        """
        if not duration:
            duration = self.duration
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        data = self.timereset(self._data)
        res_on = self.time_surface(data=data[pol_on], dt=dt, duration=duration)
        res_off = self.time_surface(data=data[pol_off], dt=dt, duration=duration)
        return res_on, res_off

    def show_time_surface(self, duration: float = 0, fullscreen: bool = False,
                          show: bool or float = True, figsize: (int, int) or None = None):
        """Show a 3D plot of all events in (x,y,t)."""
        t, x, y = self.ts, self.x, self.y
        if duration:
            idx_dur = (t <= duration + t[0])
            t, x, y = t[idx_dur], x[idx_dur], y[idx_dur]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Time surface of events')
        if fullscreen:
            # It works fine with the TkAgg backend on Ubuntu 20.04
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        ax.scatter(x, t * 1e-3, self._N_y - y, c='black', s=1, marker='.')
        ax.set_xlabel('x (pix)', fontsize=12)
        ax.set_ylabel('t (ms)', fontsize=12)
        ax.set_zlabel('y (pix)', fontsize=12)
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_time_surface_onoff(self, duration: float = 0, fullscreen: bool = False,
                                show: bool or float = True, figsize: (int, int) or None = None):
        """Show a 3D plot of events in (x,y,t) and distinguish between ON/OFF with green/red colors respectively."""
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        t_on, x_on, y_on = self.ts[pol_on], self.x[pol_on], self.y[pol_on]
        t_off, x_off, y_off = self.ts[pol_off], self.x[pol_off], self.y[pol_off]
        start = self.ts[0]
        if duration:
            idx_dur_on, idx_dur_off = (t_on <= duration + start), (t_off <= duration + start)
            t_on, x_on, y_on = t_on[idx_dur_on], x_on[idx_dur_on], y_on[idx_dur_on]
            t_off, x_off, y_off = t_off[idx_dur_off], x_off[idx_dur_off], y_off[idx_dur_off]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Time surface of ON (green) and OFF (red) events')
        if fullscreen:
            # It works fine with the TkAgg backend on Ubuntu 20.04
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        ax.scatter(x_on, t_on * 1e-3, self._N_y - y_on, c='green', s=1, marker='.')
        ax.scatter(x_off, t_off * 1e-3, self._N_y - y_off, c='red', s=1, marker='.')
        ax.set_xlabel('x (pix)', fontsize=12)
        ax.set_ylabel('t (ms)', fontsize=12)
        ax.set_zlabel('y (pix)', fontsize=12)
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
        # # Note: this is the same as doing the following
        # idx, spk_count = np.unique(self.id, return_counts=True)
        # frame = np.zeros(self._N_y * self._N_x)
        # frame[idx] = spk_count / duration
        # frame = frame.reshape(self._N_y, self._N_x)
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
        # fig, ax = plt.subplots(1, 2, figsize=figsize)
        # plt.suptitle(title)
        # im0 = ax[0].imshow(frame_on, cmap='Greens')
        # cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.04)
        # cb0.set_label('On FR (Hz)')
        # ax[0].get_xaxis().set_visible(False)
        # ax[0].get_yaxis().set_visible(False)
        # im1 = ax[1].imshow(frame_off, cmap='Reds')
        # cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.04)
        # cb1.set_label('Off FR (Hz)')
        # ax[1].get_xaxis().set_visible(False)
        # ax[1].get_yaxis().set_visible(False)
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
            # # Note: this is equivalent as doing the following
            # pixel_location = self.flat_data[np.logical_and(self.flat_data[:,0]>=k*dur_frame,
            #                                                self.flat_data[:,0]<(k+1)*dur_frame),1]
            # idx, counts = np.unique(pixel_location, return_counts=True)
            # video[k, idx] = counts
        # video = video.reshape((n_frames, self.N_y, self.N_x))
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
    def most_active_pixels(self, num_active: int, return_activity: bool = False) ->\
            np.ndarray or (np.ndarray, np.ndarray):
        neurons, num_spikes = np.unique(self.id, return_counts=True)
        idx_sortactive = np.argsort(num_spikes, kind='stable')[::-1]
        most_active_pxls = neurons[idx_sortactive[:num_active]].astype(int)
        if return_activity:
            their_activity = num_spikes[idx_sortactive[:num_active]]
            return most_active_pxls, their_activity
        return most_active_pxls

    def less_active_pixels(self, num_active: int, non_null: bool = True, return_activity: bool = False) ->\
            np.ndarray or (np.ndarray, np.ndarray):
        num_spikes, neurons = np.histogram(self.id, bins=self._N_y * self._N_x, range=(0, self._N_y * self._N_x))
        if non_null:
            neurons = neurons[:-1][num_spikes > 0]
            num_spikes = num_spikes[num_spikes > 0]
        idx_sortactive = np.argsort(num_spikes, kind='stable')[::-1]
        less_active_pxls = neurons[idx_sortactive[-num_active:]].astype(int)
        if return_activity:
            their_activity = num_spikes[idx_sortactive[-num_active:]]
            return less_active_pxls, their_activity
        return less_active_pxls

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
