import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import cv2
import time


# ======================================= Access saved recordings =======================================
def load_recordings(record_type, file_dir):
    """Given a file name (with directory and no file extension) and the type of recordings to load,
    it returns a list containing the different types of information recorded from the DAVIS sensor.
    Args:
        record_type (str, required): the type of info recorded from the sensor. Possible choices are:
            'py_dvs', 'py_aps', 'py_imu', 'py_dvsaps', 'py_all' (i.e. dvs + aps + imu) --> through dv python package
            cpp_dvs'. --> through C++ script
        file_dir (str, required): the directory and name of file to load (without info-type specification and extension,
            e.g. _dvs.npy).
    Returns:
        info (list): list containing the different types of information recorded from the DAVIS sensor.
    """
    func = {'py_dvs': lambda: np.load(file_dir + '_dvs.npy'),
            'py_aps': lambda: np.load(file_dir + '_aps.npy'),
            'py_imu': lambda: np.load(file_dir + '_imu.npy'),
            'py_dvsaps': lambda: [np.load(file_dir + ext) for ext in ['_dvs.npy', '_aps.npy']],
            'py_all': lambda: [np.load(file_dir + ext) for ext in ['_dvs.npy', '_aps.npy', '_imu.npy']],
            'cpp_dvs': lambda: pd.read_csv(file_dir + '_dvs.csv', sep=',', header=0).values
            }.get(record_type)
    if not func:
        raise ValueError("The recording type must be one of the following strings:\n"
                         "'py_dvs', 'py_aps', 'py_imu', 'py_dvsaps', 'py_all', 'cpp_dvs'.")
    data = func()
    return data


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

    def __init__(self, data: np.ndarray or None, shape: (int, int) = (260, 346)):
        """
        This class helps handling event-based data. A set of methods for transforming, taking info and visualizing
        event-based data is available.
        Inputs:
            data (np.ndarray): Events in the form of a Nx4 np.ndarray where N is the total number of events and the
                columns specify the timestamp, the x and y pixel address, and the polarity of all such events.
            shape ((int, int), optional): The shape of the full pixel array in the form of (height, width).
        Attributes:
            N_x, N_y (int, int): Width and height of the camera pixel array.
            data (np.ndarray): Array of events with shape Nx4.
            flat_data (np.ndarray): Array with same events as data but with shape Nx3 because the x,y address of pixels
                are flattened.
            camera_mtx (np.ndarray): 2D array with shape (N_y,N_x) keeping the intrinsic parameters of the camera.
            distortion_coeffs (np.ndarray): 1D array keeping the distortion coefficients given by lens' optical effects.
            _dt (int): time-step or refractory period (us).
        """
        self.N_y, self.N_x = shape if shape is not None else (None, None)
        self.flat_data = None
        if data is not None:
            self.data = self.timereset(data)
            self.__flatten_data()
        self.camera_mtx = None
        self.distortion_coeffs = None
        self._dt = None

    @property
    def ts(self):  # in micro-seconds (us)
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

    @property
    def num_events(self):
        return self.data.shape[0]

    @property
    def duration(self):  # in micro-seconds (us)
        return self.data[-1, 0]

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

    # ---------------------------------- Basic utilities -----------------------------------
    def timereset(self, data: np.ndarray or None = None) -> np.ndarray:
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but resetting all timestamps according to the first event.
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
                represents the timestamps.
        Returns:
            (np.ndarray): events array as input data but resetting all timestamps according to the first one (so that
                the timestamp of the first event is 0).
        """
        if data is not None:
            data[:, 0] -= data[0, 0]
            return data
        else:
            self.data[:, 0] -= self.data[0, 0]
            self.flat_data[:, 0] -= self.flat_data[0, 0]

    def __flatten_data(self):
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x, y address
        and polarity) it returns an array with same number of rows but 3 columns where each pair of x, y (pixel)
        coordinates is converted to an index that corresponds to a flattened pixel array.
        """
        self.flat_data = np.delete(self.data, 2, axis=1)
        try:
            # The following line is equivalent to: dvs_flatten[:, 1] = dvsnpy[:, 1] + dvsnpy[:, 2] * x_dim
            self.flat_data[:, 1] = np.ravel_multi_index((self.data[:, 2], self.data[:, 1]), (self.N_y, self.N_x))
        except:
            raise ValueError('There is an issue in flattening data, you probably set a wrong shape of the pixel array.')

    # ---------------------------------- Undistort data ------------------------------------
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
        file_read = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
        cam_tree = file_read.getNode(sensor_model + '_' + serial)
        self.camera_mtx = np.array(cam_tree.getNode('camera_matrix').mat())
        self.distortion_coeffs = np.array(cam_tree.getNode('distortion_coefficients').mat()).T
        file_read.release()

    def _inverse_map_undistortion(self, shape: (int, int) = (260, 346)) -> (np.ndarray, np.ndarray):
        """Compute the inverse undistortion maps."""
        if self.camera_mtx is None or self.distortion_coeffs is None:
            raise ValueError('If you wish to undistort the event-based data, you must call the method '
                             'load_xml_calibration() first, specifying the file where calibration info are stored '
                             '(camera matrix and lens distortion coefficients).')
        rotation = np.eye(3, dtype=int)
        map1_inv, map2_inv = cv2.initUndistortRectifyMap(self.camera_mtx, -self.distortion_coeffs, R=rotation,
                                                         newCameraMatrix=self.camera_mtx, size=shape[::-1],
                                                         m1type=cv2.CV_32FC1)
        return np.round(map1_inv).astype(int), np.round(map2_inv).astype(int)

    def _remap(self, map1_inv: np.ndarray, map2_inv: np.ndarray):
        """Remap events for removing radial and tangential lens distortion. It takes as input the inverse maps and
        updates events. We remove events falling on those pixels for which there is no correspondence in the
        original pixel array."""
        dvsdat_dst = np.copy(self.data)
        # Remap events
        dvsdat_dst[:, 1] = map1_inv[self.data[:, 2], self.data[:, 1]]
        dvsdat_dst[:, 2] = map2_inv[self.data[:, 2], self.data[:, 1]]
        # Remove all events remapped in pixels outside the shape of the pixel array
        dvsdat_dst = dvsdat_dst[np.logical_and(dvsdat_dst[:, 1] < self.N_x, dvsdat_dst[:, 1] >= 0)]
        dvsdat_dst = dvsdat_dst[np.logical_and(dvsdat_dst[:, 2] < self.N_y, dvsdat_dst[:, 2] >= 0)]
        self.data = dvsdat_dst
        self.__flatten_data()
        self.timereset()

    def undistort(self, shape: (int, int) = (260, 346)):
        """Compute undistortion maps and remap events accordingly in order to correct lens effects. Note that this
        method should be called BEFORE cropping.
        Args:
             shape (int, int): The shape of the full pixel array in the form of (height, width).
                Note that y-dimension must be specified first!
        """
        # Note: we need to compute inverse mapping T_inv such that
        #       x_src, y_src = T{x_dst, y_dst}  -->  x_dst, y_dst = T_inv{x_src, y_src}
        #       To this purpose, note the minus sign in front of the distortion coefficients inside the function
        #       for computing the INVERSE maps.
        self._remap(*self._inverse_map_undistortion(shape=shape))

    # ------------------------------------ Filter data -------------------------------------
    def refractory_filter(self, refractory: int = 100, return_filtered: bool = False, verbose: bool = False):
        """This function applies a refractory filter, removing all events, from the same pixel, coming after the
        previous event by less than a given refractory period (in micro-seconds).
        This is useful for avoiding errors when running simulation, in a clock-driven fashion, having a time-step
        bigger than the inter-spike-interval (ISI) of some pixels (i.e. to avoid that some neurons in the network will
        spike more than once during a single time-step of the simulation).
        Args:
            refractory (int, optional): refractory period each pixel should have, in microseconds (us).
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
            self.data = np.delete(self.data, idx_remove, axis=0)
            self.flat_data = np.delete(self.flat_data, idx_remove, axis=0)
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
            correct_id = all(sameid_removed)
            correct_dt = all(np.asarray(dt_removed) <= refractory)
            print('Are all events removed correct?', all([correct_dt, correct_id]))
            print('    - is dt of removed event wrt the previous one always <', refractory * 10 ** -3, 'ms?', correct_dt)
            print('    - is the previous event emitted by the same pixel as the removed event?', correct_id, '\n')

    def denoising_filter(self, time_window: int = 1e4):
        """Drops events that are 'not sufficiently connected to other events in the recording.'
        In practise that means that an event is dropped if no other event occurred within a spatial neighbourhood
        of 1 pixel and a temporal neighbourhood of time_window (us). Useful to filter noisy recordings.
        Args:
            time_window (int, optional): The maximum temporal distance (in micro-seconds) to the next event, otherwise
                it is dropped. Lower values will therefore mean more events will be filtered out.
        """
        data_filt = np.zeros(self.data.shape, dtype=self.data.dtype)
        timestamp_memory = np.zeros((self.N_y, self.N_x), dtype=self.data.dtype) + time_window
        idx = 0
        for event in self.data:
            x, y, t = int(event[1]), int(event[2]), event[0]

            # # TODO: try adding the space window
            # # Remove next lines
            # start_x, start_y = x - space_window, y - space_window
            # end_x, end_y = x + space_window + 1, y + space_window + 1
            # neighbour_events = timestamp_memory[start_y: end_y,
            #                                     start_x: end_x]
            # if neighbour_events.size:  # apply filtering only at pixels far from the border of the sensor array
            #     if neighbour_events.mean() > t:
            #         print(neighbour_events.mean())
            #         data_filt[idx] = event
            #         idx += 1
            # else:
            #     data_filt[idx] = event
            #     idx += 1

            # Keep from here
            if (
                    (x > 0 and timestamp_memory[y, x - 1] > t)
                    or (x < self.N_x - 1 and timestamp_memory[y, x + 1] > t)
                    or (y > 0 and timestamp_memory[y - 1, x] > t)
                    or (y < self.N_y - 1 and timestamp_memory[y + 1, x] > t)
            ):
                data_filt[idx] = event
                idx += 1

            timestamp_memory[y, x] = t + time_window

        print(idx, self.num_events)
        self.data = data_filt[:idx]
        self.__flatten_data()
        self.timereset()

    # ------------------------------------- Cut data ---------------------------------------
    def crop(self, shape: (int, int)):
        """Given an array of DVS events, with N rows (number of events) and 4 columns (timestamps, x, y address and
        polarity), this function returns the same array but cutting off the events where x >= dim and y >= dim.
        Args:
            shape ((int, int), required): number of pixels along the y and x dimensions (h, w) of the cropped array.
        """
        if shape[1] > self.N_x or shape[0] > self.N_y:
            raise ValueError('Cannot crop data with the given dimension.')
        x_start, y_start = int((self.N_x - shape[1]) / 2), int((self.N_y - shape[0]) / 2)
        x_end, y_end = int((self.N_x + shape[1]) / 2), int((self.N_y + shape[0]) / 2)

        self.data = self.data[np.logical_and(self.data[:, 1] >= x_start,
                                             self.data[:, 1] < x_end)]
        self.data = self.data[np.logical_and(self.data[:, 2] >= y_start,
                                             self.data[:, 2] < y_end)]
        self.data[:, 1] -= x_start
        self.data[:, 2] -= y_start
        self.N_y, self.N_x = shape
        self.__flatten_data()
        self.timereset()

    def crop_square(self, dim: int):
        """Given an array of DVS events, with N rows (number of events) and 4 columns (timestamps, x, y address and
        polarity), this function returns the same array but cutting off the events where x >= dim and y >= dim.
        Args:
            dim (int, required): number of pixels along the x and y dimensions of the new squared pixel array.
        """
        if dim > self.N_x or dim > self.N_y or dim < 0:
            raise ValueError('Cannot crop data with the given dimension.')
        x_start, y_start = int((self.N_x - dim) / 2), int((self.N_y - dim) / 2)
        x_end, y_end = int((self.N_x + dim) / 2), int((self.N_y + dim) / 2)

        self.data = self.data[np.logical_and(self.data[:, 1] >= x_start,
                                             self.data[:, 1] < x_end)]
        self.data = self.data[np.logical_and(self.data[:, 2] >= y_start,
                                             self.data[:, 2] < y_end)]
        self.data[:, 1] -= x_start
        self.data[:, 2] -= y_start
        self.N_x, self.N_y = dim, dim
        self.__flatten_data()
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
        y_start, y_end = self.N_y - end[1], self.N_y - start[1]
        self.data = self.data[np.logical_and(self.data[:, 1] >= x_start,
                                             self.data[:, 1] < x_end)]
        self.data = self.data[np.logical_and(self.data[:, 2] >= y_start,
                                             self.data[:, 2] < y_end)]
        self.data[:, 1] -= x_start
        self.data[:, 2] -= y_start
        self.N_x, self.N_y = (x_end - x_start), (y_end - y_start)
        self.__flatten_data()
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

    def select_polarity(self, polarity: str = 'on'):
        """This function selects only events of a specific polarity and removes all the others."""
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

    # ----------------------------- Some other transformations -----------------------------
    def flip_polarity(self):
        """Flip the polarity of events. ON becomes OFF and vice versa."""
        self.data[:, 3] = np.logical_not(self.data[:, 3]).astype(int)
        self.flat_data[:, 2] = np.logical_not(self.flat_data[:, 2]).astype(int)

    def merge_polarities(self):
        """Merge both polarities into ON events only."""
        self.data[self.data[:, 3] == 0, 3] = 1
        self.flat_data[self.flat_data[:, 2] == 0, 2] = 1

    def reverse_time(self):
        """Reverse the timing of events."""
        self.data[:, 0] = np.abs(self.data[:, 0] - self.data[:, 0].max())
        self.flat_data[:, 0] = np.abs(self.flat_data[:, 0] - self.flat_data[:, 0].max())

    # -------------------------------- Compute firing rates --------------------------------
    def fraction_onoff(self) -> (float, float):
        """Compute the fraction of ON and OFF events wrt all the events."""
        rate_on = (len(self.data[self.data[:, -1] == 1])) / self.data.shape[0]
        rate_off = 1 - rate_on
        return rate_on, rate_off

    def firingrate_onoff(self) -> (float, float):
        """Compute the firing-rate of ON and OFF events."""
        pol_on = (self.data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        fr_on = (len(self.data[pol_on])) / (self.N_x * self.N_y * (self.data[-1, 0] - self.data[0, 0]) * 10 ** -6)
        fr_off = (len(self.data[pol_off])) / (self.N_x * self.N_y * (self.data[-1, 0] - self.data[0, 0]) * 10 ** -6)
        return fr_on, fr_off

    def firingrate(self) -> (float, float):
        """Compute the firing-rate of ON and OFF events."""
        return self.data.shape[0] / (self.N_x * self.N_y * (self.data[-1, 0] - self.data[0, 0]) * 10 ** -6)

    def instantaneous_firingrate(self, timestamps: np.ndarray or None = None, num_neurons: int = 0, duration: int = 0,
                                 window: int = 50000, smooth: bool = True) -> np.ndarray:
        """This function computes the instantaneous firing rate of a neuron (or population of neurons, in such case you
        should specify the total number of neurons) taking as input its spike times. The temporal window for averaging
        the spiking activity is also taken as input (in us). If the bool smooth parameter is True, a gaussian window
        function will be used, else a rectangular function."""
        if not duration:
            duration = self.duration
        if not num_neurons:
            num_neurons = self.N_x * self.N_y
        if timestamps is None:
            timestamps = self.ts
        timestamps = timestamps[timestamps <= duration]
        n_steps = int(round(duration / window))
        ifr = np.zeros(n_steps)
        if smooth:
            times = np.linspace(0, duration, duration + 1, endpoint=True)
            for k in range(n_steps):
                center = k * window
                gauss = np.exp(-(times - center) ** 2 / (2 * window ** 2))
                ifr[k] = np.sum(gauss[timestamps])
        else:
            for k in range(n_steps):
                center = k * window
                idx_in_win = np.where(np.logical_and(timestamps >= center - window / 2,
                                                     timestamps < center + window / 2))[0]
                ifr[k] = len(timestamps[idx_in_win])
        ifr /= ((window * 10 ** -6) * num_neurons)
        return ifr

    def show_IFR(self, ifr: np.ndarray, duration: int or None = None,
                 show: bool = True, figsize: (int, int) or None = None):
        if not duration:
            duration = self.duration
        time = np.linspace(0, duration * 10 ** -3, len(ifr))
        plt.figure(figsize=figsize)
        plt.plot(time, ifr)
        plt.title('Instantaneous Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('IFR (Hz)')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def spike_train(self, neuron_id: (int, int) or int, return_id: bool = False) -> (np.ndarray, np.ndarray) or (np.ndarray, np.ndarray, np.ndarray):
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

    def show_spiketrain(self, timestamps: np.ndarray or list, polarities: np.ndarray or list,
                        duration: int or None = None, title: str = 'Spike Train',
                        show: bool = True, figsize: (int, int) = (12, 2)):
        if not duration:
            duration = self.duration
        pol_on = (polarities == 1)
        pol_off = np.logical_not(pol_on)
        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plot1, = plt.plot(timestamps[pol_on] * 10 ** -3, np.ones_like(timestamps[pol_on]), label='ON',
                          marker='|', markersize=20, color='g', linestyle='None')
        plot2, = plt.plot(timestamps[pol_off] * 10 ** -3, np.ones_like(timestamps[pol_off]), label='OFF',
                          marker='|', markersize=20, color='r', linestyle='None')
        plt.legend(handles=[plt.plot([], ls='-', color=plot1.get_color())[0],
                            plt.plot([], ls='-', color=plot2.get_color())[0]],
                   labels=[plot1.get_label(), plot2.get_label()])
        plt.title(title)
        plt.xlabel('Time (ms)')
        plt.xlim(0, duration * 10 ** -3)
        plt.ylim(0.8, 1.2)
        plt.yticks([])
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_rasterplot(self, data: np.ndarray or None = None, show: bool = True, figsize: (int, int) or None = None):
        if data is None:
            t, i, p = self.ts, self.id, self.pol
        elif data.shape[1] == 3:
            t, i, p = data[:, 0], data[:, 1], data[:, 2]
        else:
            raise ValueError('Data has wrong shape! It should be Nx3, in the form: [timestamp, flat index, polarity].')
        pol_on = (p == 1)
        pol_off = np.logical_not(pol_on)
        plt.figure(figsize=figsize)
        plt.plot(t[pol_on] * 10 ** -3, i[pol_on], '|g', markersize=2, label='ON')
        plt.plot(t[pol_off] * 10 ** -3, i[pol_off], '|r', markersize=2, label='OFF')
        plt.legend(loc='upper right', fontsize=14, markerscale=2.5)
        plt.title('Raster-plot of events')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    # ------------------------------ Reconstruct Time-Surfaces -----------------------------
    def timesurface(self, data: np.ndarray or None = None, dt: int or None = None, duration: int = 0,
                    verbose: bool = False) -> np.ndarray:
        """This function returns a 3D matrix of binary values. The first dimension is equal to the number of time-steps
        (dt) that are present in the whole duration; second and third dimensions are equal to the resolution of the
        sensor (number of pixels along y and x respectively). The matrix is made of all zeros except for ones in
        correspondence to the time-step and the location of an event.
        Args:
            data (np.ndarray): array of events with shape (N, 4) where N is the number of events.
            dt (int, required): the time-step in micro-seconds (you should consider applying the refractory filter with
                such refractory value before calling this method). Default value: 10000 us.
            duration (int, optional): the duration of recording/simulation.
            verbose (bool, optional): whether to print out some information.
        Return:
            (np.ndarray): the time surface made of binary values.
        """
        if data is None:
            data = self.data
        if not dt:
            # If a refractory filter was applied, the time-step is set to the refractory period, else to 10ms
            dt = self._dt if self._dt else 1e4
        if not duration:
            duration = data[-1, 0]
        else:
            data = data[data[:, 0] <= duration]
        N_t = int(np.ceil(duration / dt))
        res = np.zeros((N_t, self.N_y, self.N_x), dtype=bool)
        for t in range(N_t):
            idx = np.where(np.logical_and(data[:, 0] >= t * dt,
                                          data[:, 0] < (t+1) * dt))[0]
            if not len(idx):
                continue
            y, x = data[idx, 2], data[idx, 1]
            res[t, y, x] = 1
        if verbose:
            n_evts_surf = len(np.where(res > 0)[0])
            correct = {True: 'exact', False: 'approximate (%d events have been removed)'
                                             % (data.shape[0] - n_evts_surf)}.get(n_evts_surf == data.shape[0])
            print('\nThe time surface reconstructed from the events is {}.\n'.format(correct))
        return res

    def timesurface_onoff(self, dt: int or None = None, duration: int = 0) -> (np.ndarray, np.ndarray):
        """This function returns two 3D matrix of binary values for ON and OFF events respectively. The first dimension
        is equal to the number of time-steps (dt) that are present in the whole duration; second and third dimensions
        are equal to the resolution of the sensor (number of pixels along y and x respectively). The matrices are made
        of all zeros except for ones in correspondence to the time-step and the location of an ON/OFF event.
        Args:
            dt (int, required): the time-step in micro-seconds (you should consider applying the refractory filter with
                such refractory value before calling this method). Default value: 10000 us.
            duration (int, optional): the duration of recording/simulation.
        Return:
            (np.ndarray, np.ndarray): the time surfaces (made of binary values) of ON and OFF events separately.
        """
        if not duration:
            duration = self.data[-1, 0]
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        res_on = self.timesurface(data=self.data[pol_on], dt=dt, duration=duration)
        res_off = self.timesurface(data=self.data[pol_off], dt=dt, duration=duration)
        return res_on, res_off

    def show_timesurface(self, duration: int = 0, fullscreen: bool = False,
                         show: bool = True, figsize: (int, int) or None = None):
        """Show a 3D plot of all events in (x,y,t)."""
        t, x, y = self.ts, self.x, self.y
        if duration:
            idx_dur = (t <= duration)
            t, x, y = t[idx_dur], x[idx_dur], y[idx_dur]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Time surface of events')
        if fullscreen:
            # It works fine with the TkAgg backend on Ubuntu 20.04
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        ax.scatter(x, t * 10 ** -3, self.N_y - y, c='black', s=1, marker='.')
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

    def show_timesurface_onoff(self, duration: int = 0, fullscreen: bool = False,
                               show: bool = True, figsize: (int, int) or None = None):
        """Show a 3D plot of events in (x,y,t) and distinguish between ON/OFF with green/red colors respectively."""
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        t_on, x_on, y_on = self.ts[pol_on], self.x[pol_on], self.y[pol_on]
        t_off, x_off, y_off = self.ts[pol_off], self.x[pol_off], self.y[pol_off]
        if duration:
            idx_dur_on, idx_dur_off = (t_on <= duration), (t_off <= duration)
            t_on, x_on, y_on = t_on[idx_dur_on], x_on[idx_dur_on], y_on[idx_dur_on]
            t_off, x_off, y_off = t_off[idx_dur_off], x_off[idx_dur_off], y_off[idx_dur_off]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Time surface of ON (green) and OFF (red) events')
        if fullscreen:
            # It works fine with the TkAgg backend on Ubuntu 20.04
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        ax.scatter(x_on, t_on * 10 ** -3, self.N_y - y_on, c='green', s=1, marker='.')
        ax.scatter(x_off, t_off * 10 ** -3, self.N_y - y_off, c='red', s=1, marker='.')
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

    # --------------------------------- Reconstruct frames ---------------------------------
    def frame(self, data: np.ndarray or None = None, clip_value: float = 0.5) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address, and polarity) it returns a 2D array: the reconstructed frame, where each element represents the
        firing rate of the corresponding pixel during the whole duration of recording.
        Returns:
            img (np.ndarray): frame reconstructed from DVS info by accumulating events.
        """
        if data is None:
            data = self.data
        axrange = [(0, self.N_y), (0, self.N_x)]
        pol_on = (data[:, -1] == 1)
        pol_off = np.logical_not(pol_on)
        img_on, _, _ = np.histogram2d(data[pol_on, 2], data[pol_on, 1],
                                      bins=(axrange[0][1], axrange[1][1]), range=axrange)
        img_off, _, _ = np.histogram2d(data[pol_off, 2], data[pol_off, 1],
                                       bins=(axrange[0][1], axrange[1][1]), range=axrange)
        if clip_value:
            img = np.clip((img_on - img_off), -clip_value, clip_value) + clip_value
        else:
            img = (img_on - img_off)
        return img

    def show_frame(self, clip_value: float = 0.5, title: str = "DVS events accumulator",
                   show: bool = True, figsize: (int, int) = (5, 4)):
        """Show the reconstructed frame of events obtained accumulating all DVS events in time.
        Args:
            clip_value (float, optional): clip value of normalized array.
            title (str, optional): title of the plot.
            show (bool, optional): whether to show or not the window.
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

    def frame_firingrate(self, data: np.ndarray or None = None, duration: int = 0) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten
        pixel address and polarity) it returns a 2D array with y_dim rows and x_dim columns where each element
        represents the firing rate (in Hz) of the corresponding pixel during the whole recording.
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and 3 columns (timestamps, flatten
                pixel address and polarity).
            duration (float, optional): if it is specified, the entered value is used to compute the firing rate,
                otherwise the duration is computed from data.
        Returns:
            frame (np.ndarray): single frame (with shape (y_dim)x(x_dim)) obtained accumulating all DVS events in time.
        """
        if data is None:
            data = self.flat_data
        if not duration:
            duration = (data[-1, 0] - data[0, 0]) * 10 ** - 6  # seconds
            if duration == 0:
                return np.zeros((self.N_y, self.N_x))
        unique, counts = np.unique(data[:, 1], return_counts=True)
        fr = dict(zip(unique, counts / duration))
        frame = np.zeros(self.N_y * self.N_x)
        frame[list(fr.keys())] = list(fr.values())
        return frame.reshape(self.N_y, self.N_x)

    def frame_firingrate_onoff(self) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten
        pixel address and polarity) it returns two (ON/OFF) 2D arrays with N_y rows and N_x columns where each element
        represents the firing rate (in Hz) of the corresponding pixel during the whole recording.
        Returns:
            frame_on, frame_off (np.ndarray, np.ndarray): two frames (with shape (y_dim)x(x_dim)) obtained accumulating
                all DVS ON/OFF events in time.
        """
        duration = self.ts[-1] * 10 ** -6  # seconds
        if duration == 0:
            return np.zeros((self.N_y, self.N_x)), np.zeros((self.N_y, self.N_x))
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        frame_on = self.frame_firingrate(self.flat_data[pol_on], duration=duration)
        frame_off = self.frame_firingrate(self.flat_data[pol_off], duration=duration)
        return frame_on, frame_off

    def show_frame_firingrate(self, title: str = "DVS events accumulator",
                              show: bool = True, figsize: (int, int) = (5, 4)):
        """Show the single frame obtained accumulating all DVS events in time.
        """
        frame = self.frame_firingrate()
        plt.figure(figsize=(figsize[0] + 1.2, figsize[1]))  # + 1.2 along x for colorbar
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
                                    show: bool = True, figsize: (int, int) = (5, 8)):
        """Show the single frame obtained accumulating all DVS events in time.
        """
        frame_on, frame_off = self.frame_firingrate_onoff()
        fig, ax = plt.subplots(2, 1, figsize=(figsize[0] + 1.2, figsize[1] + 0.8))  # + 1.2 along x for colorbar
        plt.suptitle(title)
        im0 = ax[0].imshow(frame_on, cmap='Greens')
        cb0 = fig.colorbar(im0, ax=ax[0])
        cb0.set_label('On FR (Hz)')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        im1 = ax[1].imshow(frame_off, cmap='Reds')
        cb1 = fig.colorbar(im1, ax=ax[1])
        cb1.set_label('Off FR (Hz)')
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

    # --------------------------------- Reconstruct video ----------------------------------
    def video(self, refresh: float = 50, clip_value: float = 0.5) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 4 columns (timestamps, x and y
        pixel address, and polarity) it returns a 2D array: an array composed of N 2D arrays where N is the number of
        frames in the reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element
        represents the firing rate of the corresponding pixel during a 1/rate time-window.
        """
        duration = self.ts[-1] * 10 ** -6  # seconds
        if duration == 0:
            return np.zeros((1, self.N_y, self.N_x))
        n_frames = int(round(refresh * duration))
        dur_frame = int(round(10 ** 6 / refresh))  # micro-seconds
        video = np.zeros((n_frames, self.N_y, self.N_x))
        for i in range(n_frames):
            events = self.data[np.logical_and(self.data[:, 0] >= i * dur_frame,
                                              self.data[:, 0] < (i + 1) * dur_frame), :]
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
        dt = 0 if not refresh else int(round(10 ** 3 / refresh))
        dvsvid = self.video(refresh=refresh, clip_value=clip_value)
        if clip_value:
            dvsvid /= float(clip_value * 2)
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self.N_x * zoom), int(self.N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)
        cv2.destroyAllWindows()

    def video_firingrate(self, data: np.ndarray = None, refresh: float = 50, duration: int = 0) -> np.ndarray:
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten
        pixel address and polarity) it returns an array composed of N 2D arrays where N is the number of frames in the
        reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element represents the
        firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            data (np.ndarray, optional): events array with N rows (number of events) and 3 columns (timestamps, flatten
                pixel address and polarity).
            refresh (float, optional): the wanted frame rate (in Hz) for the resulting video.
            duration (float, optional): if it is specified, the entered value is used to compute the firing rate,
                otherwise the duration is computed from data.
        Returns:
            video (np.ndarray): array of frames reconstructed from DVS info by accumulating events in 1/rate-long
                time-windows. Its shape is Nx(y_dim)x(x_dim) where N is the number of frames and x_dim, y_dim are the
                dimensions of the pixel array).
        """
        if data is None:
            data = self.flat_data
        if not duration:
            duration = (data[-1, 0] - data[0, 0]) * 10 ** -6  # seconds
            if duration == 0:
                return np.zeros((1, self.N_y, self.N_x))
        n_frames = int(round(refresh * duration))
        dur_frame = int(round(10 ** 6 / refresh))  # micro-seconds
        video = np.zeros((n_frames, self.N_y * self.N_x))
        for k in range(n_frames):
            id_dvs = data[np.logical_and(data[:, 0] >= k * dur_frame,
                                         data[:, 0] < (k + 1) * dur_frame), 1]
            unique, counts = np.unique(id_dvs, return_counts=True)
            fr = dict(zip(unique, counts))
            video[k, list(fr.keys())] = list(fr.values())
        video /= (dur_frame * 10 ** -6)
        # Reshape the video to size (n_frames, y_dim, x_dim)
        return video.reshape((n_frames, self.N_y, self.N_x))

    def video_firingrate_onoff(self, refresh: float = 50) -> (np.ndarray, np.ndarray):
        """Given an array of events from the DVS, with N rows (number of events) and 3 columns (timestamps, flatten
        pixel address and polarity) it returns two (ON/OFF) arrays composed of N 2D arrays where N is the number of
        frames in the reconstructed video. Each 2D array (frame) has N_y rows and N_x columns where each element
        represents the firing rate of the corresponding pixel during a 1/rate time-window.
        Args:
            refresh (float, required): the wanted frame rate (in Hz) for the resulting video.
        Returns:
            video_on, video_off (np.ndarray, np.ndarray): two videos (with shape Nx(y_dim)x(x_dim), where N is the
                number of frames) obtained accumulating all DVS ON/OFF events in time.
        """
        duration = self.ts[-1] * 10 ** -6  # seconds
        if duration == 0:
            return np.zeros((1, self.N_y, self.N_x)), np.zeros((1, self.N_y, self.N_x))
        pol_on = (self.pol == 1)
        pol_off = np.logical_not(pol_on)
        video_on = self.video_firingrate(data=self.flat_data[pol_on], refresh=refresh, duration=duration)
        video_off = self.video_firingrate(data=self.flat_data[pol_off], refresh=refresh, duration=duration)
        return video_on, video_off

    def show_video_firingrate(self, refresh: float = 50, title: str = 'DVS reconstructed video',
                              position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of DVS-reconstructed frames (obtained accumulating events in a time-window).
        """
        dt = 0 if not refresh else int(round(10 ** 3 / refresh))
        dvsvid = self.video_firingrate(refresh=refresh)
        # Convert each pixel's firing rate to a grayscale value (in range [0, 255])
        dvsvid = np.uint8(np.interp(dvsvid, (dvsvid.min(), dvsvid.max()), (0, 255)))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self.N_x * zoom), int(self.N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    def show_video_firingrate_onoff(self, refresh: float = 50, title: str = 'On-Off DVS reconstructed video',
                                    position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of DVS-reconstructed frames (obtained accumulating events in a time-window).
        """
        dt = 0 if not refresh else int(round(10 ** 3 / refresh))
        dvsvid_on, dvsvid_off = self.video_firingrate_onoff(refresh=refresh)
        dvsvid = np.zeros((*dvsvid_on.shape, 3))  # BGR video
        minval, maxval = min(dvsvid_on.min(), dvsvid_off.min()), max(dvsvid_on.max(), dvsvid_off.max())
        dvsvid[..., 1] = np.uint8(np.interp(dvsvid_on, (minval, maxval), (0, 255)))
        dvsvid[..., 2] = np.uint8(np.interp(dvsvid_off, (minval, maxval), (0, 255)))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self.N_x * zoom), int(self.N_y * zoom))
        for image in dvsvid:
            image = cv2.resize(image, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, image)
            cv2.waitKey(dt)  # the higher the value inside cv2.WaitKey(), the slower the video will appear!!
        cv2.destroyAllWindows()

    # ----------------------------------- Utility plots ------------------------------------
    def flatten_id(self, y: np.ndarray or int, x: np.ndarray or int):
        return np.ravel_multi_index((y, x), (self.N_y, self.N_x))

    def expand_id(self, id: np.ndarray or int):
        return np.unravel_index(id, (self.N_y, self.N_x))

    def most_active_pixels(self, num_active: int) -> np.ndarray:
        neurons, num_spikes = np.unique(self.flat_data[:, 1], return_counts=True)
        idx_sortactive = np.argsort(num_spikes, kind='stable')[::-1]
        return neurons[idx_sortactive[:num_active]].astype(int)

    def start_stimulus(self) -> int:
        # TODO: check this out!
        dvsvid = self.video_firingrate(self.flat_data, refresh=60)
        dvsvid = dvsvid.reshape(dvsvid.shape[0], self.N_x * self.N_y)
        inst_fr = [dvsvid[k, :].sum() for k in range(dvsvid.shape[0])]
        start = self.flat_data[np.argmax(inst_fr)][0]
        if start > 30:
            start -= 30
        return start


def adapt_dtype(array: np.ndarray) -> np.ndarray:
    if array.max() < 2 ** 8:
        return array.astype(np.uint8)
    elif array.max() < 2 ** 16:
        return array.astype(np.uint16)
    elif array.max() < 2 ** 32:
        return array.astype(np.uint32)
    else:
        return array.astype(np.uint64)
