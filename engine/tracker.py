import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np


class Tracker:
    """
    Class for define window tracker
    """

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # Set up tracker.

    # @log
    def __init__(self, track_type: int, init_bbox: tuple,
                 init_frame: np.ndarray):
        """
        Create tracker instance

        Arguments:
            track_type {int} -- tracking type (default: {0:int})
                0 - BOOSTING
                1 - MIL
                2 - KCF
                3 - TLD
                4 - MEDIANFLOW
                5 - GOTURN (not work yet)
            init_bbox {tuple} -- bbox cv type for initialization or fixing
                tracker
            init_frame {np.ndarray} -- frame cvmat type for initialization or
                fixing tracker
        """

        self.bbox = init_bbox
        self.frame = init_frame

        # Tracker type
        self.tracker_type = self.tracker_types[track_type]

        # Create tracker
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            self.tracker = cv.TrackerMIL_create()
        elif self.tracker_type == 'KCF':
            self.tracker = cv.TrackerKCF_create()
        elif self.tracker_type == 'TLD':
            self.tracker = cv.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv.TrackerMedianFlow_create()
        elif self.tracker_type == 'GOTURN':
            self.tracker = cv.TrackerGOTURN_create()
        else:
            raise Exception('Wrong tracker type')

        # Init tracker with type
        self.tracker.init(self.frame, self.bbox)

    def track(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """
        Tracking ROI (region of interest)

        Arguments:
            frame {np.ndarray} -- track frame
            bbox {tuple} -- if not None reinitial tracking

        Returns:
            tuple -- bbox cv type
        """

        self.frame = frame

        if bbox is not None:
            # Create tracker
            if self.tracker_type == 'BOOSTING':
                self.tracker = cv.TrackerBoosting_create()
            elif self.tracker_type == 'MIL':
                self.tracker = cv.TrackerMIL_create()
            elif self.tracker_type == 'KCF':
                self.tracker = cv.TrackerKCF_create()
            elif self.tracker_type == 'TLD':
                self.tracker = cv.TrackerTLD_create()
            elif self.tracker_type == 'MEDIANFLOW':
                self.tracker = cv.TrackerMedianFlow_create()
            elif self.tracker_type == 'GOTURN':
                self.tracker = cv.TrackerGOTURN_create()
            else:
                raise Exception('Wrong tracker type')
            self.bbox = bbox
            self.tracker.init(self.frame, self.bbox)

        else:
            # Update tracker
            ok, self.bbox = self.tracker.update(self.frame)

        return self.bbox
