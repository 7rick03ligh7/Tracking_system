from engine.model.yolo import Yolo
from engine.tracker import Tracker
from engine.camread import CamRead
from engine.utils.move_module import MoveModule
from engine.utils.init_module import InitModule
from multiprocessing import Process, Pipe, Event
import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class Tracking_system:

    def __init__(self):
        """
        Create Tracking system instance
        """

        self.capture = None
        self.track_type = None

    def cv_bbox2bbox(self, cv_bbox: iter) -> tuple:
        """
        Convert from bbox cv type (xywh, when xy - left top vertex) to
            bbox yolo type (xywh, when xy - center of bbox)

        Arguments:
            cv_bbox {list} -- bbox cv type

        Returns:
            tuple -- return tuple bbox yolo type
        """

        c_x = cv_bbox[0] + cv_bbox[2] / 2
        c_y = cv_bbox[1] + cv_bbox[3] / 2
        return (c_x, c_y, cv_bbox[2], cv_bbox[3])

    def bbox2cv_bbox(self, bbox: iter) -> tuple:
        """
        Convert from bbox yolo type (xywh, when xy - center of bbox) to
        bbox cv type (xywh, when xy - left top vertex)

        Arguments:
            bbox {list} -- bbox yolo type

        Returns:
            tuple -- return tuple bbox cv type
        """

        p_x = bbox[0] - bbox[2] // 2
        p_y = bbox[1] - bbox[3] // 2
        w = bbox[2]
        h = bbox[3]
        if p_x < 0:
            p_x = 0
        if p_y < 0:
            p_y = 0
        if p_x + w > 1280:
            w = 1280 - p_x
        if p_y + h > 720:
            h = 720 - p_y
        bbox = (p_x, p_y, w, h)
        return tuple(map(int, bbox))

    def camread_proc(self, child_cam: Pipe, event_close: Event):
        """
        Parallel process for read frame from source

        Arguments:
            child_cam {Pipe} -- child pipe for communication with
                parent process, sends captured frames
            event_close {Event} -- event for close process
        """

        # initialize camread
        print("capture from",self.capture)
        cap = CamRead(self.capture)

        # initialize init state from first cam read
        
        ok, frame, frame_num = cap.read()
        child_cam.send([frame, frame_num])

        while True:
            if event_close.is_set():
                return
            #cap.read()
            #cap.read()
            ok, frame, frame_num = cap.read()
            child_cam.send([frame, frame_num])

    def tracker_proc(self, child_track: Pipe, e_track: Event):
        """
        Parallel process for tracking recognized object (in formal, tracking
        bbox)

        Arguments:
            child_track {Pipe} -- pipe for communication with parent process,
                sends bbox cv type
            e_track {Event} -- event for check tracking complete

        Keyword Arguments:
            tracker_type {int} -- tracking type (default: {0:int})
                0 - BOOSTING
                1 - MIL
                2 - KCF
                3 - TLD
                4 - MEDIANFLOW
                5 - GOTURN (not work yet)
        """
        # initialize tracker
        initialized = False
        while not initialized:
            frame, res = child_track.recv()
            if res is not None:
                cv_bbox = self.bbox2cv_bbox(res[0][1:5])
                print(cv_bbox)
                tracker = Tracker(self.track_type, cv_bbox, frame)
                initialized = True
                e_track.set()

        while True:
            recv = child_track.recv()
            frame = recv[0]
            if frame is None:
                return
            if len(recv) == 2:
                correct_cv_bbox = self.bbox2cv_bbox(recv[1])
            else:
                correct_cv_bbox = None

            bbox = tracker.track(frame, correct_cv_bbox)
            child_track.send(bbox)
            e_track.set()

    def recog_proc(self, child_recog: Pipe, e_recog: Event, yolo_type: str):
        """
        Parallel process for object recognition

        Arguments:
            child_recog {Pipe} -- pipe for communication with parent process,
                sends bbox yolo type of recognized object
            e_recog {Event} -- event for indicating complete recognize in frame
        """

        # initialize YOLO
        yolo = Yolo(yolo_type)
        e_recog.set()
        print("yolo defined")

        while True:
            frame = child_recog.recv()
            print("recog process frame recieved")
            if frame is None:
                print("FRAME NONE? R U SURE ABOUT THAT?!")
                return
            res = yolo.detect(frame, cvmat=True)
            print("recog send")
            e_recog.set()
            child_recog.send(res)

    def get_iou(self, box1: iter, box2: iter) -> float:
        """
        Calculate intersection over union (IOU) between bbox yolo type from
            recognition and tracking

        Arguments:
            box1 {iter} -- bbox yolo type
            box2 {iter} -- bbox yolo type

        Returns:
            float -- IOU result
        """

        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] +
                               box2[2] * box2[3] - intersection)

    def show_results(self, frame: np.ndarray, cv_bbox: iter,
                     check_recog_bbox: iter=None, check_track_bbox: iter=None):
        """
        Show recognition and tracking results using cv.imshow

        Arguments:
            frame {np.ndarray} -- captured frame
            cv_bbox {iter} -- bbox cv type from tracking

        Keyword Arguments:
            check_recog_bbox {iter} -- bbox cv type from recognition
                (default: {None})
            check_track_bbox {iter} -- bbox cv type from tracking in recognized
                frame (default: {None})
        """

        p1 = (int(cv_bbox[0]), int(cv_bbox[1]))
        p2 = (int(cv_bbox[0] + cv_bbox[2]),
              int(cv_bbox[1] + cv_bbox[3]))
        cv.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
        cv.putText(frame, 'tracking online', (50, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if check_recog_bbox is not None:
            rec_cv_bbox = self.bbox2cv_bbox(check_recog_bbox)
            # print('rec_cv_bbox', rec_cv_bbox)
            p1_rec = (int(rec_cv_bbox[0]), int(rec_cv_bbox[1]))
            p2_rec = (int(rec_cv_bbox[0] + rec_cv_bbox[2]),
                      int(rec_cv_bbox[1] + rec_cv_bbox[3]))
            cv.rectangle(frame, p1_rec, p2_rec, (0, 255, 0), 2, 1)
            cv.putText(frame, 'recognition check', (50, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 225, 0), 2)

        if check_track_bbox is not None:
            trck_cv_bbox = self.bbox2cv_bbox(check_track_bbox)
            # print('trck_cv_bbox', trck_cv_bbox)
            p1_trc = (int(trck_cv_bbox[0]), int(trck_cv_bbox[1]))
            p2_trc = (int(trck_cv_bbox[0] + trck_cv_bbox[2]),
                      int(trck_cv_bbox[1] + trck_cv_bbox[3]))
            cv.rectangle(frame, p1_trc, p2_trc, (255, 0, 0), 2, 1)
            cv.putText(frame, 'tracking check', (50, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv.imshow("track", frame)

    def run(self, capture=0, track_type: int=0, yolo_type: str='tiny'):
        """
        Start tracking system

        Keyword Arguments:
            capture -- capture device, if str - expected URL, movie path or
                something like that, if int - expected device connected on
                local machine  (default: {0})
            track_type {int} -- tracking type (default: {0:int})
                0 - BOOSTING
                1 - MIL
                2 - KCF
                3 - TLD
                4 - MEDIANFLOW
                5 - GOTURN (not work yet)
            yolo_type {str} -- YOLO recognition type
                'tiny'
                'small'
        """

        self.capture = capture
        self.track_type = track_type
        self.yolo_type = yolo_type
        cam_init_prop = InitModule()
        self.capture = cam_init_prop.streamURL

        # initialize Pipes
        parent_cam, child_cam = Pipe()
        parent_track, child_track = Pipe()
        parent_recog, child_recog = Pipe()
        parent_move, child_move = Pipe()

        e_track = Event()
        e_recog = Event()
        e_cam = Event()

        # ====== recognize process ======
        recog_p = Process(target=self.recog_proc, args=(
            child_recog, e_recog, self.yolo_type))
        recog_p.start()

        e_recog.wait()
        e_recog.clear()
        # ====== cam read process ======
        cam_p = Process(target=self.camread_proc, args=(child_cam, e_cam))
        cam_p.start()

        # ====== cam_move process ======
        moving = MoveModule()
        move_proc = Process(target=moving.coords, args=(child_move, cam_init_prop))
        move_proc.start()

        # recognize
        res = []
        while res == []:
            frame, frame_num = parent_cam.recv()
            parent_recog.send(frame)
            e_recog.wait()
            e_recog.clear()
            res = parent_recog.recv()
            print(frame_num)
        print(res)

        # ===== track process ======
        track_p = Process(target=self.tracker_proc,
                          args=(child_track, e_track))
        track_p.start()
        parent_track.send([frame, res])
        e_track.wait()

        recog_busy = False
        e_recog.clear()
        check_recog_bbox = (0, 0, 0, 0)
        check_track_bbox = (0, 0, 0, 0)
        while True:
            # TODO: is it needed?
            e_track.clear()

            frame, frame_num = parent_cam.recv()

            if not e_track.is_set():
                if not recog_busy:
                    check_frame = frame.copy()
                    parent_recog.send(check_frame)
                    recog_busy = True

                    parent_track.send([check_frame])
                    cv_bbox = parent_track.recv()
                    check_track_bbox = self.cv_bbox2bbox(cv_bbox)
                else:
                    if not e_recog.is_set():
                        parent_track.send([frame])
                        cv_bbox = parent_track.recv()
                        # print('curr frame ', cv_bbox)
                    else:
                        recog_busy = False
                        e_recog.clear()
                        res = parent_recog.recv()
                        if res != []:
                            check_recog_bbox = res[0][1:5]
                            iou = self.get_iou(
                                check_track_bbox, check_recog_bbox)
                            print('check track', check_track_bbox)
                            print('check recog', check_recog_bbox)
                            print("IOU", iou)
                            if iou < 0.5:
                                parent_track.send(
                                    [check_frame, check_recog_bbox])
                                parent_track.recv()
                                parent_track.send([frame])
                                cv_bbox = parent_track.recv()
                            else:
                                parent_track.send([frame])
                                cv_bbox = parent_track.recv()
                        else:
                            print("nothing recognized")
                            parent_track.send([frame])
                            cv_bbox = parent_track.recv()

                #self.show_results(frame, cv_bbox, None, None)
                bbox = self.cv_bbox2bbox(cv_bbox)
                print('send to move', bbox)
                parent_move.send(bbox)

                # close all process
                k = cv.waitKey(1) & 0xff
                if k == 27:
                    print('1')
                    parent_track.send([None])
                    print('2')
                    parent_recog.send(None)
                    print('3')
                    e_cam.set()
                    parent_cam.recv()
                    print('4')

                    track_p.join()
                    print('5')
                    recog_p.join()
                    print('6')
                    cam_p.join()
                    print('7')
                    return
