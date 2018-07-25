import cv2 as cv


class CamRead:
    """
    Deifinition Class for VideoCapture
    """

    def __init__(self, capture):
        """
        Create instance for VideoCapture

        Arguments:
            capture -- capture device, if str - expected URL, movie path or
                something like that, if int - expected device connected on
                local machine
        """

        self.frame = None
        self.frame_num = 0
        self.cap = cv.VideoCapture(capture)
        if not self.cap.isOpened():
            raise Exception("Can't open video device from --- ", capture)

    def read(self) -> iter:
        """
        Read next frame from source

        Returns:
            iter -- read_status, frame, frame_num
        """

        ok, self.frame = self.cap.read()
        self.frame_num += 1
        return ok, self.frame, self.frame_num
