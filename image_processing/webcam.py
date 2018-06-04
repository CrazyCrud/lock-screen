import cv2
import wx
from datetime import datetime
from image_processing.image_processing import ImageProcessing


class WebcamFeed(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_image(self, width=None, height=None):
        ret, frame = self.capture.read()
        if width is not None and height is not None:
            frame_resized = cv2.resize(frame, (width, height))
            return ImageProcessing.convert_image_to_rgb(frame_resized)
        else:
            return frame
