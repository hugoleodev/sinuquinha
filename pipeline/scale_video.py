import cv2

from .pipeline import Pipeline


class ScaleVideo(Pipeline):

    def __init__(self, factor):
        self.factor = factor

    def map(self, data):
        data["image"] = cv2.resize(
            data["image"], None, fx=self.factor, fy=self.factor)

        return data
