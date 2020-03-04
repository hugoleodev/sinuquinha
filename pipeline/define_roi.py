from .pipeline import Pipeline
import numpy as np
import cv2


class DefineROI(Pipeline):

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

        super(DefineROI, self).__init__()

    def map(self, data):
        image = data["image"]

        ROI = np.zeros_like(image)

        x_start, x_end = self.x_range
        y_start, y_end = self.y_range

        ROI[x_start:x_end, y_start:y_end] = image[x_start:x_end, y_start:y_end]

        LAB_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)

        data["ROI"] = ROI
        data["LAB_ROI"] = LAB_ROI

        return data
