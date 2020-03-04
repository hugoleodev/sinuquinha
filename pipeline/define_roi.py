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

        ROI[88:413, 100:660] = image[88:413, 100:660]

        LAB_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)

        data["ROI"] = ROI
        data["LAB_ROI"] = LAB_ROI

        return data
