from collections import deque

import cv2
import imutils
import numpy as np

from .pipeline import Pipeline


class TrackWhiteBall(Pipeline):

    def __init__(self):
        self.pts = deque(maxlen=50)

        super(TrackWhiteBall, self).__init__()

    def map(self, data):

        ROI = data["ROI"]
        LAB_ROI = data["LAB_ROI"]

        font = cv2.FONT_HERSHEY_SIMPLEX

        lower = np.array([238, 115, 149], dtype=np.uint8)
        upper = np.array([255, 132, 215], dtype=np.uint8)

        mask = cv2.inRange(LAB_ROI, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)

        if len(contours) == 0:
            return data

        for ctns in contours:

            ((x, y), radius) = cv2.minEnclosingCircle(ctns)
            # w, h = x
            moments = cv2.moments(ctns)

            center = (int(moments["m10"] / (moments["m00"] + 1e-7)),
                      int(moments["m01"] / (moments["m00"] + 1e-7)))

            x = int(x)
            y = int(y)
            radius = int(radius)

            if radius > 8 and radius < 30:
                cv2.circle(ROI, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)

                cv2.circle(ROI, center, 5, (0, 0, 0), -1)
                cv2.putText(ROI, "Branca".format(int(x), int(y)), (x + 5, y - 5), font, .8,
                            [255, 255, 255], 2, cv2.LINE_AA)  # imprime texto das coordenadas

                self.pts.appendleft(center)

                for i in range(1, len(self.pts)):
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue

                    thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
                    cv2.line(ROI, self.pts[i - 1],
                             self.pts[i], (0, 128, 255), thickness)

        return data
