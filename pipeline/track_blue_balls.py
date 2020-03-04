from .pipeline import Pipeline
import numpy as np
import imutils
import cv2


class TrackBlueBalls(Pipeline):

    def map(self, data):

        ROI = data["ROI"]
        LAB_ROI = data["LAB_ROI"]
        data["blue_balls_count"] = 0

        font = cv2.FONT_HERSHEY_SIMPLEX

        lower = np.array([54, 82, 49], dtype=np.uint8)
        upper = np.array([185, 142, 87], dtype=np.uint8)

        mask = cv2.inRange(LAB_ROI, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)

        if len(contours) == 0:
            return data

        for ctns in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(ctns)
            (rx, ry, rw, rh) = cv2.boundingRect(ctns)
            moments = cv2.moments(ctns)

            center = (int(moments["m10"] / (moments["m00"] + 1e-7)),
                      int(moments["m01"] / (moments["m00"] + 1e-7)))

            # only proceed if the radius meets a minimum size
            # if radius > min_radius and radius < max_radius:
            if rw > 20 and rw < 40 and rh > 15 and rh < 35:
                data["blue_balls_count"] = data["blue_balls_count"] + 1

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                x = int(x)
                y = int(y)

                cv2.circle(ROI, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(ROI, center, 5, (255, 0, 0), -1)
                cv2.putText(ROI, "Azul".format(int(x), int(y)), (x + 5, y - 5), font, .8, [
                            255, 255, 255], 1, cv2.LINE_AA)

        return data
