from .pipeline import Pipeline
import numpy as np
import cv2


class DisplayScore(Pipeline):

    def __init__(self, first_player, second_player, max_player_balls):
        self._first_player = first_player
        self._second_player = second_player
        self._max_player_balls = max_player_balls

        self._first_player_score = 0
        self._second_player_score = 0

        super(DisplayScore, self).__init__()

    def map(self, data):
        image = data["image"]
        frame_index = data["frame_idx"]

        blue_balls_count = data["blue_balls_count"]
        red_balls_count = data["red_balls_count"]

        font = cv2.FONT_HERSHEY_SIMPLEX

        if frame_index % 60 == 0:
            self._first_player_score = self._max_player_balls - blue_balls_count
            self._second_player_score = self._max_player_balls - red_balls_count

        cv2.rectangle(image, (245, 10), (350, 40), (0, 128, 255), -1)
        cv2.rectangle(image, (350, 10), (415, 40), (0, 0, 0), -1)
        cv2.rectangle(image, (415, 10), (520, 40), (0, 128, 255), -1)
        cv2.putText(image, self._first_player, (250, 30), font, .6,
                    [255, 255, 255], 1, cv2.LINE_AA)
        cv2.putText(image, self._second_player, (420, 30), font, .6,
                    [255, 255, 255], 1, cv2.LINE_AA)
        cv2.putText(image, "{} X {}".format(self._first_player_score, self._second_player_score), (355, 30), font, .6, [
                    255, 255, 255], 1, cv2.LINE_AA)

        return data
