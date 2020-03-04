import cv2
import numpy as np
import imutils
from collections import deque

pts = deque(maxlen=30)

cap = cv2.VideoCapture("videos/sinuquinha.mp4")
ret, prev_frame = cap.read()

scale = .4


min_radius = 8
max_radius = 30

font = cv2.FONT_HERSHEY_SIMPLEX

snapshot = 1

max_blue_ball = 4
max_red_ball = 4
frame_index = 0

jogador1_placar = 0
jogador2_placar = 0


bola_branca = cv2.imread("bola_branca.jpg")

bola_vermelha = cv2.imread("bola_vermelha.jpg")

bola_azul = cv2.imread("bola_azul.jpg")


prev = cv2.resize(prev_frame, None, fx=scale, fy=scale)

# fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # Cria o objeto para gravar vídeo
# writer = cv2.VideoWriter('sinuquinha_final.mp4', fourcc,
#                          30, (prev.shape[1], prev.shape[0]))

# writer = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    count_blue_balls = 0
    count_red_balls = 0
    ret, frame = cap.read()

    if not ret:
        exit()

    frame_index = frame_index + 1

    print("frame_index", frame_index)

    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # cv2.rectangle(frame, (245, 10), (350, 40), (0, 128, 255), -1)
    # cv2.rectangle(frame, (350, 10), (415, 40), (0, 0, 0), -1)
    # cv2.rectangle(frame, (415, 10), (520, 40), (0, 128, 255), -1)
    # cv2.putText(frame, "Jogador 1", (250, 30), font, .6, [255, 255, 255], 1, cv2.LINE_AA)
    # cv2.putText(frame, "Jogador 2", (420, 30), font, .6, [255, 255, 255], 1, cv2.LINE_AA)
    # cv2.putText(frame, "{} X {}".format(max_blue_ball - count_blue_balls, max_red_ball - count_red_balls), (355, 30), font, .6, [255, 255, 255], 1, cv2.LINE_AA) # imprime texto das coordenadas
    # cv2.putText(frame, "Jogador 2", (280, 10), font, .6, [255, 255, 255], 1, cv2.LINE_AA) # imprime texto das coordenadas
    ROI = np.zeros_like(frame)

    ROI[88:413, 100:660] = frame[88:413, 100:660]

    cv2.imshow("ROI", ROI)
    cv2.waitKey(1)
    continue

    # frame = cv2.GaussianBlur(frame, (3, 3), 0, 0)

    red_ball_lower = np.array([30, 180, 0], dtype=np.uint8)
    red_ball_upper = np.array([255, 213, 160], dtype=np.uint8)

    blue_ball_lower = np.array([142, 0, 0], dtype=np.uint8)
    blue_ball_upper = np.array([255, 255, 112], dtype=np.uint8)

    white_ball_lower = np.array([238, 115, 149], dtype=np.uint8)
    white_ball_upper = np.array([255, 132, 215], dtype=np.uint8)

    LAB_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)

    red_mask = cv2.inRange(LAB_ROI, red_ball_lower, red_ball_upper)

    cv2.imshow("red_mask", red_mask)
    blue_mask = cv2.inRange(ROI, blue_ball_lower, blue_ball_upper)

    white_mask = cv2.inRange(LAB_ROI, white_ball_lower, white_ball_upper)

    # output = white_mask.copy()
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,120, param1=50 ,param2=30, minRadius=min_radius, maxRadius=max_radius)

    # if circles is not None:
    #     print("Detectou", circles)
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # cv2.imshow("output", frame)
    # cv2.waitKey(0)

    contours = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        for ctns in contours:

            ((x, y), radius) = cv2.minEnclosingCircle(ctns)
            # w, h = x
            moments = cv2.moments(ctns)

            center = (int(moments["m10"] / (moments["m00"] + 1e-7)),
                      int(moments["m01"] / (moments["m00"] + 1e-7)))

            x = int(x)
            y = int(y)
            radius = int(radius)
            xx = x - radius
            yy = y - radius
            w = 2 * radius
            h = 2 * radius

            # only proceed if the radius meets a minimum size
            if radius > min_radius and radius < max_radius:
                # print("comparacao_histograma", comparacao_histograma)

                # draw the circle and centroid on the frame,
                # then update the list of tracked points

                cv2.circle(ROI, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)

                # print("center -> ", center)
                cv2.circle(ROI, center, 5, (0, 0, 0), -1)
                # imagem_cortada = frame[y:y+h,x:x+w]
                cv2.putText(ROI, "Branca".format(int(x), int(y)), (x + 5, y - 5), font, .8,
                            [255, 255, 255], 2, cv2.LINE_AA)  # imprime texto das coordenadas

                pts.appendleft(center)

                # loop over the set of tracked points
                for i in range(1, len(pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[i - 1] is None or pts[i] is None:
                        continue

                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
                    cv2.line(ROI, pts[i - 1], pts[i], (0, 128, 255), thickness)

    contours = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        for ctns in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(ctns)
            (rx, ry, rw, rh) = cv2.boundingRect(ctns)
            moments = cv2.moments(ctns)

            center = (int(moments["m10"] / (moments["m00"] + 1e-7)),
                      int(moments["m01"] / (moments["m00"] + 1e-7)))

            # only proceed if the radius meets a minimum size
            # if radius > min_radius and radius < max_radius:
            print("red rw", rw, "red rh", rh)
            if rw > 20 and rw < 40 and rh > 15 and rh < 35:

                count_red_balls = count_red_balls + 1
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                x = int(x)
                y = int(y)

                cv2.circle(ROI, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(ROI, center, 5, (0, 0, 255), -1)
                cv2.putText(ROI, "Vermelha".format(int(x), int(y)), (x + 5, y - 5), font, .8, [
                            255, 255, 255], 1, cv2.LINE_AA)  # imprime texto das coordenadas

    contours = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        for ctns in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(ctns)
            (rx, ry, rw, rh) = cv2.boundingRect(ctns)
            moments = cv2.moments(ctns)

            center = (int(moments["m10"] / (moments["m00"] + 1e-7)),
                      int(moments["m01"] / (moments["m00"] + 1e-7)))

            # only proceed if the radius meets a minimum size
            # if radius > min_radius and radius < max_radius:
            print("blue rw", rw, "blue rh", rh)
            if rw > 20 and rw < 30 and rh > 15 and rh < 25:
                count_blue_balls = count_blue_balls + 1
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                x = int(x)
                y = int(y)

                cv2.circle(ROI, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(ROI, center, 5, (255, 0, 0), -1)
                cv2.putText(ROI, "Azul".format(int(x), int(y)), (x + 5, y - 5), font, .8,
                            [255, 255, 255], 1, cv2.LINE_AA)  # imprime texto das coordenadas

    frame[88:413, 100:660] = ROI[88:413, 100:660]

    if frame_index % 60 == 0:
        jogador1_placar = max_blue_ball - count_blue_balls
        jogador2_placar = max_red_ball - count_red_balls

    cv2.rectangle(frame, (245, 10), (350, 40), (0, 128, 255), -1)
    cv2.rectangle(frame, (350, 10), (415, 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (415, 10), (520, 40), (0, 128, 255), -1)
    cv2.putText(frame, "Jogador 1", (250, 30), font, .6,
                [255, 255, 255], 1, cv2.LINE_AA)
    cv2.putText(frame, "Jogador 2", (420, 30), font, .6,
                [255, 255, 255], 1, cv2.LINE_AA)
    cv2.putText(frame, "{} X {}".format(jogador1_placar, jogador2_placar), (355, 30), font, .6, [
                255, 255, 255], 1, cv2.LINE_AA)  # imprime texto das coordenadas

    print("count_blue_balls", count_blue_balls)
    print("count_red_balls", count_red_balls)
    # writer.write(frame)

cap.release()
cv2.destroyAllWindows()
