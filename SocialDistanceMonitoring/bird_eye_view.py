import cv2 # importing opencv2 library
import numpy as np # importing numpy

# function generate bird eye view image
def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
    h = frame.shape[0] # image height
    w = frame.shape[1] # image weight

    # colour combination
    red = (0, 0, 255) 
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    # create empty white image
    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    r = []
    g = []
    y = []
    # separate high risk , low risk , no risk people based on risk factor
    for i in range(len(distances_mat)):
        # high risk people
        if distances_mat[i][2] == 0:
            # person 1
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            # person 2
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])
            # draw the line between risk people
            blank_image = cv2.line(blank_image,
                                   (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)),
                                   (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)), red,
                                   2)

    # separate low risk
    for i in range(len(distances_mat)):
        # low  risk people
        if distances_mat[i][2] == 1:
            # person 1
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            # person 2
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
            # draw the line between risk people
            blank_image = cv2.line(blank_image,
                                   (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)),
                                   (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)),
                                   yellow, 2)
    # separate low risk
    for i in range(len(distances_mat)):
        # no risk people
        if distances_mat[i][2] == 2:
            # person 1
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            # person 2
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    # draw green circle for each no risk person
    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    # draw yellow circle for each low risk people
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    # draw the red circle for each high risk person
    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, red, 10)
    # return the generated frame
    return blank_image

## main

#bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)

#cv2.imshow('Bird Eye View', bird_image)