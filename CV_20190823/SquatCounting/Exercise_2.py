import numpy as np
import cv2

# cap = cv2.VideoCapture('Squat1_8_9.avi')
# cap = cv2.VideoCapture('Squat2_16_17.avi')
cap = cv2.VideoCapture('Squat3_11_9_10.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

_, bg = cap.read()

prex = [0, 0, 0]
sited = [False, False, False]
n_sit = [0, 0, 0]
count = 0
while(cap.isOpened()):
    haveFrame, img = cap.read()

    if (not haveFrame) or (cv2.waitKey(int(500/fps)) & 0xFF == ord('q')):
        break
    diffc = cv2.absdiff(img, bg)
    diffg = cv2.cvtColor(diffc, cv2.COLOR_BGR2GRAY)
    bmask = cv2.inRange(diffg, 35, 255)
    # bwmask = bmask.copy()

    kernel = np.ones((5, 5), np.uint8)
    bwmask = cv2.morphologyEx(bmask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((30, 20), np.uint8)
    bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((5, 10), np.uint8)
    bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((25, 3), np.uint8)
    bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel, iterations=9)

    contours, hierarchy = cv2.findContours(
        bwmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    boundingBoxes.sort()

    if(count > 250 and count < length-120):
        for i in range(len(boundingBoxes)):
            x, y, w, h = boundingBoxes[i]
            color = [0, 0, 0]
            color[i % 3] = 255
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            if(count % 90 == 0):
                # print(abs(prex[i]-x))
                prex[i % 3] = x
            if(abs(prex[i % 3]-x) < 20):

                if(y > 0):
                    if(not sited[i % 3]):
                        n_sit[i % 3] += 1
                        print(n_sit)
                        sited[i % 3] = True
                else:
                    sited[i % 3] = False

    count += 1

    cv2.imshow('bmask', bmask)
    cv2.imshow('bwmask', bwmask)
    cv2.imshow('img', img)
