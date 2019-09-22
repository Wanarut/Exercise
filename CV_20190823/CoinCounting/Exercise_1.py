import numpy as np
import cv2

TARGET_WIDTH = 500.0

# def get_mousepos(event,x,y,flags,param):
#     print(im_hsv[y, x])

# cv2.namedWindow('im_hsv')
# cv2.setMouseCallback('im_hsv', get_mousepos)

for i in range(1,11):
    img = cv2.imread("./coin"+str(i)+".jpg")
    h,w = img.shape[:2]
    img = cv2.resize(img, (int(TARGET_WIDTH), int(TARGET_WIDTH*h/w)))

    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    b_mask = cv2.inRange(im_hsv, (91,95,100), (108,255,255))

    y_mask = cv2.inRange(im_hsv, (25,100,100), (30,230,255))
    y_mask = cv2.medianBlur(y_mask, 3)

    masks = [b_mask, y_mask]
    num = [0,0]
    
    for i in range(len(masks)):
        size = np.sum(masks[i])
        iter = size/551787
        kernel = np.ones((3,3),np.uint8)
        masks[i] = cv2.erode(masks[i], kernel, iterations = iter)

        masks[i] = cv2.morphologyEx(masks[i],cv2.MORPH_CLOSE,kernel, iterations = iter)
        masks[i] = cv2.morphologyEx(masks[i],cv2.MORPH_OPEN,kernel, iterations = iter/4)
        masks[i] = cv2.medianBlur(masks[i], 7)

        contours,hierarchy = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num[i] = len(contours)

        for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255*(abs(i-1)), 255*i), 2)

        
    print('Blue coin:' + str(num[0]) + ' Yellow coin:' + str(num[1]))
    cv2.imshow('im_hsv', im_hsv)
    cv2.imshow('img', img)
    cv2.imshow("masks", b_mask+y_mask)

    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if k == ord('q'):
        break