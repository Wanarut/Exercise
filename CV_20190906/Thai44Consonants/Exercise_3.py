import numpy as np
import cv2
import os
count = 0

#hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
hog = cv2.HOGDescriptor((50, 50), (20, 20), (10, 10), (10, 10), 9)
#WinSize, BlockSize, BlockStride, CellSize, NBins

label_train = np.zeros((5243, 1))

for char_id in range(0, 44):
    dir = './Train/'+str(char_id)
    print(dir)
    for file in os.listdir(path=dir):
        file = dir+'/'+file
        im = cv2.imread(file, 0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)

        if count == 0:
            features_train = h.reshape(1, -1)
        else:
            features_train = np.concatenate(
                (features_train, h.reshape(1, -1)), axis=0)

        label_train[count] = char_id
        count = count+1

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setDegree(20)
# svm.setGamma(0.15)
svm.train(features_train.astype(np.float32),
          cv2.ml.ROW_SAMPLE, label_train.astype(np.int32))

correct = 0
total = 0
for char_id in range(0, 44):
    dir = './Test/'+str(char_id)
    print(dir)
    for file in os.listdir(path=dir):
        file = dir+'/'+file
        im = cv2.imread(file, 0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)
        result = svm.predict(h.reshape(1, -1).astype(np.float32))[1]
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if char_id == result[0][0].astype(int):
            correct += 1
        total += 1

        #     im[:,:,2] = 255
        #     cv2.putText(im, charlist[result[0][0].astype(int)] , (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        # cv2.imshow(str(im_id) + "=" + charlist[result[0][0].astype(int)], cv2.resize(im, (100, 100)))
        # cv2.moveWindow(str(im_id) + "=" + charlist[result[0][0].astype(int)], 100 + ((im_id - 1) % 5) * 120, np.floor((im_id - 1) / 5).astype(int) * 150)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print(correct*100.0/total,'%')
