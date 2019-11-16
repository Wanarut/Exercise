from keras.models import load_model
from keras import backend as K
import cv2
import numpy as np
import os

IMAGE_SIZE = (512, 512)

def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(
        K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


model = load_model('model_3.h5', custom_objects={'iou': iou})

input_dir = 'Dataset/test/Input'
output_dir = 'Dataset/test/Output'
for file in os.listdir(path=input_dir):
    test_im = cv2.imread(input_dir+'/'+file)
    true_size = test_im.shape
    # imshow_size = (512, round(true_size[0]*512/true_size[1]))
    # cv2.imshow('Input',cv2.resize(test_im, imshow_size))
    # cv2.imshow('Input', test_im)

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # test_im = cv2.medianBlur(test_im, 5)
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    # segmented = np.around(segmented)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')
    # cv2.imshow('Output',cv2.resize(segmented, imshow_size))
    output = cv2.resize(segmented, (true_size[1], true_size[0]))
    output = cv2.inRange(output, 160, 255)
    cv2.imwrite(output_dir+'/'+file, output)
    print(output_dir+'/'+file)
