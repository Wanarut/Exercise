from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

BATCH_SIZE = 5
MAX_EPOCH = 500
IMAGE_SIZE = (512,512)
TRAIN_IM = 455
VALIDATE_IM = 168

model = Sequential()
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

print(model.summary())


def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy',iou])

def myGenerator(type):
    datagen = ImageDataGenerator(rescale=1./255)

    input_generator = datagen.flow_from_directory(
        'Dataset/'+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        'Dataset/'+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

def myGenerator_aug(type):
    datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=10,
                                 zoom_range=[0.8, 1.2],
                                 rotation_range=10,
                                 width_shift_range=0.10,
                                 height_shift_range=0.10)

    input_generator = datagen.flow_from_directory(
        'Dataset/'+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        'Dataset/'+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

checkpoint = ModelCheckpoint('model_3.h5', verbose=1, monitor='val_iou',save_best_only=True, mode='max')


class ShowPredictSegment(Callback):
    def on_epoch_end(self, epoch, logs={}):
        testfileloc = ['Dataset/validation/Input/1.jpg',
                       'Dataset/validation/Input/2.jpg',
                       'Dataset/validation/Input/3.jpg',
                       'Dataset/validation/Input/4.jpg']

        for k in range(len(testfileloc)):
            test_im = cv2.imread(testfileloc[k])
            true_size = test_im.shape
            if true_size[1] >=  true_size[0]:
                imshow_size = (300, round(true_size[0] * 300 / true_size[1]))
            else:
                imshow_size = (round(true_size[1] * 300 / true_size[0]),300)
            cv2.imshow('Input'+str(k), cv2.resize(test_im, imshow_size))
            cv2.moveWindow('Input'+str(k), 20 + 350 * k,10)

            test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
            # test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
            test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            test_im = test_im / 255.
            test_im = np.expand_dims(test_im, axis=0)
            segmented = model.predict(test_im)
            # segmented = np.around(segmented)
            segmented = (segmented[0, :, :, 0] * 255).astype('uint8')
            cv2.imshow('Output'+str(k), cv2.resize(segmented, imshow_size))
            cv2.moveWindow('Output'+str(k), 20 + 350 * k,400)
            cv2.waitKey(100)

show_result = ShowPredictSegment()

h = model.fit_generator(myGenerator_aug('train'),
                        steps_per_epoch=TRAIN_IM/BATCH_SIZE,
                        epochs=MAX_EPOCH,
                        validation_data=myGenerator('validation'),
                        validation_steps=VALIDATE_IM/BATCH_SIZE,
                        callbacks=[checkpoint,show_result])

plt.plot(h.history['iou'])
plt.plot(h.history['val_iou'])
plt.show()
