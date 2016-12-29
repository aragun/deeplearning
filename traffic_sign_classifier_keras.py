# Dataset: German Traffic Sign Benchmarks (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
# Accuracy: Validation (≈ 0.99), Test (≈ 0.94), New images collected manually (≈ ?)
import scipy.misc
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math

with open('train.p', 'rb') as f:
    train = pickle.load(f)
with open('test.p', 'rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np
inputs_per_class = np.bincount(y_train)
max_inputs_for_class = np.max(inputs_per_class)

import scipy.ndimage
print('Size of training set: {}'.format(len(y_train)))
print('Augmenting data...')
for i in range(len(inputs_per_class)):
    current_count = inputs_per_class[i]
    
    if (current_count == max_inputs_for_class):
        continue
    
    new_features = []
    new_labels = []
    mask = np.where(y_train == i)
    angle = -15
    
    while (current_count < max_inputs_for_class):
        for feature in X_train[mask]:
            new_features.append(scipy.ndimage.rotate(feature, angle, reshape=False))
            new_labels.append(i)
            current_count += 1
            if (current_count >= max_inputs_for_class):
                break
        angle += 5
    
    X_train = np.append(X_train, new_features, axis=0)
    y_train = np.append(y_train, new_labels, axis=0)
print("Size of training set after data augmentation: {}".format(len(y_train)))

X_train = np.array([image/255.-0.5 for image in X_train], dtype=np.float32)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(math.isclose(np.min(X_train), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_train), 0.5, abs_tol=1e-5)), "The range of the training data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))
assert(math.isclose(np.min(X_val), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_val), 0.5, abs_tol=1e-5)), "The range of the validation data is: %.1f to %.1f" % (np.min(X_val), np.max(X_val))

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

print('Training set size: {}', len(X_train))
print('Validation set size: {}', len(X_val))
print('Test set size: {}', len(X_test))

y_train = np_utils.to_categorical(y_train, 43)
y_val = np_utils.to_categorical(y_val, 43)

batch_size = 250
nb_epoch = 50

model = Sequential()
model.add(Conv2D(6, 5, 5, input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, 5, 5, input_shape=(14, 14, 6), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add((Dropout(0.5)))
model.add(Dense(43, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train,
                  y_train,
                  batch_size,
                  nb_epoch,
                  verbose=0,
                  validation_data=(X_val,y_val))

print('val_acc:', history.history['val_acc'][-1])
print('acc:', history.history['acc'][-1])

X_test = np.array([image/255.-0.5 for image in X_test], dtype=np.float32)
y_test = np_utils.to_categorical(y_test, 43)
model.evaluate(X_test, y_test)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
imgs = ['120.jpg', 'pedestrian.png', 'large_vehicle_ban.png', 'yield.png', 'road_work.png', 'slippery_road.png']
new_input = []
for imgname in imgs:
    image = scipy.misc.imread(imgname, mode='RGB')
    
    print(image.shape)
    plt.figure(figsize=(1,1))
    plt.imshow(image)
    plt.show()
    
    image = scipy.misc.imresize(image, (32,32,3))
    print(image.shape)
    plt.figure(figsize=(1,1))
    plt.imshow(image)
    plt.show()
    new_input.append(image)
    print(len(new_input))
    
new_input = np.array(new_input, dtype=np.float32)
new_input_answers = np.array([8,27,10,13,25,23])
new_input = np.array([image/255.-0.5 for image in new_input], dtype=np.float32)
new_input_answers = np_utils.to_categorical(new_input_answers, 43)
model.evaluate(new_input, new_input_answers)  