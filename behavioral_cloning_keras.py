import pandas
driving_log = pandas.read_csv('data/driving_log.csv')
print(driving_log.head())

from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(driving_log['steering'], bins=30, rwidth=2/3, color='black', range=(-0.4, 0.4))
plt.title('Steering angle distribution')
plt.show()

import numpy as np
X_train = [imread('data/'+file_name, mode='RGB') for file_name in driving_log['center']]
X_train = np.asarray(X_train)
print('X_train.shape:{}'.format(X_train.shape))
plt.imshow(X_train[0])

def normalize(image):
    return imresize(image/255-0.5, (66,200,3))

print('Preprocessing training data...')
X_train = np.asarray([normalize(image) for image in X_train])

plt.imshow(X_train[0])

y_train = driving_log['steering']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print('X_train.shape:{}'.format(X_train.shape))
print('y_train.shape:{}'.format(y_train.shape))
print('X_val.shape:{}'.format(X_val.shape))
print('y_val.shape:{}'.format(y_val.shape))

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid', input_shape=(66, 200, 3)))
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(Conv2D(64, 3, 3, subsample=(1,1)))
model.add(Conv2D(64, 3, 3, subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1, activation='tanh'))

model.summary()

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=200, nb_epoch=10, verbose=1, validation_data=(X_val,y_val))

print('val_acc:', history.history['val_acc'][-1])
print('acc:', history.history['acc'][-1])

import json
model_json = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(model_json, outfile)
model.save_weights('model.h5')