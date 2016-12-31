import cv2
import json
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import imread
from scipy.misc import imresize
get_ipython().magic('matplotlib inline')

driving_log = pandas.read_csv('data/driving_log.csv')
print(driving_log.head())

plt.hist(driving_log['steering'], bins=30, rwidth=2/3, color='black', range=(-0.4, 0.4))
plt.title('Steering angle distribution')
plt.show()

print('Loading center, left and right images...')
X_train = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['center']])
X_left = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['left']])
X_right = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['right']])

y_train = np.asarray(driving_log['steering'])

assert(len(X_train) == len(y_train))
print('X_train.shape:', X_train.shape)

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    # using small angle approximaton for tan(change in steering)
    dsteering = dx/rows   
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering  
    return image,steering

def crop_top(image):
    return image[50:-10,:]

def normalize(image):
    return image/255 - 0.5

def read_next_image(m,lcr,X_train,y_train):
    steering = y_train[m]
    if lcr == 0:
        image = X_left[m]
        steering += 0.25
    elif lcr == 1:
        image = X_train[m]
    elif lcr == 2:
        image = X_right[m]
        steering -= 0.25
    else:
        print ('Invalid lcr value :',lcr )
    return image,steering

def generate_training_example(X_train,X_left,X_right,y_train):
    m = np.random.randint(0,len(X_train))
    lcr = np.random.randint(0,3)
    image,steering = read_next_image(m,lcr,X_train,y_train)
    image = crop_top(image)
    image = random_brightness(image)
    image,steering = random_flip(image,steering)
    #image,steering = random_shear(image,steering,shear_range=100)
    image = imresize(image, (66,200,3))
    image = normalize(image)
    return image,steering

'''image, steering = generate_training_example(X_train,X_left,X_right,y_train)

plt.imshow(image)
print(image.shape)
print(steering)'''

def generate_train_batch(X_train,X_left,X_right,y_train,batch_size = 32):    
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_example(X_train,X_left,X_right,y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
        
batch_size=200
train_generator = generate_train_batch(X_train,X_left,X_right,y_train,batch_size)

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
model.add(Dense(1))

model.summary()

adam = Adam(lr=1e-4)

from keras.models import model_from_json
model_json = 'model.json'
model_weights = 'model.h5'

restart=False
if os.path.isfile(model_json) and restart:
    try:
        with open(model_json) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(model_weights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model:', e)
        raise 

model.compile(optimizer=adam,
              loss='mse',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    samples_per_epoch=40000,
                    nb_epoch=5,
                    verbose=1)

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass 


model_json_string = model.to_json()
with open(model_json, 'w') as outfile:
    json.dump(model_json_string, outfile)
model.save_weights(model_weights)

ximg = imresize(X_train[10], (66,200,3))
plt.imshow(ximg)
ximg = ximg[None, :, :, :]
print(ximg.shape)
model.predict(ximg, batch_size=1, verbose=1)
angle = float(model.predict(ximg, batch_size=1, verbose=1))
print('Prediction:{}'.format(angle))
print('Actual steering angle:{}'.format(y_train[10]))