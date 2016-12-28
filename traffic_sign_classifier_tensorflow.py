# Dataset: German Traffic Sign Benchmarks (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
# Accuracy: Validation (≈ 0.99), Test (≈ 0.91), New images collected manually (≈ 0.40)
import pickle
training_file = 'train.p'
testing_file = 'test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
assert(len(X_train) == len(y_train))
n_train = len(X_train)
assert(len(X_test) == len(y_test))
n_test = len(X_test)
image_shape = X_train[0].shape

import numpy as np
n_classes = np.max(y_train) + 1
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import cv2
def normalize(img):
    return cv2.normalize(img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

print('Creating distribution of labels in Training data...')
inputs_per_class = np.bincount(y_train)
max_inputs_for_class = np.max(inputs_per_class)
min_inputs_for_class = np.min(inputs_per_class)
print("Max inputs for a class: {}".format(max_inputs_for_class))
print("Min inputs for a class: {}".format(min_inputs_for_class))
plt.figure()
plt.bar(range(len(inputs_per_class)), inputs_per_class, 1/3, color='black', label='Inputs per class')
plt.show()

for i in range(n_classes):
    for j in range(len(y_train)):
        if (i%10 == 0 and i == y_train[j]):
            print('Class: ', i)
            plt.figure(figsize=(1,1))
            plt.imshow(X_train[j])
            plt.show()
            print('Normalized image: ')
            plt.figure(figsize=(1,1))
            plt.imshow(normalize(X_train[j]))
            plt.show()
            break

import scipy.ndimage
print('Size of training set: {}'.format(n_train))
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
inputs_per_class = np.bincount(y_train)
plt.figure()
plt.bar(range(len(inputs_per_class)), inputs_per_class, 1/3, color='black', label='Inputs per class')
plt.show()

X_train = np.array([normalize(image) for image in X_train], dtype=np.float32)

from sklearn.cross_validation import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
assert(len(X_validation) == len(y_validation))
n_validation = len(X_validation)
print("Number of training examples =", len(X_train))
print("Number of validation examples =", n_validation)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 250

from tensorflow.contrib.layers import flatten

conv1_W = tf.get_variable('conv1W', shape=[5,5,3,6], initializer=tf.contrib.layers.xavier_initializer())
conv1_b = tf.get_variable('conv1b', shape=[6], initializer=tf.contrib.layers.xavier_initializer()) 
conv2_W = tf.get_variable('conv2W', shape=[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer())
conv2_b = tf.get_variable('conv2b', shape=[16], initializer=tf.contrib.layers.xavier_initializer())
fc1_W = tf.get_variable('fc1W', shape=[400,120], initializer=tf.contrib.layers.xavier_initializer())
fc1_b = tf.get_variable('fc1b', shape=[120], initializer=tf.contrib.layers.xavier_initializer())
fc2_W = tf.get_variable('fc2W', shape=[120,84], initializer=tf.contrib.layers.xavier_initializer())
fc2_b = tf.get_variable('fc2b', shape=[84], initializer=tf.contrib.layers.xavier_initializer())
fc3_W = tf.get_variable('fc3W', shape=[84,43], initializer=tf.contrib.layers.xavier_initializer())
fc3_b = tf.get_variable('fc3b', shape=[43], initializer=tf.contrib.layers.xavier_initializer())
    
def LeNet(x):
    mu = 0
    sigma = 0.1    
    
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1) 
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding='VALID') + conv2_b    
    conv2 = tf.nn.relu(conv2)    
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    fc0 = flatten(conv2)    
    
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)
       
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b    
    fc2 = tf.nn.relu(fc2)    
    fc2 = tf.nn.dropout(fc2, 0.5)
    
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y,43)

rate = 0.001
logits = LeNet(x)
softmax_out = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy/num_examples

num_examples = len(X_train)
print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset+BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
    training_accuracy = evaluate(X_train, y_train)
    validation_accuracy = evaluate(X_validation, y_validation)
    if ((i+1)%10 == 0):
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

test_accuracy = evaluate(X_test, y_test)
print("Test Accuracy = {:.3f}".format(test_accuracy))

import matplotlib.image as mpimg
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
predictions = sess.run(softmax_out, feed_dict={x: new_input})
print(sess.run(tf.nn.top_k(tf.constant(predictions), k=3)))