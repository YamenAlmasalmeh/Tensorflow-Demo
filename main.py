#Import the required libraries
import tensorflow as tf
from tensorflow import keras

#Use to load in the data
from tensorflow.examples.tutorials.mnist import input_data

#Import other dependencies
import numpy as np
import matplotlib.pyplot as plt


#Load the data
data = input_data.read_data_sets('data/fashion')

train_images, train_labels = data.test.next_batch(60000)
test_images, test_labels = data.train.next_batch(10000)

train_images = np.array([i.reshape(28,28) for i in train_images])
test_images = np.array([i.reshape(28,28) for i in test_images])

#For each clothing object, the name is going to be assosciated with its index
#I.e. if the model returns 3 we know it's dress
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Creates the model, first flattens all inputs from matrices to arrays, then adds two layers to the NN
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print("shape of the training data", train_images.shape)


#Compiles the model so it can be used, loss is the function used to determine the accuracy while training
#Metrics is what's monitored during the training
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Train the model with the given images and their respective tags
#Epochs: number of times data passed through the neural network
model.fit(train_images, train_labels, epochs=5, verbose=0)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy", test_acc)

predictions = model.predict(test_images)

#Prints the formatted image nicely
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#Prints the prediction array for the image
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Prints the first 15 images and their predictions
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
