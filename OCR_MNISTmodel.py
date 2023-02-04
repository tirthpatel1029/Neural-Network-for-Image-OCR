import neuralnetwork as NN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn import datasets
import math

############################################################################

# Code to visualize results
# go through each line of code and try to understand what it is doing

def find_labels(prediction_array, true_label_array):
  true_index = 0
  predicted_index = 0
  max = prediction_array[0]
  for j in range(len(true_label_array)):
    if true_label_array[j] == 1:
        true_index = j
    if prediction_array[j] > max:
        max = prediction_array[j]
        predicted_index = j

  return true_index,predicted_index


def plot_image(prediction_array, true_label_array, img):
  
  true_label, predicted_label = find_labels(prediction_array,true_label_array)
  img_length = int(math.sqrt(len(img)))
  img = img.reshape(img_length,img_length)

  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")

  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(prediction_array), # need to put in actual percentage confidence here
                                class_names[true_label]),
                                color=color)

def plot_value_array(prediction_array, true_label_array):
  
  true_label, predicted_label = find_labels(prediction_array,true_label_array)

  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), prediction_array, color="#777777")
  plt.ylim([0, 1])

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#####################################################################

# load the MNIST dataset and apply min/max scaling to scale the 9 # pixel intensity values to the range [0, 1] (each image is represented by an 8 x 8 = 64-dim feature vector)

print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))

# construct the training and testing splits
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
MNISTmodel = NN.NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(MNISTmodel))
MNISTmodel.train(trainX, trainY, epochs=50)

# evaluate the network
print("[INFO] evaluating network...")
predictions = MNISTmodel.predict(testX)


# Plot the first X (in our case, we did 5 X 3) test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  
  #review the plot_image function above
  plot_image(predictions[i], testY[i], testX[i])

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  #review the plot_value_array function above
  plot_value_array(predictions[i], testY[i])

plt.tight_layout()
plt.show()

#summary of performance
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))