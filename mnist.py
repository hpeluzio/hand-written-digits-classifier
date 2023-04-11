import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

(X_training, y_training), (X_test, y_test) = mnist.load_data()


# plt.imshow(X_training[3]) # With color
plt.imshow(X_training[3], cmap = 'gray')  # Without color