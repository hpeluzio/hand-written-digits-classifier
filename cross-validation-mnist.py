from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.layers.normalization.batch_normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X , y),  (X_test, y_test) = mnist.load_data()

predictors = X.reshape(X.shape[0], 28, 28, 1)
predictors = predictors.astype('float32')
predictors /= 255

classe = np_utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state =seed ) # In most science articles is used 10

results = []


a = np.zeros(5)
b = np.zeros(shape = (classe.shape[0], 1))

for index_training, index_test in kfold.split(predictors, 
                                              np.zeros(shape = (classe.shape[0], 1))):
    # print('Index training', index_training, 'Index test', index_test)
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten()) # Use just one time    
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    classifier.compile(loss = 'categorical_crossentropy',
                       optimizer = 'adam',
                       metrics = ['accuracy'])
    classifier.fit(predictors[index_training], classe[index_training], 
                   batch_size = 128, epochs = 5)
    precision = classifier.evaluate(predictors[index_test], classe[index_test])
    results.append(precision[1])
    
    
#mean = result.mean()
mean = sum(results) / len(results)
    

