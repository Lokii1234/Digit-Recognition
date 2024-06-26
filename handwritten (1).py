import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import keras
from keras.datasets import mnist



(X_train, y_train),(X_test,y_test) = mnist.load_data()

X_train.shape , y_train.shape, X_test.shape, y_test.shape

def plot_input_img(i):
    plt.imshow(X_train[0], cmap='binary')
    plt.title(y_train[i])
    
    plt.show()

#for i in range(10):
#   plot_input_img(i)

#pre process the images

# normalize the image to [0,1] range
X_train = X_train.astype(np.float32)/255
X_test  = X_test.astype(np.float32)/255
#Reshape/expand the dimaneions 
X_train = np.expand_dims(X_train,-1)
X_test  = np.expand_dims(X_test, -1)

# convert classes to one hot vectors

y_train = keras.utils.to_categorical(y_train)
y_test  = keras.utils.to_categorical(y_test)
# building model
model = Sequential()
model.add(Conv2D(32, (3,3),input_shape = (28,28,1) , activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3) , activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer= 'adam', loss = keras.losses.categorical_crossentropy , metrics= ['accuracy'])

#callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
#Earlystopping
es = EarlyStopping(monitor='val_accuracy',min_delta=0.01, patience=4,verbose=1)

#model chackpoint

mc = ModelCheckpoint("./bestmodel.h5",monitor="val_accuracy", verbose= 1, save_best_only=True)

cb = [es,mc]

#train
his = model.fit(X_train, y_train, epochs=5, validation_split=0.3, callbacks=cb)

model_S  = keras.models.load_model("bestmodel.h5") # bestmodel.h5 file path

score = model_S.evaluate(X_test,y_test)
print(f" the model accuracy is {score[1]}")
