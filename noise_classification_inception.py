from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
#input_shape=X_train[0].shape
import keras

base = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(100,100,3))
X = base.output
X = keras.layers.Flatten()(X)
X = keras.layers.Dense(512, activation = 'relu')(X)
X = keras.layers.Dropout(0.5)(X)
X = keras.layers.BatchNormalization()(X)
X = keras.layers.Dense(256, activation = 'relu')(X)
X = keras.layers.Dropout(0.5)(X)
X = keras.layers.BatchNormalization()(X)
preds = keras.layers.Dense(9, activation = 'softmax')(X)

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model = keras.models.Model(inputs = base.input, outputs = preds)
model.compile(optimizer = opt , loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
