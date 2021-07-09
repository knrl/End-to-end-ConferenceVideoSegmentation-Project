import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization  
from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

class Unet:
    def __init__(self, height, width, seed=0):
        # input parameters
        self.height = height
        self.width = width
        self.channels = 1
        self.inputs = keras.layers.Input((self.height, self.width, self.channels))

        # network parameters
        self.kernel3 = (3, 3)
        self.size = [32, 64, 128, 256, 512]
        self.relu = 'relu'
        self.sigmoid = 'sigmoid'
        self.padding = 'same'
        self.strides = (2, 2)
        self.kernel_initializer = tf.keras.initializers.he_normal(seed=seed)#'he_normal'

    def encode(self, x, c, size):
        u = Conv2DTranspose(size, (2, 2), strides=self.strides, padding=self.padding)(x)
        x = concatenate([c, u], axis=3)

        x = Conv2D(size, self.kernel3, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Activation(self.relu)(x)
        x = BatchNormalization()(x)
        return x

    def decode(self, x, size):
        x = Conv2D(size, self.kernel3, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Activation(self.relu)(x)

        x = Conv2D(size, self.kernel3, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Activation(self.relu)(x)
        x = BatchNormalization()(x)

        p = MaxPooling2D((2, 2))(x)
        return x, p

    def bottleneck(self, x, size):
        x = Conv2D(size, self.kernel3, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Activation(self.relu)(x)

        x = Conv2D(size, self.kernel3, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Activation(self.relu)(x)
        return x

    def get_model(self):
        c1, p1 = self.decode(self.inputs, self.size[0])
        c2, p2 = self.decode(p1, self.size[1])
        c3, p3 = self.decode(p2, self.size[2])
        c4, p4 = self.decode(p3, self.size[3])

        bn = self.bottleneck(p4, self.size[4])

        u1 = self.encode(bn, c4, self.size[3])
        u2 = self.encode(u1, c3, self.size[2])
        u3 = self.encode(u2, c2, self.size[1])
        u4 = self.encode(u3, c1, self.size[0])

        mask = Conv2D(1, (1, 1), padding=self.padding, activation=self.sigmoid)(u4)

        model = keras.models.Model(self.inputs, mask)
        return model
