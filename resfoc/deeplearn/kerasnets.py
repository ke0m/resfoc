"""
Functions for building deep neural networks using Keras

@author: Joseph Jennings
@version: 2020.05.12
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Conv3D
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation, Cropping2D, concatenate
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, MaxPooling3D, UpSampling2D, LeakyReLU, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from deeplearn.keraslosses import cross_entropy_balanced
from tensorflow.keras.losses import categorical_crossentropy

def conv2d(layer_input, filters, f_size, alpha=0.2, bn=True, dropout=0.0):
  """ A wrapper for Keras Conv2D function """
  d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
  d = LeakyReLU(alpha=alpha)(d)
  if(dropout):
    d = Dropout(dropout)(d)
  if(bn):
    d = BatchNormalization(momentum=0.8)(d)
  return d

def deconv2d(layer_input, filters, f_size, alpha=0.2, skip_input=None, bn=True, dropout=0.0):
  """ A wrapper for Keras upsampled convolution """
  u = UpSampling2D(size=2)(layer_input)
  u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
  u = LeakyReLU(alpha=alpha)(u)
  if(dropout):
    u = Dropout(dropout)(u)
  if(bn):
    u = BatchNormalization(momentum=0.8)(u)
  if(skip_input != None):
    u = Concatenate()([u, skip_input])
  return u

def conv2dt(layer, filters, f_size, alpha=0.2, skip_input=None, bn=True, dropout=0.0):
  """ A wrapper for Keras transposed convolution """
  u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(u)
  u = LeakyReLU(alpha=alpha)(u)
  if(dropout):
    u = Dropout(dropout)(u)
  if(bn):
    u = BatchNormalization(momentum=0.8)(u)
  if(skip_input != None):
    u = Concatenate()([u, skip_input])
  return u

def auto_encoder(imgshape, nc_in, nc_out, nf, ksize, unet=True, dropout=0.0, reg=True):
  """ Returns a Keras autoencoder model. By default gives a u-net
  with skip connections """

  inp = Input(shape=(imgshape[0],imgshape[1],nc_in), name='input')

  # If a classification problem, set LeakyReLU to ReLU
  if(reg):
    alpha = 0.2
  else:
    alpha = 0.0

  # Encoder
  d1 = conv2d(inp, nf, ksize, bn=False)
  d2 = conv2d(d1, nf *  2, ksize, alpha)
  d3 = conv2d(d2, nf *  4, ksize, alpha)
  d4 = conv2d(d3, nf *  8, ksize, alpha)
  d5 = conv2d(d4, nf *  8, ksize, alpha)

  # Decoder
  u5 = None
  if(unet):
    u1 = deconv2d(d5, nf *  8, ksize, alpha, d4)
    u2 = deconv2d(u1, nf *  8, ksize, alpha, d3)
    u3 = deconv2d(u2, nf *  4, ksize, alpha, d2)
    u4 = deconv2d(u3, nf *  2, ksize, alpha, d1)
    u5 = deconv2d(u4, nf *  1, ksize, alpha)
  else:
    u1 = deconv2d(d5, nf *  8, ksize, alpha)
    u2 = deconv2d(u1, nf *  8, ksize, alpha)
    u3 = deconv2d(u2, nf *  4, ksize, alpha)
    u4 = deconv2d(u3, nf *  2, ksize, alpha)
    u5 = deconv2d(u4, nf *  1, ksize, alpha)

  if(reg == False):
    out = Conv2D(nc_out, kernel_size=ksize, strides=1, padding='same', activation='sigmoid', name='output')(u5)
  else:
    out = Conv2D(nc_out, kernel_size=ksize, strides=1, padding='same', activation='relu', name='output')(u5)

  return Model(inp,out)

def unetxwu(pretrained_weights = None,input_size = (128,128,1)):
  inputs = Input(input_size)
  conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

  conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

  conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

  conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)

  up5 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
  conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(up5)
  conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(conv5)

  up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv2], axis=-1)
  conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(conv6)

  up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv1], axis=-1)
  conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(conv7)

  conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)

  model = Model(inputs=[inputs], outputs=[conv8])
  print(model.summary())
  model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])

  return model

def findres(input_size = (64,64,19)):
  inputs = Input(input_size)
  conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

  conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

  conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

  conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)

  flat   = Flatten()(conv4)
  dense1 = Dense(64, kernel_initializer='normal', activation='relu')(flat)
  drop1  = Dropout(0.1)(dense1)

  #dense1a = Dense(512, kernel_initializer='normal', activation='relu')(drop1)
  #drop1a = Dropout(0.1)(dense1a)

  #dense2 = Dense(input_size[2], kernel_initializer='normal', activation='softmax')(flat)
  dense2 = Dense(input_size[2], kernel_initializer='normal', activation='softmax')(drop1)
  drop2  = Dropout(0.1)(dense2)

  model = Model(inputs=[inputs], outputs=[drop2])
  print(model.summary())
  model.compile(optimizer = Adam(lr = 1e-4), loss = categorical_crossentropy , metrics = ['accuracy'])

  return model

def vgg3(pretrained_weights=None, input_size=(64,64,1)):
  inputs = Input(input_size)
  conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling2D(pool_size=(2,2))(conv1) # 64 -> 32

  conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling2D(pool_size=(2,2))(conv2) # 32 -> 16

  conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
  pool3 = MaxPooling2D(pool_size=(2,2))(conv3) # 16 -> 8

  flat   = Flatten()(pool3)
  dense1 = Dense(128,activation='relu')(flat)
  acto   = Dense(1,activation='sigmoid')(dense1)

  model = Model(inputs=[inputs], outputs=[acto])
  model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])

  return model

def vgg3_3d(pretrained_weights=None, input_size=(64,64,64,1)):
  inputs = Input(input_size)
  conv1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1) # 64 -> 32

  conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2) # 32 -> 16

  conv3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool2)
  pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3) # 16 -> 8

  flat   = Flatten()(pool3)
  dense1 = Dense(128,activation='relu')(flat)
  acto   = Dense(1,activation='sigmoid')(dense1)

  model = Model(inputs=[inputs], outputs=[acto])
  model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])

  return model

