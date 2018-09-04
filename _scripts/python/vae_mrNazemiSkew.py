import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import h5py
from PIL import Image as im
#from scipy.ndimage import interpolation as inter
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time
import scipy.io


# load mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert y to one-hot, reshape x
#y_train = y_train.reshape((len(y_train), np.prod(y_train.shape[1:])))
y_train = to_categorical(y_train)

y_test2=y_test
y_test = to_categorical(y_test)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# DeSkew Digits

f = scipy.io.loadmat(r'/home/amir/PycharmProjects/CVAE/XtrainDeSkew.mat')
data = f.get('XtrainDeSkew')
XtrainDeSkew = np.array(data)
XtrainDeSkew = XtrainDeSkew.astype(np.float32)
XtrainDeSkew = XtrainDeSkew /255
print(XtrainDeSkew.shape)

f = scipy.io.loadmat(r'/home/amir/PycharmProjects/CVAE/XtestDeSkew.mat')
data = f.get('XtestDeSkew')
XtestDeSkew = np.array(data)
XtestDeSkew = XtestDeSkew.astype(np.float32)
XtestDeSkew = XtestDeSkew /255
print(XtestDeSkew.shape)


f = scipy.io.loadmat(r'/home/amir/PycharmProjects/CVAE/XtestTheta.mat')
data = f.get('XtestTheta')
XtestTheta = np.array(data)
print(XtestTheta.shape)


f = scipy.io.loadmat(r'/home/amir/PycharmProjects/CVAE/XtrainTheta.mat')
data = f.get('XtrainTheta')
XtrainTheta = np.array(data)
print(XtrainTheta.shape)

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 2

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]

# nubmer of epochs
n_epoch = 50

##  ENCODER ##

# encoder inputs
X = Input(shape=(784, ))
# cond = Input(shape=(n_y, ))

# merge pixel representation and label
# inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu')(X)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent spa ce
z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])

# merge latent space with label
# z_cond = concatenate([z, cond])

##
def sample_ztest(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(1,n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


##  DECODER  ##

# dense ReLU to sigmoid layers
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')
h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model(X, outputs)
encoder = Model(X, mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    bin_y_pred = K.reshape(1 - y_pred , (-1 , 784))
    sharpness_m = K.sum(bin_y_pred , axis=1)
    return recon + kl #- sharpness_m

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))# - sharpness(y_pred))

def sharpness(y_true,y_pred):
    tt = K.reshape(1-y_pred , (-1 , 784))
    return(K.sum(tt, axis=1))

def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)

# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics=[KL_loss, recon_loss])  # , sharpness])
cvae_hist = cvae.fit(X_train, XtrainDeSkew, batch_size=m, epochs=n_epoch,
                     validation_data=(X_test[1000:], XtestDeSkew[1000:]),
                     callbacks=[EarlyStopping(patience=5)])
'''
for i in range(999):#X_test.shape[0]):
    if  np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax() != 20:
        ztmp = encoder.predict(np.reshape(np.transpose(X_test[i,:]),[1,X_test.shape[1]]))
        print(ztmp)
        dtmp = ztmp
        generated = decoder.predict(dtmp)
        file_name = './MeEnhanced/img' + str(i) + '_' +str(np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax())+ '.jpg'
        file_name_input = './MeEnhanced/img' + str(i) + '_' +str(np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax())+ '_input.jpg'
        file_name_target = './MeEnhanced/img' + str(i) + '_' + str(np.reshape(np.transpose(y_test[i, :]), [1, y_test.shape[1]]).argmax()) + '_target.jpg'
        imsave(file_name, generated.reshape((28, 28)))
        imsave(file_name_input, np.reshape(np.transpose(X_test[i,:]),[1,X_test.shape[1]]).reshape((28, 28)))
        imsave(file_name_target,np.reshape(np.transpose(XtestDeSkew[i, :]), [1, XtestDeSkew.shape[1]]).reshape((28, 28)))
        time.sleep(0.1)
# this loop prints the one-hot decodings
'''
with open("variationalSkew.txt", "a") as myfile:
    for i in range(999):#X_test.shape[0]):
        ztmp = encoder.predict(np.reshape(np.transpose(X_test[i, :]), [1, X_test.shape[1]]))
        print(ztmp)
        myfile.write(str(ztmp[0][:])+str(np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax())+'\n')
        dtmp = ztmp
        generated = decoder.predict(dtmp)
        file_name = './MeEnhanced/img' + str(i) + '_' +str(np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax())+ '.jpg'
        file_name_input = './MeEnhanced/img' + str(i) + '_' +str(np.reshape(np.transpose(y_test[i,:]),[1,y_test.shape[1]]).argmax())+ '_input.jpg'
        file_name_target = './MeEnhanced/img' + str(i) + '_' + str(np.reshape(np.transpose(y_test[i, :]), [1, y_test.shape[1]]).argmax()) + '_target.jpg'
        imsave(file_name, generated.reshape((28, 28)))
        imsave(file_name_input, np.reshape(np.transpose(X_test[i,:]),[1,X_test.shape[1]]).reshape((28, 28)))
        imsave(file_name_target,np.reshape(np.transpose(XtestDeSkew[i, :]), [1, XtestDeSkew.shape[1]]).reshape((28, 28)))
        time.sleep(0.1)