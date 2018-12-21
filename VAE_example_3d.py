# Simple VAE implementation
# Read MNIST dataset images, train a VAE, generate MNIST-like images
from __future__ import  division, print_function
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

import ROOT
import root_numpy

K.set_image_data_format('channels_last')

#parameters
#batch_size = 16
batch_size = 2
encoder_dim = 32
latent_dim = 2  # Dimensionality of the latent space (a cube)
img_shape = (128,128,128,2)#(128, 128, 128, 3) # shape of the input images (MNIST 25x25 pixels x 1 (gray scale))

# Latent space sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

# custom VAE loss function (needed as VAE dual loss dosen't fit traditional keras loss loss(input, output))
class CustomVariationalLayer(keras.layers.Layer):


    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


# Encoder:  Image --> Latent Space
input_img = keras.Input(shape=img_shape)
#x = layers.Conv3D(8, 3, padding='same', activation='relu')(input_img)
#x = layers.Conv3D(8, 3, padding='same', activation='relu', strides=(2, 2, 2))(x)
## x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
#x = layers.Conv3D(8, 3, padding='same', activation='relu')(x)
x = layers.Conv3D(64, 3, padding='same', activation='relu')(input_img)
x = layers.Conv3D(64, 3, padding='same', activation='relu', strides=(2,2,2))(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(32, 3, padding='same', activation='relu', strides=(2,2,2))(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)
print("shape before flattening",shape_before_flattening)
x = layers.Flatten()(x)
print("shape alter flattening", K.int_shape(x))
#dimensione spazio trasformato dell'encoder o dimensione dell'encoder
#provare un numero piu' grande
x = layers.Dense(encoder_dim, activation='tanh')(x)

#latent space model parametrised by a mean and a log_variance
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
print("shape z mean e var", K.int_shape(z_mean), K.int_shape(z_log_var))


# Decoder: Latent Space -> Image
z = layers.Lambda(sampling)([z_mean, z_log_var])
print("shape z lambda", K.int_shape(z))
# This is the input where we will feed `z`.
decoder_input = layers.Input(K.int_shape(z)[1:])

print("shape decoder input", K.int_shape(decoder_input))
#print(K.int_shape(z))
# Upsample to the correct number of units
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
print("shape after upsample dense", K.int_shape(x))
# Reshape into an image of the same shape as before our last `Flatten` layer
x = layers.Reshape(shape_before_flattening[1:])(x)
print("shape after upsample reshaping", K.int_shape(x))
# We then apply then reverse operation to the initial
# stack of convolution layers: a `Conv2DTranspose` layers
# with corresponding parameters.
#x = layers.Conv3DTranspose(8, 3, padding='same', activation='relu', strides=(2, 2, 2))(x)
#x = layers.Conv3D(1, 3, padding='same', activation='sigmoid')(x)
x = layers.Conv3DTranspose(32, 3, padding='same', activation='relu', strides=(2,2,2))(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3DTranspose(32, 3, padding='same', activation='relu', strides=(2,2,2))(x)
x = layers.Conv3D(2, 3, padding='same', activation='sigmoid')(x)
# We end up with a feature map of the same size as the original input.
print("shape after upsample conv",K.int_shape(x))
# This is our decoder model.
decoder = Model(decoder_input, x)

# We then apply it to `z` to recover the decoded `z`.
z_decoded = decoder(z)

# We call our custom layer on the input and the decoded output,
# to obtain the final model output.
y = CustomVariationalLayer()([input_img, z_decoded])

# VAE Model
vae = Model(input_img, y)
vae.compile(optimizer=RMSprop(lr=0.0005), loss=None)  #note: loss = None as included in the custom loss
vae.summary()


# Read MNIST dataset, normalize it and reshape to fit the expected input (1,28,28)
#(x_train, _), (x_test, y_test) = mnist.load_data()
# x_train = np.load("out.npy")
# x_test = np.load("out_test.npy")

dataFile = "out.root"
print("reading data file:",dataFile)
inputfile = ROOT.TFile(dataFile,"READ")

nTrainingSet = 900
nTestSet = 100

x_train = np.zeros([nTrainingSet,128,128,128,2])
x_test  = np.zeros([nTestSet,128,128,128,2])

for i in range(0,nTrainingSet+nTestSet):
    hX = inputfile.Get("hX1;"+str(i+1))
    hP = inputfile.Get("hP1;"+str(i+1))
    x = root_numpy.hist2array(hX)
    p = root_numpy.hist2array(hP)
    x /= x.max()
    p /= p.max()
    if i < nTrainingSet:
        x_train[i,:,:,:,0] = x
        x_train[i,:,:,:,1] = p
    else:
        x_test[i-nTrainingSet,:,:,:,0] = x
        x_test[i-nTrainingSet,:,:,:,1] = p
inputfile.Close()

# x_train = np.random.randint(255, size=(32, 28,28,28,2))
# x_test = np.random.randint(255, size=(32, 28,28,28,2))
# x_train = x_train.astype('float32') / 255.
# #x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.astype('float32') / 255.
# #x_test = x_test.reshape(x_test.shape + (1,))

#Train the VAE
vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, None)
        )

vae.save_weights("weights", overwrite=True)
decoder.save_weights("decoder_weights", overwrite=True)



# Generate images
import matplotlib.pyplot as plt
from scipy.stats import norm

# Display a 2D manifold of the MNIST digits 
n = 6  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# Linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z,
# since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

out_size = 1

pExtr = 600.
xExtr = 60.

print("Starting to generate...")
outputFile = ROOT.TFile("vae_output.root","RECREATE")
for i, yi in enumerate(grid_x):
    # print("i:",i)
    for j, xi in enumerate(grid_y):
        # print("j:",j)        
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, out_size).reshape(out_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        for k in range(0, out_size):
            # print("k:",k)        
            hX = ROOT.TH3F("hX_"+str(k)+"_"+str(j)+"_"+str(i),
                           "hX_"+str(k)+"_"+str(j)+"_"+str(i),
                          128,-xExtr,+xExtr,
                          128,-xExtr,+xExtr,
                          128,-xExtr,+xExtr,)
            hP = ROOT.TH3F("hP_"+str(k)+"_"+str(j)+"_"+str(i),
                           "hP_"+str(k)+"_"+str(j)+"_"+str(i),
                           128,-pExtr,+pExtr,
                           128,-pExtr,+pExtr,
                           128,-pExtr,+pExtr,)
            print("x mean:",np.mean(x_decoded[k,:,:,:,0]))
            print("x std:",np.std(x_decoded[k,:,:,:,0]))
            print("x min:",np.min(x_decoded[k,:,:,:,0]))
            print("x max:",np.max(x_decoded[k,:,:,:,0]))
            print("p mean:",np.mean(x_decoded[k,:,:,:,1]))
            print("p std:",np.std(x_decoded[k,:,:,:,1]))
            print("p min:",np.min(x_decoded[k,:,:,:,1]))
            print("p max:",np.max(x_decoded[k,:,:,:,1]))
            root_numpy.array2hist(x_decoded[k,:,:,:,0],hX)
            root_numpy.array2hist(x_decoded[k,:,:,:,1],hP)
            hX.Write()
            hP.Write()
outputFile.Close()

        # digit = x_decoded[0].reshape(digit_size, digit_size)
        # figure[i * digit_size: (i + 1) * digit_size,
        #        j * digit_size: (j + 1) * digit_size] = digit
        

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.savefig("vae_sim.pdf")



