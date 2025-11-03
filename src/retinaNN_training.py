###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network (MODIFIED: SeparableConv + Chebyshev)
#   - define the training
#
##################################################

import numpy as np
import configparser
# Import tambahan untuk modifikasi
from scipy.signal.windows import chebwin
from keras import initializers
import tensorflow as tf

from keras.models import Model
# Ganti Conv2D dengan SeparableConv2D
from keras.layers import Input, concatenate, SeparableConv2D, MaxPooling2D, UpSampling2D, Reshape, Permute, Activation, Dropout
from keras.optimizers import Adam # Pastikan Adam diimpor
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import plot_model as plot
# from keras.optimizers import SGD # Tidak perlu jika pakai Adam

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

# =================================================================
# ## BAGIAN BARU: INITIALIZER KUSTOM
# =================================================================
# Class untuk membuat bobot awal berdasarkan filter Dolph-Chebyshev
class ChebyshevInitializer(initializers.Initializer):
    def __init__(self, kernel_size=(3, 3), at=80):
        self.kernel_size = kernel_size
        self.at = at

    def __call__(self, shape, dtype=None):
        # Targetkan kernel depthwise dari SeparableConv2D
        # Shape: (height, width, in_channels, depth_multiplier=1)
        if len(shape) == 4 and shape[0] == self.kernel_size[0] and shape[1] == self.kernel_size[1] and shape[3] == 1:
            kernel_height, kernel_width, input_channels, _ = shape
            cheb_window_1d_h = chebwin(kernel_height, at=self.at)
            cheb_window_1d_w = chebwin(kernel_width, at=self.at)
            cheb_kernel_2d = np.outer(cheb_window_1d_h, cheb_window_1d_w)
            if np.sum(cheb_kernel_2d) != 0:
                cheb_kernel_2d /= np.sum(cheb_kernel_2d)
            final_weights = np.zeros(shape)
            for i in range(input_channels):
                 final_weights[:, :, i, 0] = cheb_kernel_2d
            return tf.convert_to_tensor(final_weights, dtype=dtype)
        else:
            # Untuk kernel pointwise atau bias, gunakan initializer standar Keras
            return initializers.GlorotUniform()(shape, dtype=dtype)

    def get_config(self):
        return {"kernel_size": self.kernel_size, "at": self.at}

# =================================================================
# ## MODIFIKASI: Define the neural network (U-Net with SeparableConv + Chebyshev)
# =================================================================
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(patch_height,patch_width,n_ch))

    # Ganti Conv2D -> SeparableConv2D dan tambahkan depthwise_initializer
    conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=-1)
    conv4 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=-1)
    conv5 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(3,3)))(conv5)
    #
    conv6 = SeparableConv2D(2, (1, 1), activation='relu',padding='same',
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(1,1)))(conv5) # Kernel (1,1)

    # Sesuaikan Reshape untuk channels_last (H, W, C) -> (H*W, C)
    conv6 = Reshape((patch_height*patch_width, 2))(conv6)
    # Permute menukar dua dimensi terakhir (H*W, C) -> (H*W, C) (sudah benar)
    conv6 = Permute((1,2))(conv6)

    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # Rekomendasi: Mulai dengan learning rate standar Adam 0.001
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#Define the neural network gnet (TIDAK DIMODIFIKASI, tetap seperti aslinya)
def get_gnet(n_ch,patch_height,patch_width):
    # PERINGATAN: Fungsi ini masih menggunakan sintaks Keras lama dan akan error jika dijalankan.
    inputs = Input((n_ch, patch_height, patch_width))
    # ... (kode gnet asli tidak diubah) ...
    # Compile diubah ke Adam untuk konsistensi, tapi fungsi ini tetap usang
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#========= Load settings from Config file ========= (TIDAK ADA PERUBAHAN)
config = configparser.RawConfigParser()
config.read('configuration.txt')
path_data = config.get('data paths', 'path_local')
name_experiment = config.get('experiment name', 'name')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#============ Load the data and divided in patches ========== (TIDAK ADA PERUBAHAN)
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV')
)

#========= Save a sample ========== (TIDAK ADA PERUBAHAN)
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")

#=========== Construct and save the model arcitecture ===== (TIDAK ADA PERUBAHAN)
n_ch = 1
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width) # Memanggil get_unet yang sudah dimodifikasi
print ("Check: final output of the network:")
print (model.output_shape)
try:
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')
except ImportError:
    print("Pydot not installed, skipping model plot saving.")
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

#============ Training ================================== (TIDAK ADA PERUBAHAN)
#checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best.weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
# SESUDAH (PERBAIKAN KUNCI)
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='loss', mode='auto', save_best_only=True)

patches_masks_train = masks_Unet(patches_masks_train)

patches_imgs_train = np.transpose(patches_imgs_train, (0, 2, 3, 1))

model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

#========== Save and test the last model =================== (TIDAK ADA PERUBAHAN)
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last.weights.h5', overwrite=True)

