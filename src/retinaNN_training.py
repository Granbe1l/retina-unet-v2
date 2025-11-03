###################################################
#
#   Script to:
#   - Load the images and extract the patches (via Generator)
#   - Define the neural network (MODIFIED: SeparableConv + Chebyshev)
#   - define the training
#
##################################################
import keras
import numpy as np
import configparser
import random
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
from keras.utils import Sequence # Penting untuk Generator

import sys
sys.path.insert(0, './lib/')
from help_functions import *
# HAPUS: get_data_training tidak lagi digunakan
# from extract_patches import get_data_training
from pre_processing import my_PreProc

# =================================================================
# ## BAGIAN BARU: INITIALIZER KUSTOM
# =================================================================
@keras.saving.register_keras_serializable() # Tambahkan dekorator
class ChebyshevInitializer(initializers.Initializer):
    def __init__(self, kernel_size=(3, 3), at=80):
        self.kernel_size = kernel_size
        self.at = at

    def __call__(self, shape, dtype=None):
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
            return initializers.GlorotUniform()(shape, dtype=dtype)

    def get_config(self):
        return {"kernel_size": self.kernel_size, "at": self.at}

# =================================================================
# ## MODIFIKASI: Define the neural network (U-Net with SeparableConv + Chebyshev)
# =================================================================
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(patch_height,patch_width,n_ch))

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
                              depthwise_initializer=ChebyshevInitializer(kernel_size=(1,1)))(conv5)

    # PERBAIKAN: Sesuaikan Reshape dan HAPUS Permute
    conv6 = Reshape((patch_height*patch_width, 2))(conv6)
    # conv6 = Permute((1,2))(conv6) # <-- HAPUS BARIS INI

    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # Rekomendasi: Mulai dengan learning rate standar Adam 0.001
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# Define the neural network gnet (TIDAK DIUBAH)
def get_gnet(n_ch,patch_height,patch_width):
    # ... (kode gnet asli tidak diubah, tetap akan error jika dipanggil) ...
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# =================================================================
# ## BAGIAN BARU: DATA GENERATOR UNTUK MENGATASI MASALAH RAM
# =================================================================
class DataGenerator(Sequence):
    def __init__(self, imgs_path, masks_path, N_subimgs, patch_height, patch_width, batch_size, validation_split=0.0, subset='training'):
        self.imgs_path = imgs_path
        self.masks_path = masks_path
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.batch_size = batch_size
        self.subset = subset
        self.validation_split = validation_split

        print(f"Loading and preprocessing data for {self.subset} generator...")
        original_imgs = load_hdf5(self.imgs_path)
        ground_truth = load_hdf5(self.masks_path)
        
        self.full_imgs = my_PreProc(original_imgs)
        self.full_masks = ground_truth / 255.
        
        self.full_imgs = self.full_imgs[:,:,9:574,:]
        self.full_masks = self.full_masks[:,:,9:574,:]
        
        self.num_images = self.full_imgs.shape[0]
        
        # Bagi data index untuk training dan validasi
        indices = np.arange(self.num_images)
        np.random.shuffle(indices)
        split_at = int(self.num_images * (1 - self.validation_split))
        
        if self.subset == 'training':
            self.image_indices = indices[:split_at]
            self.total_patches_in_set = int(N_subimgs * (1 - self.validation_split))
        elif self.subset == 'validation':
            self.image_indices = indices[split_at:]
            self.total_patches_in_set = int(N_subimgs * self.validation_split)
        else:
            raise ValueError("Subset must be 'training' or 'validation'")
            
        print(f"Generator '{self.subset}' using {len(self.image_indices)} images.")
        
    def __len__(self):
        # Menghitung jumlah batch per epoch
        return self.total_patches_in_set // self.batch_size

    def __getitem__(self, index):
        # Fungsi ini akan dipanggil untuk menghasilkan satu batch data
        X = np.empty((self.batch_size, self.patch_height, self.patch_width, 1))
        y = np.empty((self.batch_size, self.patch_height * self.patch_width, 2))
        
        img_h = self.full_imgs.shape[2]
        img_w = self.full_imgs.shape[3]
        
        for i in range(self.batch_size):
            # Pilih gambar secara acak DARI SUBSET
            img_idx = np.random.choice(self.image_indices)
            
            x_center = random.randint(0 + self.patch_width // 2, img_w - self.patch_width // 2)
            y_center = random.randint(0 + self.patch_height // 2, img_h - self.patch_height // 2)
            
            patch_img = self.full_imgs[img_idx, 0, y_center - self.patch_height // 2:y_center + self.patch_height // 2, x_center - self.patch_width // 2:x_center + self.patch_width // 2]
            patch_mask = self.full_masks[img_idx, 0, y_center - self.patch_height // 2:y_center + self.patch_height // 2, x_center - self.patch_width // 2:x_center + self.patch_width // 2]
            
            X[i, ] = np.reshape(patch_img, (self.patch_height, self.patch_width, 1))
            y[i, ] = masks_Unet(np.reshape(patch_mask, (1, 1, self.patch_height, self.patch_width)))[0]

        return X, y
        
    def on_epoch_end(self):
        # Acak ulang image indices jika perlu
        np.random.shuffle(self.image_indices)

#========= Load settings from Config file =========
config = configparser.RawConfigParser()
config.read('configuration.txt')
path_data = config.get('data paths', 'path_local')
name_experiment = config.get('experiment name', 'name')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
N_subimgs = int(config.get('training settings', 'N_subimgs')) # Total patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
val_split = 0.1 # Definisikan split validasi di sini

#============ HAPUS Load data lama ===================
# patches_imgs_train, patches_masks_train = get_data_training(...)
# ... (semua visualisasi data lama dihapus) ...

#=========== Construct and save the model arcitecture =====
model = get_unet(1, patch_height, patch_width) # Memanggil get_unet yang sudah dimodifikasi
print ("Check: final output of the network:")
print (model.output_shape)
try:
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')
except ImportError:
    print("Pydot not installed, skipping model plot saving.")
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

# =================================================================
# ## PERBAIKAN: Gunakan Data Generator
# =================================================================
training_generator = DataGenerator(
    imgs_path=path_data + config.get('data paths', 'train_imgs_original'),
    masks_path=path_data + config.get('data paths', 'train_groundTruth'),
    N_subimgs=N_subimgs,
    patch_height=patch_height,
    patch_width=patch_width,
    batch_size=batch_size,
    validation_split=val_split,
    subset='training'
)

validation_generator = DataGenerator(
    imgs_path=path_data + config.get('data paths', 'train_imgs_original'),
    masks_path=path_data + config.get('data paths', 'train_groundTruth'),
    N_subimgs=N_subimgs,
    patch_height=patch_height,
    patch_width=patch_width,
    batch_size=batch_size,
    validation_split=val_split,
    subset='validation'
)

#============ Training ==================================
# PERBAIKAN: Monitor 'val_loss' kembali
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best.weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

# HAPUS: Tidak perlu lagi karena generator yang menangani
# patches_masks_train = masks_Unet(patches_masks_train)
# patches_imgs_train = np.transpose(patches_imgs_train, (0, 2, 3, 1))

# PERBAIKAN: Gunakan model.fit dengan generator
model.fit(
    training_generator,
    epochs=N_epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[checkpointer]
)

#========== Save and test the last model ===================
# PERBAIKAN: Ubah nama file agar konsisten dengan Keras 3
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last.weights.h5', overwrite=True)