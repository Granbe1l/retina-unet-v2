###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import configparser
from matplotlib import pyplot as plt
import h5py
# Import tambahan yang dibutuhkan Initializer
from scipy.signal.windows import chebwin
from keras import initializers
import tensorflow as tf
import keras # Pastikan keras diimpor

#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone_overlap, pred_only_FOV, get_data_testing_overlap, kill_border
# pre_processing.py
from pre_processing import my_PreProc

# =================================================================
# ## TAMBAHKAN KODE INI
# =================================================================
@keras.saving.register_keras_serializable() # <-- Dekorator Registrasi
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

#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================
path_data = config.get('data paths', 'path_local')
# ... (sisa kode pemuatan config sama) ...
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)
gtruth_path = path_data + config.get('data paths', 'test_groundTruth')
gtruth_masks_all = load_hdf5(gtruth_path)
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
N_visual = int(config.get('testing settings', 'N_group_visual'))


#================ Load model ==================================
best_last = config.get('testing settings', 'best_last')
# Pemanggilan ini sekarang akan berhasil karena Initializer sudah terdaftar
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
try:
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'.weights.h5')
except IOError:
    # Fallback jika nama file bobot masih format lama
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')

#======= PREDICTION LOOP =========
all_predictions = []
all_masks = []
for i in range(Imgs_to_test):
    print("Predicting image " + str(i+1) + "/" + str(Imgs_to_test))

    test_img_original_single = test_imgs_orig[i:i+1, ...]
    gtruth_single = gtruth_masks_all[i:i+1, ...]

    with h5py.File('temp_img.hdf5', 'w') as hf:
        hf.create_dataset('image', data=test_img_original_single)
    with h5py.File('temp_mask.hdf5', 'w') as hf:
        hf.create_dataset('image', data=gtruth_single)

    # Diasumsikan extract_patches.py versi ASLI
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original='temp_img.hdf5',
        DRIVE_test_groudTruth='temp_mask.hdf5',
        Imgs_to_test=1,
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )

    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1)) # Ke channels_last untuk model

    prediction = model.predict(patches_imgs_test, batch_size=32, verbose=0) # Hasilnya 3D

    pred_patches = pred_to_imgs(prediction, patch_height, patch_width, "original") # Kembali ke 4D channels_first

    # recompone_overlap asli mengharapkan channels_first
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)

    all_predictions.append(pred_img)
    all_masks.append(masks_test)

pred_imgs = np.concatenate(all_predictions, axis=0)
gtruth_masks = np.concatenate(all_masks, axis=0)

#========== Proses selanjutnya ... ====================
orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])
kill_border(pred_imgs, test_border_masks)
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

# ... sisa kode evaluasi dan penyimpanan sama persis ...
print("Orig imgs shape: " + str(orig_imgs.shape))
print("pred imgs shape: " + str(pred_imgs.shape))
print("Gtruth imgs shape: " + str(gtruth_masks.shape))
# ... (sisa kode visualisasi dan evaluasi sama persis) ...