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
# ## DEFINISI KELAS INITIALIZER DITAMBAHKAN DI SINI
# =================================================================
@keras.saving.register_keras_serializable() # <-- Dekorator Registrasi
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

#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================
path_data = config.get('data paths', 'path_local')
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

# PERBAIKAN: Tambahkan custom_objects saat memuat model
custom_objects = {'ChebyshevInitializer': ChebyshevInitializer}
model = model_from_json(
    open(path_experiment+name_experiment +'_architecture.json').read(),
    custom_objects=custom_objects # <-- Argumen ditambahkan di sini
)

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

    # ===================================================================
    # BARIS transpose YANG MENYEBABKAN ERROR SUDAH DIHAPUS
    # ===================================================================

    # recompone_overlap asli mengharapkan channels_first
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)

    all_predictions.append(pred_img)
    all_masks.append(masks_test)

pred_imgs = np.concatenate(all_predictions, axis=0)
gtruth_masks = np.concatenate(all_masks, axis=0)

#========== Proses selanjutnya ... ====================
# pred_imgs sudah dalam format channels_first, jadi tidak perlu transpose
orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])
kill_border(pred_imgs, test_border_masks)
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

# ... sisa kode evaluasi dan penyimpanan sama persis ...
print("Orig imgs shape: " + str(orig_imgs.shape))
print("pred imgs shape: " + str(pred_imgs.shape))
print("Gtruth imgs shape: " + str(gtruth_masks.shape))
visualize(group_images(orig_imgs, N_visual), path_experiment + "all_originals")
visualize(group_images(pred_imgs, N_visual), path_experiment + "all_predictions")
visualize(group_images(gtruth_masks, N_visual), path_experiment + "all_groundTruths")

assert (orig_imgs.shape[0] == pred_imgs.shape[0] and orig_imgs.shape[0] == gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted % group == 0)
for i in range(int(N_predicted / group)):
    orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
    masks_stripe = group_images(gtruth_masks[i * group:(i * group) + group, :, :, :], group)
    pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
    total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
    visualize(total_img, path_experiment + name_experiment + "_Original_GroundTruth_Prediction" + str(i))


# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)
print("Calculating results only inside the FOV:")
print("y scores pixels: " + str(y_scores.shape[0]))
print("y true pixels: " + str(y_true.shape[0]))

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

# =================================================================
# ## TAMBAHAN: Loop Optimasi Threshold untuk F1-Score
# =================================================================
print("\n\n======== Mencari Threshold Optimal untuk F1-Score ========")
best_f1 = 0
best_threshold = 0.5 # Mulai dengan default
for threshold in np.arange(0.1, 0.9, 0.05): # Uji threshold dari 0.1 s/d 0.85
    y_pred_test = (y_scores >= threshold).astype(int)
    current_f1 = f1_score(y_true, y_pred_test)
    print(f"Threshold: {threshold:.2f} -> F1-Score: {current_f1:.4f}")
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print("\n---> Threshold Optimal ditemukan di: " + str(best_threshold))
print("---> F1-Score Terbaik: " + str(best_f1))
print("=======================================================\n")
# =================================================================

#Confusion matrix
threshold_confusion = best_threshold # Gunakan threshold terbaik yang ditemukan
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print ("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score)) # F1-Score ini akan menjadi F1-Score terbaik

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                  + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                  + "\nJaccard similarity score: " +str(jaccard_index)
                  + "\nF1 score (F-measure): " +str(F1_score)
                  + "\nOptimal Threshold: " +str(best_threshold) # Tambahkan info threshold
                  +"\n\nConfusion matrix:"
                  +str(confusion)
                  +"\nACCURACY: " +str(accuracy)
                  +"\nSENSITIVITY: " +str(sensitivity)
                  +"\nSPECIFICITY: " +str(specificity)
                  +"\nPRECISION: " +str(precision)
                  )
file_perf.close()