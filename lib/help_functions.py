import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
# --- MODIFIKASI SKRIPSI ---
# Impor 'scipy.signal' untuk membuat window Chebyshev
import scipy.signal

# ===================================================================
# === BAGIAN 1: FUNGSI UTILITAS ASLI (TELAH DIPERBAIKI) ===
# ===================================================================

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
         totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                 if pred[i,pix,1]>=0.5:
                     pred_images[i,pix]=1
                 else:
                    pred_images[i,pix]=0
    else:
        print ("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

# ===================================================================
# === BAGIAN 2: KODE BARU TAMBAHAN UNTUK SKRIPSI ANDA ===
# ===================================================================

# === 1. FUNGSI FOCAL LOSS (FINAL) ===
def focal_loss(gamma=2., alpha=.25):
    """
    Implementasi Keras untuk Focal Loss.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating_factor = tf.pow(1. - pt, gamma)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        alpha_t = y_true[..., 1] * alpha + y_true[..., 0] * (1. - alpha)
        loss = alpha_t * modulating_factor * ce
        
        return tf.reduce_mean(loss)
    return focal_loss_fixed


# === 2. LAYER DOLPH-CHEBYSHEV (IMPLEMENTASI LOGIKA FINAL) ===
class DolphChebyshevModulatedConv(Layer):
    """
    Ini adalah implementasi logis dari layer Dolph-Chebyshev
    untuk skripsi Anda.
    
    Ini meniru metodologi Gabor dengan:
    1. Membuat filter Dolph-Chebyshev 2D yang 'fixed' (tidak dilatih).
    2. Membuat 'kernel' konvolusi yang 'learnable' (bisa dilatih).
    3. Mengalikan keduanya (modulasi) untuk membuat kernel final.
    """
    def __init__(self, filters, kernel_size=(3, 3), sidelobe_attenuation=80, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.sidelobe_attenuation = sidelobe_attenuation # 'at' untuk chebwin

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Dimensi channel input harus diketahui.")

        # --- INI ADALAH LOGIKA FILTER SKRIPSI ANDA ---
        # 1. Buat window Chebyshev 1D untuk tinggi dan lebar
        # 'at' (attenuation) adalah parameter kunci untuk penelitian Anda
        cheb_win_h = scipy.signal.chebwin(self.kernel_size[0], at=self.sidelobe_attenuation)
        cheb_win_w = scipy.signal.chebwin(self.kernel_size[1], at=self.sidelobe_attenuation)
        
        # 2. Buat kernel 2D dari window 1D (menggunakan outer product)
        kernel_2d = np.outer(cheb_win_h, cheb_win_w)
        
        # 3. Normalisasi kernel (opsional, tapi praktik yang baik)
        kernel_2d /= np.sum(kernel_2d)
        
        # 4. Bentuk ulang agar bisa di-broadcast (dikalikan)
        # Shape menjadi: (h, w, 1, 1)
        cheb_filter_shape = self.kernel_size + (1, 1)
        self.chebyshev_filter = tf.constant(
            np.reshape(kernel_2d, cheb_filter_shape), 
            dtype=tf.float32
        )
        
        # --- KERNEL YANG BISA DILATIH ---
        # 5. Buat kernel 'learnable' (yang akan dimodulasi)
        # Shape: (h, w, input_channels, output_filters)
        kernel_shape = self.kernel_size + (input_channels, self.filters)
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform', # Inisialisasi standar Keras
            trainable=True  # INI YANG PALING PENTING
        )
        # --- SELESAI ---
        
        super(DolphChebyshevModulatedConv, self).build(input_shape) 

    def call(self, inputs):
        # 6. Modulasi: kalikan kernel 'learnable' dengan filter 'fixed'
        # [h,w,C_in,C_out] * [h,w,1,1] -> [h,w,C_in,C_out]
        modulated_kernel = self.kernel * self.chebyshev_filter

        # 7. Lakukan konvolusi dengan kernel yang sudah dimodulasi
        return tf.nn.conv2d(
            inputs,
            modulated_kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)