import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
# Anda mungkin perlu impor ini untuk membuat kernel Chebyshev:
# from scipy import signal 

# ===================================================================
# === BAGIAN 1: FUNGSI UTILITAS ASLI ===
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
    
    # --- PERBAIKAN UNTUK NameError ---
    # [cite_start]Dua baris ini hilang di kode Anda sebelumnya [cite: 951]
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    # -------------------------------
    
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            # Ini juga memperbaiki error indentasi (garis merah)
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

# === 1. FUNGSI FOCAL LOSS ===
# (Sesuai saran dari paper Gabor untuk data tidak seimbang)
def focal_loss(gamma=2., alpha=.25):
    """
    Implementasi Keras untuk Focal Loss.
    Fungsi ini menangani y_true yang sudah di one-hot encode (misal: [0, 1] atau [1, 0])
    """
    def focal_loss_fixed(y_true, y_pred):
        # 1. Pastikan y_true bertipe float32
        y_true = tf.cast(y_true, tf.float32)
        
        # 2. Clip y_pred untuk menghindari nilai log(0) (error numerik)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
         # 3. Hitung pt (probabilitas prediksi untuk kelas yang BENAR)
        # Karena y_true sudah one-hot, perkalian dan penjumlahan ini
        # akan memilih probabilitas yang benar (misal: 0.9 dari [0.1, 0.9])
        pt = K.sum(y_true * y_pred, axis=-1)
        
        # 4. Hitung modulating factor (faktor modulasi)
        modulating_factor = K.pow(1. - pt, gamma)
          
        # 5. Hitung cross-entropy standar
        # K.categorical_crossentropy menghitung: -sum(y_true * log(y_pred))
        ce = K.categorical_crossentropy(y_true, y_pred)
        
        # 6. Hitung pembobotan alpha
        # y_true[..., 1] adalah kelas 'vessel' (positif)
        # y_true[..., 0] adalah kelas 'background' (negatif)
        alpha_t = y_true[..., 1] * alpha + y_true[..., 0] * (1. - alpha)
        
        # 7. Gabungkan semua: alpha * (1-pt)^gamma * ce
        loss = alpha_t * modulating_factor * ce
        
        # Kembalikan rata-rata loss
        return K.mean(loss)
    return focal_loss_fixed


# === 2. KERANGKA LAYER DOLPH-CHEBYSHEV ===
# (Ini adalah inti dari skripsi Anda)
class DolphChebyshevModulatedConv(Layer):
    """
    Ini adalah Kerangka (Boilerplate) Keras Layer untuk Dolph-Chebyshev.
    
    ==================================================================
    TANTANGAN SKRIPSI UTAMA ANDA:
    ==================================================================
    Anda harus mengganti bagian '--- MULAI LOGIKA FILTER ANDA ---'
    di dalam fungsi 'build' di bawah ini.
    
    Gantilah dengan kode NumPy/SciPy yang membangkitkan 
    kernel filter Dolph-Chebyshev 2D yang Anda teliti.
    ==================================================================
    """
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super(DolphChebyshevModulatedConv, self).__init__(**kwargs)
         self.filters = filters
        self.kernel_size = kernel_size
        # Anda bisa menambahkan argumen lain di sini, misalnya:
        # self.sidelobe_attenuation = sidelobe_attenuation

    def build(self, input_shape):
        # Dapatkan jumlah channel dari input (misal: 32, 64, 128)
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Dimensi channel input harus diketahui.")

        # --------------------------------------------------
        # --- MULAI TUGAS PENELITIAN ANDA (LOGIKA FILTER) ---
        # --------------------------------------------------
        
        # TUGAS: Buat kernel filter Dolph-Chebyshev 2D Anda di sini.
        # Ini adalah inti dari penelitian skripsi Anda.
        # Anda mungkin perlu menggunakan 'scipy.signal.chebwin'
        # atau implementasi 2D kustom.
        # INI HANYA CONTOH/PLACEHOLDER (GANTI INI!)
        # Shape kernel akhir harus: (h, w, input_channels, output_filters)
        print("="*50)
        print("PERINGATAN: Menggunakan kernel placeholder untuk DolphChebyshevModulatedConv.")
        print("Anda harus mengganti logika ini di lib/help_functions.py")
        print("="*50)
        
        # Placeholder: kernel acak (ganti dengan kode filter Anda)
        chebyshev_kernel_np = np.random.rand(
            self.kernel_size[0], 
            self.kernel_size[1], 
            input_channels, 
            self.filters
        )
        
        # --------------------------------------------------
        # --- AKHIR TUGAS PENELITIAN ANDA ---
        # --------------------------------------------------

         # Ubah kernel NumPy Anda menjadi 'weight' TensorFlow/Keras
        # Ini adalah kernel filter yang 'fixed' (tidak bisa dilatih).
        self.chebyshev_kernel = self.add_weight(
            name='chebyshev_kernel',
            shape=chebyshev_kernel_np.shape,
            initializer=tf.keras.initializers.Constant(chebyshev_kernel_np),
            trainable=False  # train=False karena ini adalah filter 'fixed',
                             # sama seperti Gabor di paper.
        )
        
        super(DolphChebyshevModulatedConv, self).build(input_shape)

    def call(self, inputs):
        # Menerapkan konvolusi 2D standar menggunakan kernel Chebyshev 'fixed' Anda
        return K.conv2d(
            inputs,
            self.chebyshev_kernel,
            padding='same' # 'same' agar dimensi H, W tidak berubah
         )

    def compute_output_shape(self, input_shape):
        # Menghitung shape output
        # Output memiliki H, W yang sama (karena padding='same')
        # tetapi dengan jumlah 'filters' yang baru (sesuai definisi layer).
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)