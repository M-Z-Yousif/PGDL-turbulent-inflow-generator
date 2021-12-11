import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D,BatchNormalization, Add, Reshape,Input,Activation,Dense
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from Subpixel import Subpixel # Subpixel should be in the same path as this file
import keras.backend as k
from tensorflow.keras.applications import VGG19

sess=tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()
batch_size=40





def PSNR( y_true, y_pred):  
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * k.log(k.mean(k.square(y_pred - y_true))) / k.log(10.0) 

def build_vgg():
        vgg = VGG19(weights="imagenet",include_top=False)
        block3_conv3_copy = Conv2D(filters=256, kernel_size=(3, 3), padding='same',name='out_put_before_cactivation')
        injection_model = Sequential(vgg.layers[:10] + [block3_conv3_copy])
        block3_conv3_copy.set_weights(vgg.layers[10].get_weights())
        img = Input(shape=(128,256,3))
        # Extract image features
        img_features = injection_model(img)
        # Create model and compile
        model = Model(img, img_features)
        model.trainable = False
        return model
    

vgg = build_vgg()

# Physics-guided Loss
def physics_loss(y_true,y_pred):
    
    # MSE Loss
    loss1= tf.losses.mean_squared_error(y_true, y_pred)
    loss1=k.mean(loss1)
    
    # Gradient Loss
    grad_upred=np.zeros((128,255))
    grad_upred=tf.convert_to_tensor(grad_upred,dtype=tf.float32)
    grad_utrue=np.zeros((128,255))
    grad_utrue=tf.convert_to_tensor(grad_utrue,dtype=tf.float32)
    grad_u=np.zeros((128,255))
    grad_u=tf.convert_to_tensor(grad_u,dtype=tf.float32)
    
    grad_vpred=np.zeros((128,255))
    grad_vpred=tf.convert_to_tensor(grad_vpred,dtype=tf.float32)
    grad_vtrue=np.zeros((128,255))
    grad_vtrue=tf.convert_to_tensor(grad_vtrue,dtype=tf.float32)
    grad_v=np.zeros((128,255))
    grad_v=tf.convert_to_tensor(grad_v,dtype=tf.float32)
    
    
    grad_wpred=np.zeros((128,255))
    grad_wpred=tf.convert_to_tensor(grad_wpred,dtype=tf.float32)
    grad_wtrue=np.zeros((128,255))
    grad_wtrue=tf.convert_to_tensor(grad_wtrue,dtype=tf.float32)
    grad_w=np.zeros((128,255))
    grad_w=tf.convert_to_tensor(grad_w,dtype=tf.float32)
    
    
    grad_ppred=np.zeros((128,255))
    grad_ppred=tf.convert_to_tensor(grad_ppred,dtype=tf.float32)
    grad_ptrue=np.zeros((128,255))
    grad_ptrue=tf.convert_to_tensor(grad_ptrue,dtype=tf.float32)
    grad_p=np.zeros((128,255))
    grad_p=tf.convert_to_tensor(grad_p,dtype=tf.float32)
    
    
    ###################################################
    
    
    ggrad_upred=np.zeros((127,256))
    ggrad_upred=tf.convert_to_tensor(ggrad_upred,dtype=tf.float32)
    ggrad_utrue=np.zeros((127,256))
    ggrad_utrue=tf.convert_to_tensor(ggrad_utrue,dtype=tf.float32)
    ggrad_u=np.zeros((127,256))
    ggrad_u=tf.convert_to_tensor(ggrad_u,dtype=tf.float32)
    
    ggrad_vpred=np.zeros((127,256))
    ggrad_vpred=tf.convert_to_tensor(ggrad_vpred,dtype=tf.float32)
    ggrad_vtrue=np.zeros((127,256))
    ggrad_vtrue=tf.convert_to_tensor(ggrad_vtrue,dtype=tf.float32)
    ggrad_v=np.zeros((127,256))
    ggrad_v=tf.convert_to_tensor(ggrad_v,dtype=tf.float32)
    
    
    ggrad_wpred=np.zeros((127,256))
    ggrad_wpred=tf.convert_to_tensor(ggrad_wpred,dtype=tf.float32)
    ggrad_wtrue=np.zeros((127,256))
    ggrad_wtrue=tf.convert_to_tensor(ggrad_wtrue,dtype=tf.float32)
    ggrad_w=np.zeros((127,256))
    ggrad_w=tf.convert_to_tensor(ggrad_w,dtype=tf.float32)
    
    
    ggrad_ppred=np.zeros((127,256))
    ggrad_ppred=tf.convert_to_tensor(ggrad_ppred,dtype=tf.float32)
    ggrad_ptrue=np.zeros((127,256))
    ggrad_ptrue=tf.convert_to_tensor(ggrad_ptrue,dtype=tf.float32)
    ggrad_p=np.zeros((127,256))
    ggrad_p=tf.convert_to_tensor(ggrad_p,dtype=tf.float32)
    
    
    
    for i in range(batch_size):
        upred=y_pred[i,:,:,0]
        utrue=y_true[i,:,:,0]
        grad_upred=tf.subtract(upred[:,1:],upred[:,:-1])
        grad_utrue=tf.subtract(utrue[:,1:],utrue[:,:-1])
        grad_u+=tf.math.squared_difference(grad_upred, grad_utrue)
        
        ggrad_upred=tf.subtract(upred[1:,:],upred[:-1,:])
        ggrad_utrue=tf.subtract(utrue[1:,:],utrue[:-1,:])
        ggrad_u+=tf.math.squared_difference(ggrad_upred, ggrad_utrue)
        
        
        
        
        vpred=y_pred[i,:,:,1]
        vtrue=y_true[i,:,:,1]
        grad_vpred=tf.subtract(vpred[:,1:],vpred[:,:-1])
        grad_vtrue=tf.subtract(vtrue[:,1:],vtrue[:,:-1])
        grad_v+=tf.math.squared_difference(grad_vpred, grad_vtrue)
        
        ggrad_vpred=tf.subtract(vpred[1:,:],vpred[:-1,:])
        ggrad_vtrue=tf.subtract(vtrue[1:,:],vtrue[:-1,:])
        ggrad_v+=tf.math.squared_difference(ggrad_vpred, ggrad_vtrue)
        
        wpred=y_pred[i,:,:,2]
        wtrue=y_true[i,:,:,2]
        grad_wpred=tf.subtract(wpred[:,1:],wpred[:,:-1])
        grad_wtrue=tf.subtract(wtrue[:,1:],wtrue[:,:-1])
        grad_w+=tf.math.squared_difference(grad_wpred, grad_wtrue) 
        
        ggrad_wpred=tf.subtract(wpred[1:,:],wpred[:-1,:])
        ggrad_wtrue=tf.subtract(wtrue[1:,:],wtrue[:-1,:])
        ggrad_w+=tf.math.squared_difference(ggrad_wpred, ggrad_wtrue)   
        
        ppred=y_pred[i,:,:,3]
        ptrue=y_true[i,:,:,3]
        grad_ppred=tf.subtract(ppred[:,1:],ppred[:,:-1])
        grad_ptrue=tf.subtract(ptrue[:,1:],ptrue[:,:-1])
        grad_p+=tf.math.squared_difference(grad_ppred, grad_ptrue) 
    
        ggrad_ppred=tf.subtract(ppred[1:,:],ppred[:-1,:])
        ggrad_ptrue=tf.subtract(ptrue[1:,:],ptrue[:-1,:])
        ggrad_p+=tf.math.squared_difference(ggrad_ppred, ggrad_ptrue)
        
  
        
        
    grad_u=tf.reduce_sum(grad_u/batch_size,0)
    grad_u=grad_u/(dx*128)
    grad_u=k.mean(grad_u)
    
    grad_v=tf.reduce_sum(grad_v/batch_size,0)
    grad_v=grad_v/(dx*128)
    grad_v=k.mean(grad_v)
    
    grad_w=tf.reduce_sum(grad_w/batch_size,0)
    grad_w=grad_w/(dx*128)
    grad_w=k.mean(grad_w)
    
    grad_p=tf.reduce_sum(grad_p/batch_size,0)
    grad_p=grad_p/(dx*128)
    grad_p=k.mean(grad_p)
    
    ####################
    ggrad_u=tf.reduce_sum(ggrad_u/batch_size,1)
    ggrad_u=ggrad_u/(dy*255)
    ggrad_u=k.mean(ggrad_u)
    
    ggrad_v=tf.reduce_sum(ggrad_v/batch_size,1)
    ggrad_v=ggrad_v/(dy*255)
    ggrad_v=k.mean(ggrad_v)
    
    ggrad_w=tf.reduce_sum(ggrad_w/batch_size,1)
    ggrad_w=ggrad_w/(dy*255)
    ggrad_w=k.mean(ggrad_w)
    
    ggrad_p=tf.reduce_sum(ggrad_p/batch_size,1)
    ggrad_p=ggrad_p/(dy*255)
    ggrad_p=k.mean(ggrad_p)
    
    ################################################
    
    loss2=(grad_u+grad_v+grad_w+grad_p+ggrad_u+ggrad_v+ggrad_w+ggrad_p)/8
    
    # Reynolds stress Loss
    y_true=y_true
    y_pred=y_pred
    
    uu=tf.square(((y_true[:,:,:,0]*y_true[:,:,:,0])-(y_pred[:,:,:,0]*y_pred[:,:,:,0])))
    vv=tf.square(((y_true[:,:,:,1]*y_true[:,:,:,1])-(y_pred[:,:,:,1]*y_pred[:,:,:,1])))
    ww=tf.square(((y_true[:,:,:,2]*y_true[:,:,:,2])-(y_pred[:,:,:,2]*y_pred[:,:,:,2])))
    uv=tf.square(((y_true[:,:,:,0]*y_true[:,:,:,1])-(y_pred[:,:,:,0]*y_pred[:,:,:,1])))
    uw=tf.square(((y_true[:,:,:,0]*y_true[:,:,:,2])-(y_pred[:,:,:,0]*y_pred[:,:,:,2])))
    vw=tf.square(((y_true[:,:,:,1]*y_true[:,:,:,2])-(y_pred[:,:,:,1]*y_pred[:,:,:,2])))
    
    loss3=uu+vv+ww+uv+uw+vw     
    loss3=k.mean(loss3)
    
    # Perceptual Loss
    y_trueu,y_truev,y_truew,y_truep= tf.split(y_true, axis = -1, num_or_size_splits = 4)
    y_predu,y_predv,y_predw,y_predp= tf.split(y_pred, axis = -1, num_or_size_splits = 4)
    
    y_truevgg=tf.concat([y_trueu,y_truev,y_truew], axis = -1)
    y_predvgg=tf.concat([y_predu,y_predv,y_predw], axis = -1)
    
    
    initial_feature1=vgg(y_truevgg)
    cnnae_feature1=vgg(y_predvgg)
    percept_loss_o1 = tf.losses.mean_squared_error(cnnae_feature1, initial_feature1)
    percept_loss1 = k.mean(percept_loss_o1)
    percept_loss = percept_loss1 
    
    # Spectrum Loss
    u1=y_true[:,:,:,0]
    u2=y_pred[:,:,:,0]
    
    v1=y_true[:,:,:,1]
    v2=y_pred[:,:,:,1]
    
    w1=y_true[:,:,:,2]
    w2=y_pred[:,:,:,2]
    
    p1=y_true[:,:,:,3]
    p2=y_pred[:,:,:,3]
    
    
    
    u1=tf.dtypes.cast(u1,tf.complex64)
    u2=tf.dtypes.cast(u2,tf.complex64)
    v1=tf.dtypes.cast(v1,tf.complex64)
    v2=tf.dtypes.cast(v2,tf.complex64)
    w1=tf.dtypes.cast(w1,tf.complex64)
    w2=tf.dtypes.cast(w2,tf.complex64)
    p1=tf.dtypes.cast(p1,tf.complex64)
    p2=tf.dtypes.cast(p2,tf.complex64)
    
    
    uu1=np.zeros((128,256))
    
    uu1=tf.convert_to_tensor(uu1,dtype=tf.double)
    
    Eu1=np.zeros((256,))
    Eu1=tf.convert_to_tensor(Eu1,dtype=tf.double)
    

    uu2=np.zeros((128,256))
    uu2=tf.convert_to_tensor(uu2,dtype=tf.double)
    Eu2=np.zeros((256,))
    Eu2=tf.convert_to_tensor(Eu2,dtype=tf.double)
    
    
    vv1=np.zeros((128,256))
    vv1=tf.convert_to_tensor(vv1,dtype=tf.double)
    Ev1=np.zeros((256,))
    Ev1=tf.convert_to_tensor(Ev1,dtype=tf.double)
    
    
    vv2=np.zeros((128,256))
    vv2=tf.convert_to_tensor(vv2,dtype=tf.double)
    Ev2=np.zeros((256,))
    Ev2=tf.convert_to_tensor(Ev2,dtype=tf.double)
    
    
    
    ww1=np.zeros((128,256))
    ww1=tf.convert_to_tensor(ww1,dtype=tf.double)
    Ew1=np.zeros((256,))
    Ew1=tf.convert_to_tensor(Ew1,dtype=tf.double)
    
    
    ww2=np.zeros((128,256))
    ww2=tf.convert_to_tensor(ww2,dtype=tf.double)
    Ew2=np.zeros((256,))
    Ew2=tf.convert_to_tensor(Ew2,dtype=tf.double)
    
    pp1=np.zeros((128,256))
    pp1=tf.convert_to_tensor(pp1,dtype=tf.double)
    
    Ep1=np.zeros((256,))
    Ep1=tf.convert_to_tensor(Ep1,dtype=tf.double)
    
    pp2=np.zeros((128,256))
    pp2=tf.convert_to_tensor(pp2,dtype=tf.double)
    
    Ep2=np.zeros((256,))
    Ep2=tf.convert_to_tensor(Ep2,dtype=tf.double)
    
    
    
  
    for i in range (batch_size):
        
        upp1=u1[i,:,:]
        upp2=u2[i,:,:] 
        
        vpp1=v1[i,:,:]
        vpp2=v2[i,:,:]
        
        wpp1=w1[i,:,:] 
        wpp2=w2[i,:,:] 
        
        ppp1=p1[i,:,:]
        ppp2=p2[i,:,:]
        
        
        
        upp1=upp1[8,:]
        upp2=upp2[8,:]
        
        vpp1=vpp1[8,:]
        vpp2=vpp2[8,:]
        
        wpp1=wpp1[8,:]
        wpp2=wpp2[8,:]
        
        ppp1=ppp1[8,:]
        ppp2=ppp2[8,:]
     
        
                 
        wku1=tf.signal.fft(upp1)
        wku2=tf.signal.fft(upp2)
        
        wkv1=tf.signal.fft(vpp1)
        wkv2=tf.signal.fft(vpp2)
        
        wkw1=tf.signal.fft(wpp1)
        wkw2=tf.signal.fft(wpp2)
        
        wkp1=tf.signal.fft(ppp1)
        wkp2=tf.signal.fft(ppp2)
        
    
        
        
        abs_fourier_transformu1 = tf.abs(wku1)
        abs_fourier_transformu2 = tf.abs(wku2)
        
        abs_fourier_transformv1 = tf.abs(wkv1)
        abs_fourier_transformv2 = tf.abs(wkv2)
        
        abs_fourier_transformw1 = tf.abs(wkw1)
        abs_fourier_transformw2 = tf.abs(wkw2)
        
        abs_fourier_transformp1 = tf.abs(wkp1)
        abs_fourier_transformp2 = tf.abs(wkp2)
        
        power_spectrumu1 = tf.square(abs_fourier_transformu1)
        power_spectrumu2 = tf.square(abs_fourier_transformu2)
    
        power_spectrumv1 = tf.square(abs_fourier_transformv1)
        power_spectrumv2 = tf.square(abs_fourier_transformv2)
        
        power_spectrumw1 = tf.square(abs_fourier_transformw1)
        power_spectrumw2 = tf.square(abs_fourier_transformw2)
        
        power_spectrump1 = tf.square(abs_fourier_transformp1)
        power_spectrump2 = tf.square(abs_fourier_transformp2)
       
        power_spectrumu1=tf.dtypes.cast(power_spectrumu1,tf.float64)
        power_spectrumu2=tf.dtypes.cast(power_spectrumu2,tf.float64)
        
        power_spectrumv1=tf.dtypes.cast(power_spectrumv1,tf.float64)
        power_spectrumv2=tf.dtypes.cast(power_spectrumv2,tf.float64)
        
        power_spectrumw1=tf.dtypes.cast(power_spectrumw1,tf.float64)
        power_spectrumw2=tf.dtypes.cast(power_spectrumw2,tf.float64)
        
        power_spectrump1=tf.dtypes.cast(power_spectrump1,tf.float64)
        power_spectrump2=tf.dtypes.cast(power_spectrump2,tf.float64)
        
        
        Eu1+=power_spectrumu1
        Eu2+=power_spectrumu2
    
        Ev1+=power_spectrumv1
        Ev2+=power_spectrumv2
    
        Ew1+=power_spectrumw1
        Ew2+=power_spectrumw2
        
        Ep1+=power_spectrump1
        Ep2+=power_spectrump2
    
    
    print(Eu1)
    Eumean1=(Eu1)/(0.00035*0.063*256**2*B_S)
    # Eumean1=tf.reshape(Eumean1,256)
    
    Eumean2=(Eu2)/(0.00035*0.063*256**2*B_S)
    # Eumean2=tf.reshape(Eumean2,256)
    
    Evmean1=(Ev1)/(0.00035*0.063*256**2*B_S)
    # Evmean1=tf.reshape(Evmean1,256)
    Evmean2=(Ev2)/(0.00035*0.063*256**2*B_S)
    # Evmean2=tf.reshape(Evmean2,256)
    
    Ewmean1=(Ew1)/(0.00035*0.063*256**2*B_S)
    # Ewmean1=tf.reshape(Ewmean1,256)
    Ewmean2=(Ew2)/(0.00035*0.063*256**2*B_S)
    # Ewmean2=tf.reshape(Ewmean2,256)
    
    Epmean1=(Ep1)/(256**2*B_S)
    # Epmean1=tf.reshape(Epmean1,256)
    Epmean2=(Ep2)/(256**2*B_S)
    # Epmean2=tf.reshape(Epmean2,256)
    
    
    E_TRUE=Eumean1+Evmean1+Ewmean1+Epmean1
    E_PRED=Eumean2+Evmean2+Ewmean2+Epmean2
 
    loss4=k.mean(tf.abs(E_TRUE-E_PRED ))
    loss4=tf.dtypes.cast(loss4,tf.float32)

    loss = 200*loss1 + 1*loss2 + 10*loss3  + 0.00005*loss4 + 0.002*percept_loss # Total Loss with weight
     
    return loss
######################################################

 

DATA=np.load('.npy') ## Input the file's path and name to load training data.

xyz=np.load('./x_y_z.npy') ## Input the file's path and name to load x_y_z coordinate data.

x=np.reshape(xyz[:,2],(128,256))
y=np.reshape(xyz[:,1],(128,256))
x=tf.convert_to_tensor(x,dtype=tf.float32)
y=tf.convert_to_tensor(y,dtype=tf.float32)
 
x=x[0,:]
y=y[:,0]
dx=tf.subtract(x[1:],x[:-1])
dy=tf.subtract(y[1:],y[:-1])

y_mean=np.mean(DATA,axis=0)

B_S=40 # Batch_size

pat=50 # Patience
name='inflow_fluc_phy' 



# Building the model
#Encoder
input_data=Input(shape=(128,256,4))

filter_num=24

xx1 = Conv2D(filter_num, (3,3),activation='relu', padding='same')(input_data)
x1 = Conv2D(filter_num, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num, (3,3),activation='relu', padding='same')(x1)

x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)



xx1 = Conv2D(filter_num*2, (3,3),activation='relu', padding='same')(x1)
x1 = Conv2D(filter_num*2, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*2, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)


xx1 = Conv2D(filter_num*4, (3,3),activation='relu', padding='same')(x1)
x1 = Conv2D(filter_num*4, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*4, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)


xx1 = Conv2D(filter_num*8, (3,3),activation='relu', padding='same')(x1)
x1 = Conv2D(filter_num*8, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*8, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)

x1 = Conv2D(filter_num*12, (3,3),activation='relu', padding='same')(x1)



xx2 = Conv2D(filter_num, (5,5),activation='relu', padding='same')(input_data)
x2 = Conv2D(filter_num, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num, (5,5),activation='relu', padding='same')(x2)
x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling2D((2,2),padding='same')(x2)

xx2= Conv2D(filter_num*2, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*2, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*2, (5,5),activation='relu', padding='same')(x2)
x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling2D((2,2),padding='same')(x2)

xx2 = Conv2D(filter_num*4, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*4, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*4, (5,5),activation='relu', padding='same')(x2)
x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling2D((2,2),padding='same')(x2)

xx2 = Conv2D(filter_num*8, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*8, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*8, (5,5),activation='relu', padding='same')(x2)
x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling2D((2,2),padding='same')(x2)

x2 = Conv2D(filter_num*12, (5,5),activation='relu', padding='same')(x2)



xx3 = Conv2D(filter_num, (7,7),activation='relu', padding='same')(input_data)
x3 = Conv2D(filter_num, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling2D((2,2),padding='same')(x3)
 
xx3= Conv2D(filter_num*2, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*2, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*2, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling2D((2,2),padding='same')(x3)

xx3 = Conv2D(filter_num*4, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*4, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*4, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling2D((2,2),padding='same')(x3)

xx3 = Conv2D(filter_num*8, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*8, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*8, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling2D((2,2),padding='same')(x3)

x3 = Conv2D(filter_num*12, (7,7),activation='relu', padding='same')(x3)


#latent vector
x_add = Add()([x1,x2,x3])
x_d=Dense(4,activation='tanh',name='latent')(x_add)


x_lnt = Reshape((512,),name='latent_reshape1')(x_d) # For LSTM training

x = Reshape((8,16,4),name='latent_reshape2')(x_lnt)
#Decoder
upscale_factor = 2

xx1 = Conv2D(filter_num*12, (3,3),activation='relu', padding='same')(x)
x1 = Conv2D(filter_num*12, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*12, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = UpSampling2D((2,2))(x1)

xx1 = Conv2D(filter_num*6, (3,3),activation='relu', padding='same')(x1) 
x1 = Conv2D(filter_num*6, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*6, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)

x1 = UpSampling2D((2,2))(x1)

xx1 = Conv2D(filter_num*3, (3,3),activation='relu', padding='same')(x1)
x1 = Conv2D(filter_num*3, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(filter_num*3, (3,3),activation='relu', padding='same')(x1)
x1= tf.concat([x1,xx1],axis=-1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = UpSampling2D((2,2))(x1)

xx1 = Conv2D(filter_num*1.5, (3,3),activation='relu', padding='same')(x1)
x1 = Conv2D(filter_num*1.5, (3,3),activation='relu', padding='same')(xx1)
x1 = BatchNormalization()(x1)

x1d = Conv2D(16, (3,3),activation='relu', padding='same')(x1)





xx2 = Conv2D(filter_num*12, (5,5),activation='relu', padding='same')(x)
x2 = Conv2D(filter_num*12, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*12, (5,5),activation='relu', padding='same')(x2)
x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = UpSampling2D((2,2))(x2)

xx2 = Conv2D(filter_num*6, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*6, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*6, (5,5),activation='relu', padding='same')(x2)

x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = UpSampling2D((2,2))(x2)

xx2 = Conv2D(filter_num*3, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*3, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(filter_num*3, (5,5),activation='relu', padding='same')(x2)

x2= tf.concat([x2,xx2],axis=-1)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = UpSampling2D((2,2))(x2)

xx2 = Conv2D(filter_num*1.5, (5,5),activation='relu', padding='same')(x2)
x2 = Conv2D(filter_num*1.5, (5,5),activation='relu', padding='same')(xx2)
x2= BatchNormalization()(x2)
x2 = Activation('relu')(x2)

x2d = Conv2D(16, (5,5),activation='relu', padding='same')(x2)





xx3 = Conv2D(filter_num*12, (7,7),activation='relu', padding='same')(x)
x3 = Conv2D(filter_num*12, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*12, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = UpSampling2D((2,2))(x3)

xx3 = Conv2D(filter_num*6, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*6, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*6, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = UpSampling2D((2,2))(x3)


xx3 = Conv2D(filter_num*3, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*3, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(filter_num*3, (7,7),activation='relu', padding='same')(x3)
x3= tf.concat([x3,xx3],axis=-1)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = UpSampling2D((2,2))(x3)

xx3 = Conv2D(filter_num*1.5, (7,7),activation='relu', padding='same')(x3)
x3 = Conv2D(filter_num*1.5, (7,7),activation='relu', padding='same')(xx3)
x3= BatchNormalization()(x3)
x3 = Activation('relu')(x3)


x3d = Conv2D(16, (7,7),activation='relu', padding='same')(x3)
x_add = tf.concat([x1d,x2d,x3d],axis=-1)
x_f_1 = Conv2D(16, (3,3),activation='relu', padding='same')(x_add)
x_f_2 = Conv2D(4, (3,3),activation='relu', padding='same')(x_f_1)
x_SBP = Subpixel(4, (3,3), r=upscale_factor, padding='same')(x_f_2)
x_final=Activation('sigmoid')(x_SBP )

autoencoder = Model(input_data, x_final)

autoencoder.compile(optimizer='adam',metrics=['accuracy',PSNR] ,loss=physics_loss)
# Model Summary
autoencoder.summary()

#save weights
tempfn='./'+name+'.hdf5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]
#save architecture
json_string =autoencoder.to_json()  
open(name+'_architecture.json','w').write(json_string)
X_train,X_test,y_train,y_test=train_test_split(DATA,DATA,test_size=0.2,random_state=1)



history=autoencoder.fit(X_train, y_train,
                epochs=400,
                batch_size=B_S,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=cb )

df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
tempfn='./'+name+'.csv'
df_results.to_csv(path_or_buf=tempfn,index=False)


 
