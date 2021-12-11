import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Add,LSTM,Dense
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping




data_input=np.load(file=".npy")## Input the file's path and name to load input data, the shape is (n,5,512).
data_label=np.load(file=".npy")## Input the file's path and name to load label data, the shape is (n,5,512).



n=1
x=512


input_img= Input(shape=(5,512),name='input')



def lstm(input_img,ss):
    
    x1 = LSTM(x,return_sequences=True,activation='relu',batch_input_shape=(5,x*n))(input_img)
    x1 = LSTM(x*n,return_sequences=True,activation='relu')(x1)
    x2 = LSTM(int(x*1.5),return_sequences=True,activation='relu',batch_input_shape=(5, x*n))(input_img)
    x2 = LSTM(x,return_sequences=True,activation='relu')(x2)
    x3 = LSTM(int(x*2),return_sequences=True,activation='relu',batch_input_shape=(5, x*n))(input_img)
    x3 = LSTM(x*n,return_sequences=True,activation='relu')(x3)
    x_add = Add()([x1,x2,x3])
    x4 = LSTM(int(x*ss),return_sequences=True,activation='relu',batch_input_shape=(5, x*n))( x_add)
    x4 = LSTM(x*n,return_sequences=True,activation='relu')(x4)
    return x4

x1=lstm(input_img,ss=1)
x2=lstm(input_img,ss=1.5)
x3=lstm(input_img,ss=2)

x_add= Add()([x1,x2,x3])


x_final=Dense(512,activation='tanh',name='output')(x_add)


model_LSTM = Model(input_img, x_final)
model_LSTM.compile(optimizer='adam', loss='mse')
model_LSTM.summary()

model_cb=ModelCheckpoint('./LSTM1209.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=1600,verbose=1)


cb = [model_cb, early_cb]
history = model_LSTM.fit(data_input, data_label,
          batch_size=100,
          epochs=1600,
          
          callbacks=cb,
          
          )
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./LSTM1209.csv',index=False)
#save architecture
json_string =model_LSTM.to_json()  
open('LSTM_architecture1209.json','w').write(json_string)

#save weights
model_LSTM.save_weights('weights1209.h5')