# Physics-guided deep learning for generating turbulent inflow conditions

## Overview

We utilise the combination of a multiscale convolutional
auto-encoder with a sub-pixel convolution layer (MSCSP-AE) and a long short-term
memory (LSTM) model. Physical constraints represented by flow gradient, Reynolds
stress tensor, and spectral content of the flow are embedded in the loss function of
the MSCSP-AE to force the model to generate realistic turbulent inflow conditions
with accurate statistics and spectra, as compared with the ground truth data. 

![Fig2](https://user-images.githubusercontent.com/60691960/145676185-ab8745d6-f87e-48ee-879e-5695b940db5f.png)


### Dependencies

Python 3.6-3.8\
tensorflow >=2.2.0 <2.4.0 (cuDNN=7.6, CUDA=10.1 for tensorflow-gpu)\
Numpy <1.19

## Data preparation
Use code normalization.py to get fluctuation-normalized training data.


#### Training
- MSCSP-AE training\
Use the fluctuation-normalized training data and MSCSP-AE.py to train the auto-encoder.

- LSTM traning
1. From trained MSCSP-AE model, you can generate the latent sapce data, and transform it to the LSTM training data form.
2. Use the LSTM training data (include input data and label data) and LSTM.py to tranin the  LSTM model.

#### Prediction
1. After finishing the training of the LSTM model, use the trained LSTM model to generate the prediction of latent space data.
2. Use the Decoder of MSCSP-AE to decode the predicted latent space data.
3. Use code denormalization.py to get the final prediction.
