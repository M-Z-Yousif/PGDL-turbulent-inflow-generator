Python 3.6-3.8
tensorflow >=2.2.0 <2.4.0 (cuDNN=7.6, CUDA=10.1 for tensorflow-gpu)
Numpy <1.19

1. Data preparation
Use code normalization.np to get fluctuation-normalized training data.

2. MSCSP-AE training
Use the fluctuation-normalized training data and code MSCSP-AE.py to tranin the  MSCSP-AE model.

3. LSTM traning
3.1 From trained MSCSP-AE model, we can generate the latent sapce data, and transform it to the LSTM training data form.
3.2 Use the LSTM training data(include input data and label data) and code LSTM.py to tranin the  LSTM model.

4. Prediction
4.1 After finishing the training of LSTM, we use the trained LSTM to generate the prediction of latent space data.
4.2 Use the Decoder of MSCSP-AE to decode the predicted latent space data.
4.3 Use code denormalization.py to get the final prediction.