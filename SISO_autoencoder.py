from keras.layers import Input, Dense, BatchNormalization, GaussianNoise 
from keras.models import Model  
import numpy as np 
import random as rd

eye_matrix = np.eye(1000)
x_train = np.tile(eye_matrix, (60, 1))  
x_test = np.tile(eye_matrix, (10, 1)) 
rd.shuffle(x_train)
rd.shuffle(x_test)
print(x_train.shape)  
print(x_test.shape) 

encoding_dim = 5
input_sys = Input(shape=(1000,))  

#SNR
Eb_No_dB = 0
noise = 1/(10**(Eb_No_dB/10))
noise_sigma = np.sqrt(noise)
  
#单层自编码器
"""
encoded = Dense(encoding_dim, activation='relu')(input_sys)  
decoded = Dense(1000, activation='sigmoid')(encoded)
"""

#深度自编码器
encoded = Dense(encoding_dim, activation='relu')(input_sys)  
encoded = BatchNormalization(epsilon=1e-6, weights=None)(encoded)
encoded = GaussianNoise(noise_sigma)(encoded)#噪声层
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(1000, activation='softmax')(decoded)

autoencoder = Model(inputs=input_sys, outputs=decoded)  
encoder = Model(inputs=input_sys, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
  
hist = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256,validation_data=(x_test, x_test))

encoded_sys = encoder.predict(x_test) 
decoded_sys = autoencoder.predict(x_test)
decoded_sys_round = np.round(decoded_sys)
error_num = np.sum(x_test-decoded_sys_round)