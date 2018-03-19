from keras.layers import Input, Dense, Lambda, Add
from keras.models import Model  
from keras import backend as K
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 

from numpy.random import seed
from tensorflow import set_random_seed
seed(5)
set_random_seed(3.8)

#initial
k = 2
n = 2
M = 2**k
R = k/n

eye_matrix = np.eye(M)
x_train = np.tile(eye_matrix, (2000, 1))  
x_test = np.tile(eye_matrix, (100, 1)) 
x_try = np.tile(eye_matrix, (10000, 1)) 
rd.shuffle(x_train)
rd.shuffle(x_test)
rd.shuffle(x_try)
print(x_train.shape)  
print(x_test.shape) 

#误码率
def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)  

#SNR
Eb_No_dB = 7
noise = 1/(10**(Eb_No_dB/10))
noise_sigma = np.sqrt(noise)
belta = 1/(2*R*(10**(Eb_No_dB/10)))
belta_sqrt = np.sqrt(belta)

noise_train = belta_sqrt * np.random.randn(np.shape(x_train)[0],n)
noise_test = belta_sqrt * np.random.randn(np.shape(x_test)[0],n)
noise_try = belta_sqrt * np.random.randn(np.shape(x_try)[0],n)

#autoencoder
input_sys = Input(shape=(M,))
input_noise = Input(shape=(n,))
  
#深度自编码器
encoded = Dense(M, activation='relu')(input_sys)  
encoded = Dense(n)(encoded) 
#encoded = ActivityRegularization(l2=0.02)(encoded)
encoded = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded) #energy constraint
#encoded = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded) #average power constraint
encoded_noise = Add()([encoded, input_noise]) #使用多输入改变噪声
#encoded_noise = GaussianNoise(belta_sqrt)(encoded)#噪声层
decoded = Dense(M, activation='relu')(encoded_noise)
decoded = Dense(M, activation='softmax')(decoded)

autoencoder = Model(inputs=[input_sys,input_noise], outputs=decoded)  
encoder = Model(inputs=input_sys, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy',BER])  
  
hist = autoencoder.fit([x_train,noise_train], x_train, epochs=100, batch_size=32, validation_data=([x_test, noise_test], x_test))#without batch_size

#误码个数
#encoded_sys = encoder.predict(x_try) 
#decoded_sys = autoencoder.predict([x_try,noise_try])
#decoded_sys_round = np.round(decoded_sys)
#error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))

#星座图
encoded_planisphere = encoder.predict(eye_matrix) 
plt.figure()
plt.title('Constellation')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')
plt.grid(True)

#loss曲线
plt.figure()
plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

#save
#autoencoder.save('autoencoder22.h5')



