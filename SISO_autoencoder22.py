from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.models import Model  
from keras import backend as K
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 

#initial
k = 2
n = 2
M = 2**k
R = k/n

eye_matrix = np.eye(M)
x_train = np.tile(eye_matrix, (600, 1))  
x_test = np.tile(eye_matrix, (100, 1)) 
x_try = np.tile(eye_matrix, (1000, 1)) 
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

#autoencoder
input_sys = Input(shape=(M,))
  
#深度自编码器
encoded = Dense(M, activation='relu')(input_sys)  
encoded = Dense(n)(encoded) 
#encoded = ActivityRegularization(l2=0.02)(encoded)
encoded = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded) #energy constraint
#encoded = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded) #average power constraint
encoded_noise = GaussianNoise(belta_sqrt)(encoded)#噪声层
decoded = Dense(M, activation='relu')(encoded_noise)
decoded = Dense(M, activation='softmax')(decoded)

autoencoder = Model(inputs=input_sys, outputs=decoded)  
encoder = Model(inputs=input_sys, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy',BER])  
  
hist = autoencoder.fit(x_train, x_train, epochs=200, validation_data=(x_test, x_test))#without batch_size

#误码个数
#encoded_sys = encoder.predict(x_try) 
#decoded_sys = autoencoder.predict(x_try)
#decoded_sys_round = np.round(decoded_sys)
#error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))

#星座图
encoded_planisphere = encoder.predict(eye_matrix) 
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

