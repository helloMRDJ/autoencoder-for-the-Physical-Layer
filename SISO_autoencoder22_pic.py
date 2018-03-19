#####################################################
#after run  SISO_autoencoder22_copy.py
#####################################################
from keras.models import load_model
import numpy as np 
import random as rd
import matplotlib.pyplot as plt



#initial
k = 2
n = 2
M = 2**k
R = k/n


eye_matrix = np.eye(M)
x_try = np.tile(eye_matrix, (250000, 1))
rd.shuffle(x_try)
print(x_try.shape) 

ER = []

#load model
#autoencoder = load_model('autoencoder22.h5')   #without n and BER

for Eb_No_dB in np.arange(-2.0, 10.0, 0.5):
    belta = 1/(2*R*(10**(Eb_No_dB/10)))
    belta_sqrt = np.sqrt(belta)
    noise_try = belta_sqrt * np.random.randn(np.shape(x_try)[0],n)
    #encoded_sys = encoder.predict(x_try) 
    decoded_sys_round = np.round(autoencoder.predict([x_try,noise_try]))
    error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))
    ER.append(error_rate)
    print(Eb_No_dB)
    print(error_rate)

#误码曲线
plt.yscale('log')
plt.plot(np.arange(-2.0, 10.0, 0.5),ER,'r.-')
plt.grid(True)
plt.ylim(10**-5, 1)
plt.xlim(-2, 10)
plt.title("Block error rate")