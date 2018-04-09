import numpy as np
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout,embeddings, Flatten, Add
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from  keras.optimizers import Adam, SGD
from keras import  backend as K
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(3)

NUM_EPOCHS = 100
BATCH_SIZE = 32
M = 4
k = np.log2(M)
k = int(k)
n_channel  = 2
emb_k = 4
R = k / n_channel
train_data_size=10000
bertest_data_size=50000
EbNodB_train = 7
EbNo_train = 10 ** (EbNodB_train / 10.0)
noise_std= np.sqrt( 1/ (2 * R * EbNo_train))
alpha = K.variable(0.5)
beta = K.variable(0.5)

#define the function for mixed AWGN channel
def mixed_AWGN(x):
    signal = x[0]
    interference = x[1]
    noise = K.random_normal(K.shape(signal),
                            mean=0,
                            stddev=noise_std)
    signal = Add()([signal, interference])
    signal = Add()([signal, noise])
    return signal

#define the dynamic loss weights
class Mycallback(Callback):
    def __init__(self,alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.epoch_num = 0
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num = self.epoch_num + 1
        loss1 = logs.get('u1_receiver_loss')
        loss2 = logs.get('u2_receiver_loss')
        print("epoch %d" %self.epoch_num)
        print("total_loss%f" %logs.get('loss'))
        print("u1_loss %f"%(loss1))
        print("u2_loss %f" % (loss2))
        a = loss1 / (loss1 + loss2)
        b = 1 - a
        K.set_value(self.alpha, a)
        K.set_value(self.beta, b)
        print("alpha %f" %K.get_value(alpha))
        print("beta %f" % K.get_value(beta))
        print("selfalpha %f" % K.get_value(self.alpha))
        print("selfbeta %f" % K.get_value(self.beta))

#generating train and test data
#user 1
seed(1)
train_label_s1 = np.random.randint(M,size= train_data_size)
train_label_out_s1 = train_label_s1.reshape((-1,1))
test_label_s1 = np.random.randint(M, size= bertest_data_size)
test_label_out_s1 = test_label_s1.reshape((-1,1))
#user 2
seed(2)
train_label_s2 = np.random.randint(M,size= train_data_size)
train_label_out_s2 = train_label_s2.reshape((-1,1))
test_label_s2 = np.random.randint(M, size= bertest_data_size)
test_label_out_s2 = test_label_s2.reshape((-1,1))

# Embedding Model for Two User using real signal
#user1's transmitter
u1_input_signal = Input(shape=(1,))
u1_encoded = embeddings.Embedding(input_dim=M, output_dim=emb_k, input_length=1)(u1_input_signal)
u1_encoded1 = Flatten()(u1_encoded)
u1_encoded2 = Dense(M, activation= 'relu')(u1_encoded1)
u1_encoded3 = Dense(n_channel, activation= 'linear')(u1_encoded2)
u1_encoded4 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x,axis=1))(u1_encoded3)
#u1_encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(u1_encoded3)
#user2's transmitter
u2_input_signal = Input(shape=(1,))
u2_encoded = embeddings.Embedding(input_dim=M, output_dim=emb_k, input_length=1)(u2_input_signal)
u2_encoded1 = Flatten()(u2_encoded)
u2_encoded2 = Dense(M, activation= 'relu')(u2_encoded1)
u2_encoded3 = Dense(n_channel, activation= 'linear')(u2_encoded2)
u2_encoded4 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x,axis=1))(u2_encoded3)
#u2_encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(u2_encoded3)

#mixed AWGN channel
u1_channel_out = Lambda(lambda x: mixed_AWGN(x))([ u1_encoded4, u2_encoded4])
u2_channel_out = Lambda(lambda x: mixed_AWGN(x))([ u2_encoded4, u1_encoded4])

#user1's receiver
u1_decoded = Dense(M, activation='relu',name= 'u1_pre_receiver')(u1_channel_out)
u1_decoded1 = Dense(M, activation= 'softmax', name= 'u1_receiver')(u1_decoded)

#user2's receiver
u2_decoded = Dense(M, activation='relu',name='u2_pre_receiver')(u2_channel_out)
u2_decoded1 = Dense(M, activation= 'softmax',name='u2_receiver')(u2_decoded)

twouser_autoencoder = Model(inputs=[u1_input_signal, u2_input_signal],
                            outputs=[u1_decoded1, u2_decoded1])
adam =Adam(lr = 0.01)
twouser_autoencoder.compile( optimizer=adam,
                             loss='sparse_categorical_crossentropy',
                             loss_weights=[alpha, beta])#loss=a*loss1+b*loss2
print(twouser_autoencoder.summary())
twouser_autoencoder.fit( [train_label_s1,train_label_s2],
                         [train_label_out_s1, train_label_out_s2],
                         epochs=45,
                         batch_size=32,
                         callbacks= [Mycallback(alpha,beta)])

#from keras.utils.vis_utils import plot_model
#plot_model(twouser_autoencoder, to_file= 'model.png')

#generating the encoder and decoder for user1
u1_encoder = Model(u1_input_signal, u1_encoded4)
u1_encoded_input = Input(shape= (n_channel,))
u1_deco = twouser_autoencoder.get_layer("u1_pre_receiver")(u1_encoded_input)
u1_deco = twouser_autoencoder.get_layer("u1_receiver")(u1_deco)
u1_decoder = Model(u1_encoded_input, u1_deco)

#generating the encoder and decoder for user1
u2_encoder = Model(u2_input_signal, u2_encoded4)
u2_encoded_input = Input(shape= (n_channel,))
u2_deco = twouser_autoencoder.get_layer("u2_pre_receiver")(u2_encoded_input)
u2_deco = twouser_autoencoder.get_layer("u2_receiver")(u2_deco)
u2_decoder = Model(u2_encoded_input, u2_deco)

#plotting the constellation diagram
#user1
u1_scatter_plot = []
for i in range(M):
    u1_scatter_plot.append(u1_encoder.predict(np.expand_dims(i,axis=0)))
u1_scatter_plot = np.array(u1_scatter_plot)
u1_scatter_plot = u1_scatter_plot.reshape(M, 2, 1)
plt.scatter(u1_scatter_plot[:, 0], u1_scatter_plot[:, 1],color='red',label = 'user1(2,2)')

u2_scatter_plot = []
for i in range(M):
    u2_scatter_plot.append(u2_encoder.predict(np.expand_dims(i,axis=0)))
u2_scatter_plot = np.array(u2_scatter_plot)
u2_scatter_plot = u2_scatter_plot.reshape(M, 2, 1)
plt.scatter(u2_scatter_plot[:, 0], u2_scatter_plot[:, 1], color = 'blue',label = 'user2(2,2)')

plt.legend(loc='upper left',ncol= 1)
plt.axis((-2.5, 2.5, -2.5, 2.5))
plt.grid()
fig = plt.gcf()
fig.set_size_inches(16,12)
#fig.savefig('graph/TwoUsercons(2,2)0326_1.png',dpi=100)
plt.show()

#calculating BER for embedding
EbNodB_range = list(np.linspace(0, 14 ,28))
ber = [None] * len(EbNodB_range)
u1_ber = [None] * len(EbNodB_range)
u2_ber = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo))
    noise_mean  = 0
    no_errors = 0
    nn = bertest_data_size
    noise1 = noise_std * np.random.randn(nn, n_channel)
    noise2 = noise_std * np.random.randn(nn, n_channel)
    u1_encoded_signal = u1_encoder.predict(test_label_s1)
    u2_encoded_signal = u2_encoder.predict(test_label_s2)
    u1_final_signal = u1_encoded_signal + u2_encoded_signal + noise1
    u2_final_signal = u2_encoded_signal + u1_encoded_signal + noise2
    u1_pred_final_signal = u1_decoder.predict(u1_final_signal)
    u2_pred_final_signal = u2_decoder.predict(u2_final_signal)
    u1_pred_output = np.argmax(u1_pred_final_signal, axis=1)
    u2_pred_output = np.argmax(u2_pred_final_signal, axis=1)
    u1_no_errors = (u1_pred_output != test_label_s1)
    u1_no_errors = u1_no_errors.astype(int).sum()
    u2_no_errors = (u2_pred_output != test_label_s2)
    u2_no_errors = u2_no_errors.astype(int).sum()
    u1_ber[n] = u1_no_errors / nn
    u2_ber[n] = u2_no_errors / nn
    ber[n] = (u1_ber[n] + u2_ber[n]) / 2
    print('U1_SNR:', EbNodB_range[n], 'U1_BER:', u1_ber[n])
    print('U2_SNR:', EbNodB_range[n], 'U1_BER:', u2_ber[n])
    print('SNR:', EbNodB_range[n], 'BER:', ber[n])

plt.plot(EbNodB_range, u1_ber ,label = 'TwoUserSNR(2,2)U1,emb_k=4,')
plt.plot(EbNodB_range, u2_ber ,label = 'TwoUserSNR(2,2)U2,emb_k=4,')
plt.plot(EbNodB_range, ber ,label = 'TwoUserSNR(2,2),emb_k=4,')

plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right',ncol= 1)

fig = plt.gcf()
fig.set_size_inches(16,12)
#fig.savefig('graph/TwoUserSNR(2,2)0326_1.png',dpi=100)
plt.show()
