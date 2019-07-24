# REDES NEURAIS CONVOLUCIONAIS 
from keras.layers import Conv3D, Conv3DTranspose
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import SimpleRNNCell, LSTMCell, GRUCell
from keras.models import Sequential, Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# o array possui shape (20, 400, 3) para cada amostra 
# para 10 amostras o array completo fica (20, 10, 400,3)
# A rede elaborada até o momento é usar redes convolucionais e
# redes recorrentes
dummy_array = np.random.uniform(-1, 1, size=(10, 20, 400, 3))

x_train = dummy_array[:7, :, :, :]
print(x_train.shape)

# x_train, x_test, y_train, y_test = train_test_split()

# Parametro associado ao número de Reynolds para o problema de lid-cavity-flow
nu_input_param = np.linspace(0.0008, 0.4, 10)

# Definições para a rede neural que será elaborada

# Número de entradas, parametro e as velocidades 
num_inputs = 4 

# Número de neuronios na camada escondida
n_neuron = 30

# Usando a rede Convolutional de duas dimensões

model = Sequential()
# first_layer = Conv2D(filters=30, kernel_size=(), input_dim=(x_train.shape), data_format='chanels_last')
