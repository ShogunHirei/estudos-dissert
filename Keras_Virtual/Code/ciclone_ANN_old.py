"""
File: ciclone_ANN.py
Author: ShogunHirei
Description: Uso de redes neurais para predizer componentes de velocidade em
             valor fixo de y (Slice em -0.2).
"""

import re
import os
from datetime import datetime
from joblib import dump
from pandas import read_csv
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.regularizers import l2
from keras.initializers import Orthogonal
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # StandardScaler, Normalizer, MinMaxScaler
from Scripts.auxiliar_functions import rec_function, TrainingData
import numpy as np

# Preparar input da rede neural
# Usando dados em caminho relativo

# Será utilizada a velocidade de entrada como parametro de distinção entre as
# análises

ANN_FOLDER = '/home/lucashqr/Documentos/Cursos/Keras Training/Virtual/'\
             'estudos-dissert/Keras_Virtual/Ciclone/ANN_DATA/'

# Usando a classe construída para obter os dados de trainamento
DATA = TrainingData(ANN_FOLDER)  # Usando MinMaxScaler

# Gerando o conjunto de dados de treinamento
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen()

# CRIANDO MODEL DE REDE NEURAL (Multi-Input)

# Camada de inputs da rede

# Input de posição e velocidade de entrada
XZ_input = Input(shape=(X_TRAIN.shape[1], 2),
                 dtype='float32', name='XZ_input')
# Criando camada completamente conectada
XZ_out = Dense(512, activation=None)(XZ_input)

# Camada de Input de Velocidade de entrada
U_entr = Input(shape=(X_TRAIN.shape[1], 1),
               dtype='float32', name='U_entr')
# Criando camada completamente conectada
U_out = Dense(512, activation=None)(U_entr)

# Concatenando as camadas de U_entr e XZ_input
Conc1 = concatenate([XZ_out, U_out])

# Criando Camadas escondidas
x = Dense(256, activation='tanh',)(Conc1)
x = Dense(256, activation='sigmoid',)(x)
x = Dense(128, activation='tanh')(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='sigmoid')(x)
x = Dense(512, activation='tanh')(x)

# Output layer (obrigatoriamente depois)
Output_layer = Dense(3, activation='sigmoid', name='Uxyz_Output')(x)

# Criando modelo
model = Model(inputs=[XZ_input, U_entr], outputs=[Output_layer])

# COMPILANDO A REDE
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', 'mae', 'hinge'])

# Gerando pastas para armazenar os dados do tensorboard
FOLDER = './Models/Multi_Input/AutoEncoder/'
NOW = datetime.now()
LOGDIR = FOLDER + NOW.strftime("%Y%m%d-%H%M%S") + "/"
os.mkdir(LOGDIR)

# Criando Callbacks para poder ver o treinamento
# Tensorboard
TB = TensorBoard(log_dir=LOGDIR, histogram_freq=30, write_grads=False,
                 write_images=False)

# Interromper Treinamento
ES = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=175,
                   restore_best_weights=True, baseline=0.7, mode='max')

# Reduzir taxa de aprendizagem
RLRP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1,
                         min_lr=1E-7)

# Lista de Callbacks completa
CBCK = [TB, ES, RLRP]

# ETAPA DE TREINAMENTO DA REDE
# Output de Arquitetura da rede para facilitar optimização
# Função recursiva que itera sobre o dicionário de configuração da rede
NET_CONFIG = model.get_config()
with open(f'{LOGDIR}lognet.txt', 'w') as logfile:
    rec_function(NET_CONFIG, logfile)

# Garantir que o shape depois de realizar o slice tenha a
# mesma quantidade de dimensões que o definido nas camadas Inputs
print("Início de treinamento")

# X_TRAIN.shape[0] é a quantidade de amostras
model.fit({'XZ_input': X_TRAIN[..., :2],
           'U_entr': X_TRAIN[..., 2].reshape(X_TRAIN.shape[0], -1, 1)},
          {'Uxyz_Output': Y_TRAIN},
          validation_data=({'XZ_input': X_TEST[..., :2],
                            'U_entr': X_TEST[..., 2].reshape(X_TEST.shape[0], -1, 1)},
                           {'Uxyz_Output': Y_TEST}),
          epochs=5000, batch_size=4, callbacks=CBCK, verbose=0)
print("Finished Trainning")


# Avaliando rede neural
scores = model.evaluate({'XZ_input': X_TEST[..., :2],
                         'U_entr': X_TEST[..., 2].reshape(X_TEST.shape[0], -1, 1)},
                        {'Uxyz_Output': Y_TEST})
print(f'Acurácia do modelo: {scores}')


# Salvar rede para gerar arquivos depois
# mesma pasta TensorBoard
SAVE_FOLDER = LOGDIR
NET_NAME = "CicloneNet_" + NOW.strftime("%Y%m%d-%H%M%S")
model.save(SAVE_FOLDER + NET_NAME)
