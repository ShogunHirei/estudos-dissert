"""
File: resid_centered_prediction.py
Author: ShogunHirei
Description: Neural Networks prediction of Mass and Momentum conservation 
             residues
"""

import os
import sys
import numpy as np
from keras.models import Model
from keras.layers import Dense, concatenate, Input
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from Scripts.auxiliar_functions import rec_function, TrainingData, mag_diff_loss
from Scripts.auxiliar_functions import NeuralTopology
from functools import partial, update_wrapper
from datetime import datetime
from pandas import DataFrame, read_csv, concat

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
             
# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER, scaler_dir='./test/')

# Dados de treinamento
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.25,
                                                 inp_labels=['Point', 'Inlet'],
                                                 out_labels=['U', 'div', 'Res'],
                                                 U_mag=True,
                                                 load_sc=False,
                                                 save_sc=True)

# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
# NOW = datetime.now()
# BASE_DIR = './Models/Multi_Input/AutoEncoder/' + NOW.strftime("%Y%m%d-%H%M%S") + '/'
# os.mkdir(BASE_DIR)

# Criando modelo de rede
tpl = NeuralTopology(MODEL=Model(), init_lyr=256)

# Criando hidden layers
STCK = [Dense(128, activation='tanh') for i in range(5)]

nets = tpl.multi_In_Out(DATA.ORDER[0], DATA.ORDER[1], LAYER_STACK=STCK)

model = Model(inputs=nets[0], outputs=nets[1])
model.compile(optimizer='rmsprop', loss='mse')

X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)

model.fit(X, Y, batch_size=8, epochs=5)
print('FIM!! \o/')





