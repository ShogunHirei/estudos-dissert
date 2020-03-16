"""
File: skip_connections_net.py
Author: ShogunHirei
Description: Usage of residual connections in neural network
                Goals: Speed-up convergence time
"""

import numpy as np
from keras import optimizers as Opts
from keras.models import Model
from keras.layers import Dense, Input, Reshape, concatenate, add, RepeatVector
from keras.callbacks import  TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from Scripts.auxiliar_functions import *
from joblib import dump

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
FOLD = sys.argv[2]
EVAL_CASE = sys.argv[3]
             
# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
BASE_DIR, INFO_DIR, SCALER_FOLDER = make_folder(FOLD)

# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER, FACTOR=5283.80102)
DATA.save_dir = BASE_DIR
DATA.scaler_folder = SCALER_FOLDER
DATA.info_folder = INFO_DIR

# Dados de treinamento
print("=".rjust(35,'=') + ' TRAINING ' + '='.ljust(35, '='))
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.20,
                                                 inp_labels=['Points', 'Inlet'],
                                                 out_labels=['U', 'div', 'Res'],
                                                 mag=[],
                                                 load_sc=False,)

# Creating dicts for inputs
X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)
# Para criar os dicionários dedados de validação 
V_X = DATA.training_dict(X_TEST, 0)
V_Y = DATA.training_dict(Y_TEST, 1)

# Skip conecttions model 
net_input = []
for key in X.keys():
    if key != 'Inlet_U':
        net_input.append(Input(shape=X[key].shape[1:], dtype='float32', name=key))
    else:
        # para a entrada da velocidade associada ao decoder
        B = Input(shape=(1,), dtype='float32', name='Inlet_U')


# A = Input(shape=(300,3), dtype='float32', name='A' ) # Camada para entrada dos pontos x,y,z
# B = Input(shape=(1,), dtype='float32', name='B') 

# Layers made with name sof the keys of dict
A = concatenate(net_input)

a = Dense(20, activation='tanh')(A)
a = Dense(20, activation='tanh')(a)
leaped_lyr = Dense(20, activation='tanh')(a)
leaped_lyr = Dense(20, activation='tanh')(leaped_lyr)
leaped_lyr = Dense(20, activation='tanh')(leaped_lyr)

# Velocity insertion
b = RepeatVector(int(net_input[0].shape[-2]))(B)

LY = concatenate([b, leaped_lyr])

c = Dense(20, activation='tanh')(LY)
c = Dense(20, activation='tanh')(c)

c = add([c,a])

d = Dense(20, activation='tanh')(c)

out_put = []
for key in Y.keys():
    out_put.append(Dense(1, activation='tanh', name=key)(d))

model = Model(net_input + [B], out_put)

# Optimizer setup
opt = Opts.Adam()

model.compile(optimizer=opt, loss={'U_0':'mae','U_1':'mae','U_2':'mae',
                                   'Res_0':'mse','Res_1':'mse','Res_2':'mse',                         
                                    },                              
             loss_weights={'U_0':0.9,'U_1':0.9,'U_2':0.9,
                           'Res_0':0.3,'Res_1':0.3,'Res_2':0.3,})


# Instation of class to saving model inforamtion
tpl = NeuralTopology()
# Persistence of model information
tpl.set_info(model, INFO_DIR+'model_info')

# Reshaping data for inlet_U
X['Inlet_U'] = X['Inlet_U'][:, 0, 0]
V_X['Inlet_U'] = V_X['Inlet_U'][:, 0, 0]

CBCK = DATA.list_callbacks(BASE_DIR)

hist = model.fit(X,Y,validation_data=(V_X, V_Y),
                 batch_size=16, epochs=300, callbacks=CBCK)

print("Salvando modelo para futuro treinamento")
model.save(BASE_DIR+f'model_trained.h5')






# For treating the fashion of the inlet_input

