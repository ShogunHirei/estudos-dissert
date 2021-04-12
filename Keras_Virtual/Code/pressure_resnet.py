"""
File: pressure_resnet.py
Author: ShogunHirei
Description: Pressure PINN-model
"""

import numpy as np
from keras import optimizers as Opts
from keras.models import Model
from keras.layers import Dense, Input, Reshape, concatenate, add, RepeatVector
from keras.callbacks import  TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from Scripts.auxiliar_functions import *
from joblib import dump

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
FOLD = sys.argv[2]
             
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
                                                 out_labels=['p', 'Res'],
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

# 56, 37, 25, 17, 12, 8, 6 <--- Encoder
# 37, +, 6, 10, 8, 6 <----Decoder 

# Layers made with names of the keys of dict
A = concatenate(net_input)

# Encoder 
enc_a = Dense(56, activation='tanh', name='enc_1')(A)
enc_b = Dense(37, activation='tanh', name='enc_2')(enc_a) # <--- The layer which will bypass
enc_a = Dense(25, activation='tanh', name='enc_3')(enc_b)
enc_a = Dense(17, activation='tanh', name='enc_4')(enc_a)
enc_a = Dense(12, activation='tanh', name='enc_5')(enc_a)
enc_a = Dense(8, activation='tanh', name='enc_6')(enc_a)
enc_a = Dense(6, activation='tanh', name='enc_7')(enc_a)

# Insertion of the Reynolds number
Inlet_U = RepeatVector(int(net_input[0].shape[-2]))(B)
# Concatenation of the last encoder layer with input Inlet
LY = concatenate([Inlet_U, enc_a])

# Decoder 
dec_a = Dense(37, activation='tanh', name='dec_1')(LY)
dec_a = add([dec_a, enc_b])   # <----- The layer adding the bypassed info
dec_a = Dense(6, activation='tanh', name='dec_2')(dec_a)
dec_a = Dense(10, activation='tanh', name='dec_3')(dec_a)
dec_a = Dense(8, activation='tanh', name='dec_4')(dec_a)
dec_a = Dense(6, activation='tanh', name='dec_5')(dec_a)

# Output layer
out_put = []
for key in Y.keys():
    out_put.append(Dense(1, activation='tanh', name=key)(dec_a))


# Model creation, connecting graphs
model = Model(net_input + [B], out_put)

# Optimizer setup
opt = Opts.Adam()

model.compile(optimizer=opt, loss={'p':'mse', 'Res_0':'mse','Res_1':'mse',
                                   'Res_2':'mse'},                              
              loss_weights={'p':0.7, 'Res_0':0.5,'Res_1':0.5,'Res_2':0.5,})


# Instation of class to saving model inforamtion
tpl = NeuralTopology()
# Persistence of model information
tpl.set_info(model, INFO_DIR+'model_info')

# Reshaping data for inlet_U
X['Inlet_U'] = X['Inlet_U'][:, 0, 0]
V_X['Inlet_U'] = V_X['Inlet_U'][:, 0, 0]

CBCK = DATA.list_callbacks(BASE_DIR, monit='p_loss')

# Save model after some training 
MDCPT = ModelCheckpoint(BASE_DIR + '/model_cpnt.h5', save_freq=250, save_best_only=True,
                        verbose=2, monitor='loss')

# Reduce output
ED = EpochDots(report_every=250)

CBCK += [MDCPT, ED]


hist = model.fit(X,Y,validation_data=(V_X, V_Y),
                 batch_size=48, epochs=5000, callbacks=CBCK)

print("Salvando modelo para futuro treinamento")
model.save(BASE_DIR+f'model_trained.h5')

