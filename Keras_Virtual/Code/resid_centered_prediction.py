"""
File: resid_centered_prediction.py
Author: ShogunHirei
Description: Neural Networks prediction of Mass and Momentum conservation 
             residues
"""

import os
import sys
from keras.models import Model
from keras.layers import Dense
from keras.utils import plot_model
from keras.regularizers import l1, l2
from keras import optimizers
from Scripts.auxiliar_functions import TrainingData, NeuralTopology
from joblib import dump
from datetime import datetime

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
             
# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
NOW = datetime.now()
BASE_DIR = './Models/Multi_Input/Regularization/' + NOW.strftime("%Y%m%d-%H%M%S") + '/'
SCALER_FOLDER = BASE_DIR + 'Scalers/'
os.mkdir(BASE_DIR)
os.mkdir(SCALER_FOLDER)

# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER)
DATA.scaler_folder = SCALER_FOLDER
DATA.save_dir = BASE_DIR

# Dados de treinamento
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.15,
                                                 inp_labels=['Point', 'Inlet'],
                                                 out_labels=['U', 'div', 'Res'],
                                                 # U_mag=False,
                                                 load_sc=False,
                                                 save_sc=True)


# Criando modelo de rede
tpl = NeuralTopology(MODEL=Model(), init_lyr=64) 


nets = tpl.multi_In_Out(DATA.ORDER[0], DATA.ORDER[1], 
                        # Criando hidden layers
                        LAYER_STACK = [Dense(64,
                                             bias_initializer='random_normal',
                                             kernel_regularizer=l1(l=0.001),
                                             activation='tanh') 
                                       for i in range(8)],
                        # Se adicionar mais uma camada densa para corrigir 
                        # dimensões
                        ADD_DENSE=False)

model = Model(inputs=nets[0], outputs=nets[1])

# Definindo optimizer
opt = optimizers.SGD(lr=0.01, momentum= 0.01, decay=0.01, nesterov=True)

# Compilando modelo
model.compile(optimizer=opt, loss='mse')

# Mostrando figura representativa do model
plot_model(model, to_file=BASE_DIR+'model.png', show_shapes=True)

# Salvando informações em tipo texto
tpl.set_result(model, BASE_DIR+'model_info.txt')

# Criando dicionários com os rotulos corretos e os respectivos dados
X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)
# Para criar os dicionários dedados de validação 
V_X = DATA.training_dict(X_TEST, 0)
V_Y = DATA.training_dict(Y_TEST, 1)

# Lista de Callbacks completa
CBCK = DATA.list_callbacks(BASE_DIR)

history = model.fit(X, Y, validation_data=(V_X, V_Y), batch_size=4, epochs=500, callbacks=CBCK)

# Gerar um arquivo para reconstrução do objeto history
print("Guardando History...")
dump(history, BASE_DIR + 'history_object.joblib')

print('FIM!! \o/')





