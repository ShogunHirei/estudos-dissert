"""
File: resid_centered_prediction.py
Author: ShogunHirei
Description: Neural Networks prediction of Mass and Momentum conservation 
             residues
"""

import os, re, sys
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.utils import plot_model
from keras.regularizers import l1, l2
from keras import optimizers
from keras.backend import sigmoid
from keras.layers import BatchNormalization
from sklearn.preprocessing import MaxAbsScaler
from Scripts.auxiliar_functions import TrainingData, NeuralTopology
from joblib import dump
from datetime import datetime

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
FOLD = sys.argv[2]
EVAL_CASE = sys.argv[3]
             
# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
NOW = datetime.now()
BASE_DIR = FOLD + NOW.strftime("%Y%m%d-%H%M%S") + '/'
SCALER_FOLDER = BASE_DIR + 'Scalers/'
INFO_DIR = BASE_DIR + 'Info/' 
os.mkdir(BASE_DIR)
os.mkdir(SCALER_FOLDER)
os.mkdir(INFO_DIR)

# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER, FACTOR=5283.80102, scaler=MaxAbsScaler)
DATA.scaler_folder = SCALER_FOLDER
DATA.save_dir = BASE_DIR
DATA.info_folder = INFO_DIR

# Dados de treinamento
print("=".rjust(35,'=') + ' TRAINING ' + '='.ljust(35, '='))
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.20,
                                                 inp_labels=['Points', 'Inlet'],
                                                 out_labels=['U', 'div', 'Res'],
                                                 mag=[],
                                                 load_sc=False,)

# Gerador para lidar com dados que não cabem na memória
def generator(dict_x,dict_y, batch_size, samples):
    low = 0
    high = batch_size
    while True:
        if high+batch_size < samples:
            low = high
            high += batch_size
            new_dict1 = {a:e[low:high] for a,e in dict_x.items()}
            new_dict2 = {a:e[low:high] for a,e in dict_y.items()}
            yield new_dict1, new_dict2
        else:
            high = samples
            new_dict1 = {a:e[low:high] for a,e in dict_x.items()}
            new_dict2 = {a:e[low:high] for a,e in dict_y.items()}
            yield new_dict1, new_dict2

# Implemetando swish activation function
def swish(x, beta=1):
    """
        File: resid_centered_prediction.py
        Function Name: swish
        Summary: Leve alteração da função sigmoid
        Description: Função de ativação definida em Bhatnagar (2019)
    """
    return  (x * sigmoid(beta * x))
    
# Registrando a função 



# Criando modelo de rede
tpl = NeuralTopology(MODEL=Model(), init_lyr=64) 

DEEP = 4

# AUTO_ENCODER_STACK = [Dense(int(tpl.layer0*3**(-i*0.9)),
                            # kernel_initializer='he_normal', activation=swish) 
                      # for i in range(DEEP) if i<=DEEP//2]
# AUTO_ENCODER_STACK += [Dense(int(tpl.layer0*3**((i-DEEP)*0.9)), 
                             # kernel_initializer='he_normal', activation=swish)
                       # for i in range(DEEP) if i>=DEEP//2 ]

AUTO_ENCODER_STACK = tpl.layer_stack_creation(DEEP, tpl.layer0, [2, 0.7])

for lyr in AUTO_ENCODER_STACK:
    if isinstance(lyr, Dense):
        AUTO_ENCODER_STACK.insert(AUTO_ENCODER_STACK.index(lyr)+1, BatchNormalization())

# print("AUTO_ENCODER_STACK -->", AUTO_ENCODER_STACK)

nets = tpl.multi_In_Out(DATA.ORDER[0], DATA.ORDER[1], 
                        # Criando hidden layers
                        LAYER_STACK = AUTO_ENCODER_STACK,
                        ADD_DENSE=False, MASKING=-101.0)

model = Model(inputs=nets[0], outputs=nets[1])

# Definindo optimizer
# opt = optimizers.SGD(lr=0.01, momentum= 0.01, decay=0.01, nesterov=True)
opt = optimizers.Adamax()

# Compilando modelo
model.compile(optimizer=opt, loss={'U_0':'mae','U_1':'mae','U_2':'mae',
                                   'Res_0':'mse','Res_1':'mse','Res_2':'mse',                         
                                    },                              
             loss_weights={'U_0':0.1,'U_1':0.5,'U_2':0.9,
                           'Res_0':0.1,'Res_1':0.1,'Res_2':0.1,})

print("Modelo Compilado!")

# Salvando informações em tipo texto
tpl.set_info(model, INFO_DIR+'model_info')

# Criando dicionários com os rotulos corretos e os respectivos dados
X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)
# Para criar os dicionários dedados de validação 
V_X = DATA.training_dict(X_TEST, 0)
V_Y = DATA.training_dict(Y_TEST, 1)

# Número de dados para os geradores
T_SMP = len(X[list(X.keys())[0]])
V_SMP = len(V_X[list(V_X.keys())[0]])

# Lista de Callbacks completa
CBCK = DATA.list_callbacks(BASE_DIR)

history = model.fit_generator(generator(X,Y, 10, T_SMP), 
                              # validation_data=generator(V_X, V_Y, 4, V_SMP),
                              validation_data=(V_X, V_Y),
                              steps_per_epoch=10, epochs=30,
                              # validation_steps=2,
                              # batch_size=32, epochs=5000,
                              callbacks=CBCK)

print("Salvando modelo para futuro treinamento")
model.save(BASE_DIR+f'model_{NOW}.h5')

# Gerar um arquivo para reconstrução do objeto history
print("Guardando History...")
dump(history, BASE_DIR + 'history_object.joblib')

print("Inserindo dados de previsão...")

# print("Determinando os valores de Inlet para todos os casos")
# suffix, caso_name, inp_data = DATA.pickup_data(V_X, RND=10.0)

# DATA.predict_data_generator(model, inp_data, f'slice_data_{suffix}.csv',
                            # DATA.data_folder+caso_name)

print("=".rjust(35,'=') + 'VALIDATION' + '='.ljust(35, '='))
EVAL = TrainingData(EVAL_CASE)
EVAL.save_dir = DATA.save_dir
EVAL.scaler_folder = DATA.scaler_folder

# Gerar resultados a partir de diretório com 74 planos
INPs, OUTs = EVAL.data_gen(inp_labels=['Points','Inlet'],
                           out_labels=['U', 'Res'],
                           mag=[],
                           # são utilizados os mesmos normalizadores para o
                           #  treinado, e com EVAL=True os dados não divididos
                           #  retornando apenas os arrays de inputs e outputs
                           load_sc=True, EVAL=True)

# Gerando lista de dicionários para serem inseridas em diretamente em model.predict
PRE_INP = [EVAL.training_dict(INPs[p][np.newaxis], 0) for p in range(len(INPs))];

# Função que chama model predict para todos os planos em PRE_INP
final_dataframe = DATA.batch_prediction(model, PRE_INP, EVAL.ORDER)

# Nome apenas para identificação 
name = str(list(set(final_dataframe['Inlet_U']))[0])

# Salvando full grid data
print('Salvando dados em:', EVAL.save_dir)
final_dataframe.to_csv(EVAL.save_dir + f'full_grid_prediction_{name}.csv', index=False)

print('FIM!! \o/')





