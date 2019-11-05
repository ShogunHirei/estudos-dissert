"""
File: isolated_prediction.py
Author: ShogunHirei
Description: Implementação de rede neuronal para prever componentes
             de velocidade isoladamente
"""

import re
import os
from datetime import datetime
from joblib import load
from pandas import read_csv, DataFrame, concat
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.regularizers import l2
from keras.initializers import Orthogonal
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
XZ_input = Input(
    shape=(X_TRAIN.shape[1], 2), dtype='float32', name='XZ_input')
# Criando camada completamente conectada
XZ_out = Dense(256, activation=None)(XZ_input)

# Camada de Input de Velocidade de entrada
U_entr = Input(
    shape=(X_TRAIN.shape[1], 1), dtype='float32', name='U_entr')
# Criando camada completamente conectada
U_out = Dense(256, activation=None)(U_entr)

# Concatenando as camadas de U_entr e XZ_input
Conc1 = concatenate([XZ_out, U_out])


#  HIDDEN LAYERS
x = Dense(200, activation='tanh')(Conc1)
x = Dense(150, activation='sigmoid')(x)
x = Dense(100, activation='tanh',)(x)
x = Dense(150, activation='sigmoid')(x)
x = Dense(200, activation='tanh')(x)
x = Dense(256, activation='sigmoid')(x)

# Output layer (obrigatoriamente depois)
Out_Ux = Dense(1, activation=None, name='Ux_Output')(x)
Out_Uy = Dense(1, activation='tanh', name='Uy_Output')(x)
Out_Uz = Dense(1, activation='tanh', name='Uz_Output')(x)

# Criando modelo
model = Model(inputs=[XZ_input, U_entr], outputs=[Out_Ux, Out_Uy, Out_Uz])

# Compilando modelo
model.compile(optimizer='nadam', loss='mse', metrics=['mae', 'acc'])

# Gerando pastas para armazenar os dados do tensorboard
FOLDER = './Models/Multi_Input/AutoEncoder/'
NOW = datetime.now()
BASE_DIR = FOLDER + NOW.strftime("%Y%m%d-%H%M%S") + '/'
os.mkdir(BASE_DIR)

# Criando Callbacks para poder ver o treinamento
# Tensorboard
TB = TensorBoard(log_dir=BASE_DIR, histogram_freq=90, write_grads=True,
                 write_images=False)

# Interromper Treinamento
ES = EarlyStopping(monitor='loss', min_delta=0.00001, patience=175,
                   restore_best_weights=True, )

# Reduzir taxa de aprendizagem
RLRP = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=30, verbose=1,
                         min_lr=1E-7)

# Lista de Callbacks completa
CBCK = [TB, ES, RLRP]

# ETAPA DE TREINAMENTO DA REDE
# Output de Arquitetura da rede para facilitar optimização
# Função recursiva que itera sobre o dicionário de configuração da rede
NET_CONFIG = model.get_config()
with open(f'{BASE_DIR}/lognet.txt', 'w') as logfile:
    rec_function(NET_CONFIG, logfile)

# Garantir que o shape depois de realizar o slice tenha a
# mesma quantidade de dimensões que o definido nas camadas Inputs

#  Treinamento de Ux
print("Início de treinamento")

# X_TRAIN.shape[0] é a quantidade de amostras
model.fit({'XZ_input': X_TRAIN[..., :2],
           'U_entr': X_TRAIN[..., 2].reshape(X_TRAIN.shape[0], -1, 1)},
          {'Ux_Output': Y_TRAIN[..., 0].reshape(Y_TRAIN.shape[0], -1, 1),
           'Uy_Output': Y_TRAIN[..., 1].reshape(Y_TRAIN.shape[0], -1, 1),
           'Uz_Output': Y_TRAIN[..., 2].reshape(Y_TRAIN.shape[0], -1, 1)},
          validation_data=({'XZ_input': X_TEST[..., :2],
                            'U_entr': X_TEST[..., 2].reshape(X_TEST.shape[0], -1, 1)},
                           {'Ux_Output': Y_TEST[..., 0].reshape(Y_TEST.shape[0], -1, 1),
                            'Uy_Output': Y_TEST[..., 1].reshape(Y_TEST.shape[0], -1, 1),
                            'Uz_Output': Y_TEST[..., 2].reshape(Y_TEST.shape[0], -1, 1)}
                           ),
          epochs=2000, batch_size=4, callbacks=CBCK, verbose=0)
print("Finished Trainning Ux")

# Salvar rede para gerar arquivos depois
# mesma pasta TensorBoard
print('Salvando modelo da rede')
SAVE_FOLDER = BASE_DIR
NET_NAME = "CicloneNet_" + NOW.strftime("%Y%m%d-%H%M%S")
model.save(SAVE_FOLDER + NET_NAME)


# Gerando dados para comparação com caso original
print("Gerando dados de previsão")
VEL_ARR = np.array([[10.0]*868]).reshape(-1, 1)
VEL_ARR = INPUT_U_scaler.transform(VEL_ARR).reshape(1, -1, 1)

# Valores previstos para Ux, Uy e Uz
PREDICs = model.predict({'XZ_input': scaled_XZ[0].reshape(1, -1, 2), 'U_entr': VEL_ARR})
print([p.shape for p in PREDICs])


# Retornando os dados para a escala anterior
Ux = DataFrame(Ux_scaler.inverse_transform(PREDICs[0][..., 0]).reshape(-1), columns=['U:0'])
Uy = DataFrame(Uy_scaler.inverse_transform(PREDICs[1][..., 0]).reshape(-1), columns=['U:1'])
Uz = DataFrame(Uz_scaler.inverse_transform(PREDICs[2][..., 0]).reshape(-1), columns=['U:2'])

# Inserindo valor dos pontos de Y
XYZ = read_csv(os.scandir(ANN_FOLDER).__next__().path)[['Points:0', 'Points:1', 'Points:2']]

# Geração de arquivo .CSV para leitura
FILENAME = f'NEW_SLICE_10_Isolated.csv'

SLICE_DATA = concat([Ux, Uy, Uz, XYZ], sort=True, axis=1)

# Escrevendo o header no formato do paraview
with open(BASE_DIR+FILENAME, 'w') as filename:
    HEADER = ''
    for col in list(SLICE_DATA.columns):
        HEADER += '\"' + col + '\",'
    filename.write(HEADER[:-1])
    filename.write('\n')

SLICE_DATA.to_csv(BASE_DIR + FILENAME, index=False, header=False, mode='a')
print("Dados de previsão copiados!")


# Diferença do valor previsto e o caso original
print("Calculando diferença...")
ORIGIN_DATA = read_csv(ANN_FOLDER+'SLICE_DATA_U_10_0.csv')

DIFF = SLICE_DATA[['U:0', 'U:1', 'U:2']] - ORIGIN_DATA[['U:0', 'U:1', 'U:2']]

RESULT_DATA = concat([DIFF, XYZ], axis=1)

print('Escrevendo dados DIFERENÇA')
RESULT_DATA.to_csv(BASE_DIR + 'DIFF_SLICE_U_10.csv', index=False)
print('Dados de diferença copiados!')
