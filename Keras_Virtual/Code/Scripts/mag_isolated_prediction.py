"""
File: mag_isolated_prediction.py
Author: ShogunHirei
Description: Predição de componentes de velocidade e inserção da magnitude
             como parametro para elevar precisão
"""

import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, concatenate, Input
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # StandardScaler
from Scripts.auxiliar_functions import rec_function, TrainingData
from datetime import datetime
from pandas import DataFrame


# Carregando dados para Treinamento
ANN_FOLDER = '/home/lucashqr/Documentos/Cursos/Keras Training/'\
             'Virtual/estudos-dissert/Keras_Virtual/Ciclone/ANN_DATA/'

DATA = TrainingData(ANN_FOLDER)

# Dados de treinamento
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(U_mag=True)

# MODELO DE REDE NEURAL

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
Out_U_mag = concatenate([Out_Ux, Out_Uy, Out_Uz], name='Mag')

# Criando modelo
model = Model(inputs=[XZ_input, U_entr],
              outputs=[Out_Ux, Out_Uy, Out_Uz, Out_U_mag])

## Função loss Customizada para magnitude

def mag_loss(y_pred, y_true):
    """
        File: mag_isolated_prediction.py
        Function Name: mag_loss
        Summary: Função de custo para rede neural
        Description: Loss que adiciona a diferença da magnitude como penalidade
    """
    # Magnitude dos valores reais
    M_t = K.sqrt(K.sum(K.square(y_true), axis=-1))
    # Magnitude dos valores previstos
    M_p = K.sqrt(K.sum(K.square(y_pred), axis=-1))

    return K.mean(K.square(y_pred - y_true), axis=-1) + K.abs(M_p - M_t)


# Compilando modelo
model.compile(optimizer='rmsprop',
              loss={'Ux_Output': 'mse', 'Uy_Output': 'mse', 'Uz_Output': 'mse',
                    'Mag': mag_loss},
              loss_weights={'Ux_Output': 0.2, 'Uy_Output': 0.2, 'Uz_Output': 0.2,
                            'Mag': 1},
              metrics={'Ux_Output': 'mae', 'Uy_Output': 'mae', 'Uz_Output': 'mae',
                       'Mag': 'mae'})

# Gerando pastas para armazenar os dados do tensorboard
FOLDER = './Models/Multi_Input/AutoEncoder/'
NOW = datetime.now()
BASE_DIR = FOLDER + NOW.strftime("%Y%m%d-%H%M%S") + '/'
os.mkdir(BASE_DIR)

# Criando Callbacks para poder ver o treinamento
# Tensorboard
TB = TensorBoard(log_dir=BASE_DIR, histogram_freq=90, write_grads=False,
                 write_images=False)

# Interromper Treinamento
ES = EarlyStopping(monitor='mag_loss', min_delta=0.00001, patience=175,
                   restore_best_weights=True, )

# Reduzir taxa de aprendizagem
RLRP = ReduceLROnPlateau(monitor='mag_loss', factor=0.3, patience=30, verbose=1,
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
                            'Uz_Output': Y_TEST[..., 2].reshape(Y_TEST.shape[0], -1, 1),
                            'Mag': Y_TEST[..., 3].reshape(Y_TEST.shape[0], -1, 1)},
                           ),
          epochs=10000, batch_size=4, callbacks=CBCK, verbose=0)
print("Finished Trainning U")

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

