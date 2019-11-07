"""
File: zero_wall_loss.py
Author: ShogunHirei
Description: Rede neural para a previsão de vetor velocidade utilizando
             função loss que penaliza componentes de velocidade não-nulos
             na parede
"""

import os
import numpy as np
from datetime import datetime
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from Scripts.auxiliar_functions import TrainingData, rec_function, zero_wall_mag
from functools import partial, update_wrapper
from pandas import DataFrame, read_csv, concat

# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
NOW = datetime.now()
BASE_DIR = './Models/Multi_Input/AutoEncoder/' + NOW.strftime("%Y%m%d-%H%M%S") + '/'
os.mkdir(BASE_DIR)

# Geração de dados
# Utilizar vetor de velocidade para a previsão
# Carregando dados para Treinamento
ANN_FOLDER = '/home/lucashqr/Documentos/Cursos/Keras Training/'\
             'Virtual/estudos-dissert/Keras_Virtual/Ciclone/ANN_DATA/'

# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER, scaler_dir=BASE_DIR)

# Dados de treinamento
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.25,
                                                 load_sc=False,
                                                 save_sc=True)
# MODELO DE REDE NEURAL
# TODO: Criar função ou classe para implementar topologia de
#       de rede baseada em determinado padrão de entrada
# Camada de inputs da rede

# Input de posição e velocidade de entrada
XZ_input = Input(shape=(X_TRAIN.shape[1], 2), dtype='float32',  name='XZ_input')
# Criando camada completamente conectada
XZ_out = Dense(128, activation='tanh')(XZ_input)

# Camada de Input de Velocidade de entrada
U_entr = Input(
    shape=(X_TRAIN.shape[1], 1), dtype='float32', name='U_entr')
# Criando camada completamente conectada
U_out = Dense(128, activation='tanh')(U_entr)

# Concatenando as camadas de U_entr e XZ_input
Conc1 = concatenate([XZ_out, U_out])

#  HIDDEN LAYERS
x = Dense(128, activation='tanh')(Conc1)
x = Dense(64, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(16, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(64, activation='tanh')(x)
x = Dense(128, activation='tanh')(x)
Out_U_mag = Dense(3, activation='tanh', name='Mag')(x)

# Output layer (obrigatoriamente depois)
model = Model(inputs=[XZ_input, U_entr],
              outputs=[Out_U_mag])

# Pontos da parede para função de loss
wall_map = DATA.wall_data(X_TRAIN[0, :, :2])
print(wall_map.shape)

# Nova loss function com a velocidade nula na parede
new_mag = update_wrapper(partial(zero_wall_mag,
                                 wall_val=wall_map),
                         zero_wall_mag)

# Compilando modelo
model.compile(optimizer='rmsprop', loss={'Mag': new_mag})

# Criando Callbacks para poder ver o treinamento
# Tensorboard
TB = TensorBoard(log_dir=BASE_DIR, histogram_freq=100, write_grads=False,
                 write_images=False)

# Interromper Treinamento
ES = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=175,
                   restore_best_weights=True, )

# Reduzir taxa de aprendizagem
RLRP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=70, verbose=1,
                         min_lr=1E-10)

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
          {'Mag': Y_TRAIN.reshape(Y_TRAIN.shape[0], -1, 3)},
          validation_data=({'XZ_input': X_TEST[..., :2],
                            'U_entr': X_TEST[..., 2].reshape(X_TEST.shape[0], -1, 1)},
                           {'Mag': Y_TEST.reshape(Y_TEST.shape[0], -1, 3)},
                           ),
          epochs=30000, batch_size=4, callbacks=CBCK, verbose=1)
print("Finished Training U")


# Salvar rede para gerar arquivos depois
# TODO: Implementar função para geração de dados adicionais
# mesma pasta TensorBoard
print('Salvando modelo da rede')
SAVE_FOLDER = BASE_DIR
NET_NAME = "CicloneNet_" + NOW.strftime("%Y%m%d-%H%M%S")
model.save(SAVE_FOLDER + NET_NAME)


# Gerando dados para comparação com caso original
print("Gerando dados de previsão")
scaler_dict = DATA.return_scaler(load_sc=True)
VEL_ARR = np.array([[10.0]*868]).reshape(-1, 1)
VEL_ARR = scaler_dict['U_in'].transform(VEL_ARR).reshape(1, -1, 1)

# Valores previstos para Ux, Uy e Uz
PREDICs = model.predict({'XZ_input': X_TRAIN[..., :2][0].reshape(1, -1, 2), 'U_entr': VEL_ARR})
print([p.shape for p in PREDICs])


# Retornando os dados para a escala anterior
Ux = DataFrame(scaler_dict['Ux_scaler'].inverse_transform(PREDICs[..., 0]).reshape(-1), columns=['U:0'])
Uy = DataFrame(scaler_dict['Uy_scaler'].inverse_transform(PREDICs[..., 1]).reshape(-1), columns=['U:1'])
Uz = DataFrame(scaler_dict['Uz_scaler'].inverse_transform(PREDICs[..., 2]).reshape(-1), columns=['U:2'])

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
