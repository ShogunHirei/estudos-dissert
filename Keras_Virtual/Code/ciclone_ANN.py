from __future__ import print_function
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
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # , Normalizer, MinMaxScaler
from Scripts.auxiliar_functions import rec_function
# Módulos para a otimização dos hyperparametros da rede
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
import numpy as np

# Preparar input da rede neural
# Usando dados em caminho relativo

# Será utilizada a velocidade de entrada como parametro de distinção entre as
# análises


def data_generator():
    ANN_FOLDER = '/home/lucashqr/Documentos/Cursos/Keras Training/Virtual/'\
                 'estudos-dissert/Keras_Virtual/Ciclone/ANN_DATA/'

    # Extraindo informações de arquivos CSV e valor da velocidade de entrada
    DF = [(read_csv(dado.path), re.findall(r'\d+\.?\d*_', dado.path)[0][:-1])
          for dado in os.scandir(ANN_FOLDER)]

    N_SAMPLES = len(os.listdir(ANN_FOLDER))

    # Separando os dados de posição para X e Z (y fixo)
    XZ = [dado[0][['Points:0', 'Points:2']] for dado in DF]

    # Valores de velocidade com o mesmo shape dos outros inputs
    INPUT_U = [[float(dado[1])] * len(XZ[0]) for dado in DF]

    # Componentes de velocidade dos pontos (OUTPUT)
    U_xyz = [dado[0][['U:0', 'U:1', 'U:2']] for dado in DF]

    XZ = np.array([np.array(sample) for sample in XZ])
    U_xyz = np.array([np.array(sample) for sample in U_xyz])
    INPUT_U = np.array([np.array(sample) for sample in INPUT_U])

    # Convertendo shape das velocidades para ficarem de acordo input de posição
    INPUT_U = INPUT_U.reshape(XZ.shape[0], XZ.shape[1], 1)

    # Liberando espaço na memória
    del DF

    # ETAPA DE PADRONIZAÇÃO DE DADOS
    # Dados de posição são repetidos
    XZ_scaler = StandardScaler().fit(XZ[0])
    INPUT_U_scaler = StandardScaler().fit(INPUT_U[:, 0])
    Ux_scaler = StandardScaler().fit(U_xyz[..., 0])
    Uy_scaler = StandardScaler().fit(U_xyz[..., 1])
    Uz_scaler = StandardScaler().fit(U_xyz[..., 2])

    # Para reutilizar os parametros dos padronizadores
    # salvar em partes externas
    SC_DIR = './Models/Multi_Input/Scaler/'
    dump(XZ_scaler, SC_DIR+'points_scaler.joblib')
    dump(INPUT_U_scaler, SC_DIR+'U_input_scaler.joblib')
    dump(Ux_scaler, SC_DIR+'Ux_scaler.joblib')
    dump(Uy_scaler, SC_DIR+'Uy_scaler.joblib')
    dump(Uz_scaler, SC_DIR+'Uz_scaler.joblib')

    scaled_XZ = np.array([XZ_scaler.transform(sample) for sample in XZ])
    # ORIGINAL_XZ = np.array([ XZ_scaler.inverse_transform(sample) for sample in scaled_XZ])
    scaled_inputU = np.array(
        [INPUT_U_scaler.transform(sample) for sample in INPUT_U])
    scaled_Ux = np.array(Ux_scaler.transform(U_xyz[:, :, 0]))
    scaled_Uy = np.array(Uy_scaler.transform(U_xyz[:, :, 1]))
    scaled_Uz = np.array(Uz_scaler.transform(U_xyz[:, :, 2]))

    # Mudando o shape para adequar ao formato original
    scaled_Ux = scaled_Ux.reshape(N_SAMPLES, -1, 1)
    scaled_Uy = scaled_Uy.reshape(N_SAMPLES, -1, 1)
    scaled_Uz = scaled_Uz.reshape(N_SAMPLES, -1, 1)
    scaled_Uxyz = np.concatenate((scaled_Ux, scaled_Uy, scaled_Uz), axis=2)
    print(scaled_Uxyz.shape)

    # Concatenado os array de posição e velocidade para usar como entrada
    scaled_U_XZ = np.concatenate((scaled_XZ, scaled_inputU), axis=2)

    # PREPARANDO CONJUNTOS PARA TREINAMENTO E TESTE
    x_train, x_test, y_train, y_test = train_test_split(
                                                        scaled_U_XZ,
                                                        scaled_Uxyz,
                                                        test_size=0.15)

    return x_train, y_train, x_test, y_test

    # CRIANDO MODEL DE REDE NEURAL (Multi-Input)

    # Camada de inputs da rede


def model_creator(x_train, y_train, x_test, y_test):
    # Input de posição e velocidade de entrada
    XZ_input = Input(shape=(x_train.shape[1], x_train[..., :2].shape[-1]),
                     dtype='float32', name='XZ_input')

    # Criando camada completamente conectada
    XZ_out = Dense(256,
                   activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                   kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(XZ_input)

    # Camada de Input de Velocidade de entrada
    U_entr = Input(shape=(x_train.shape[1],
                          1), dtype='float32', name='U_entr')
    # Criando camada completamente conectada
    U_out = Dense(256,
                  activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                  kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(U_entr)

    # Concatenando as camadas de U_entr e XZ_input
    Conc1 = concatenate([XZ_out, U_out])

    # Criando Camadas escondidas
    x = Dense({{choice([128, 256, 512])}},
              activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
              kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(Conc1)
    x = Dense({{choice([128, 256, 512])}},
              activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
              kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(x)
    x = Dense({{choice([128, 256, 512])}},
              activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
              kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(x)

    if {{choice(['four', 'five', 'six'])}} == 'four':
        x = Dense({{choice([128, 256, 512])}},
                  activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                  kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(x)
    if {{choice(['four', 'five', 'six'])}} == 'five':
        x = Dense({{choice([128, 256, 512])}},
                  activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                  kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(x)
    if {{choice(['four', 'five', 'six'])}} == 'six':
        x = Dense({{choice([128, 256, 512])}},
                  activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                  kernel_regularizer=l2({{uniform(0.001, 0.01)}}))(x)

    # Output layer (obrigatoriamente depois)
    Output_layer = Dense(3,
                         activation={{choice(['tanh', None, 'sigmoid', 'softmax'])}},
                         name='Uxyz_Output')(x)

    # Criando modelo
    model = Model(inputs=[XZ_input, U_entr], outputs=[Output_layer])

    # COMPILANDO A REDE
    model.compile(optimizer={{choice(['rmsprop', 'nadam'])}},
                  loss={{choice(['hinge', 'mse', 'mae'])}},
                  metrics=['accuracy', 'mae'])

    # Gerando pastas para armazenar os dados do tensorboard
    FOLDER = './Models/Multi_Input/Optimzation/'
    NOW = datetime.now()
    LOGDIR = FOLDER + NOW.strftime("%Y%m%d-%H%M%S") + "/"
    os.mkdir(LOGDIR)

    # Criando Callbacks para poder ver o treinamento
    # Tensorboard
    TB = TensorBoard(log_dir=LOGDIR, write_grads=True,  #  histogram_freq=30,
                     write_images=False)

    # Interromper Treinamento
    ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=175,
                       restore_best_weights=True)

    # Reduzir taxa de aprendizagem
    RLRP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                             verbose=1, min_lr=1E-10)

    # Lista de Callbacks
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
    result = model.fit({'XZ_input': x_train[..., :2],
                        'U_entr': x_train[..., 0].reshape(x_train.shape[0], -1, 1)},
                       {'Uxyz_Output': y_train},
                       validation_data=({'XZ_input': x_test[..., :2],
                                         'U_entr': x_test[..., 0].reshape(x_test.shape[0], -1, 1)},
                                        {'Uxyz_Output': y_test}),
                       epochs=3000, batch_size={{choice([4, 8, 12])}}, callbacks=CBCK, verbose=1)
    # Salvar rede para gerar arquivos depois
    # mesma pasta TensorBoard
    SAVE_FOLDER = LOGDIR
    NET_NAME = "CicloneNet_" + NOW.strftime("%Y%m%d-%H%M%S")
    model.save(SAVE_FOLDER + NET_NAME)
    
    # Melhor modelo (tutorial)
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    print("Finished Trainning")
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


# Para chamar as funções e iniciar o treinamento
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model_creator,
                                          data=data_generator,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials(),
                                          eval_space=True)
    X_train, Y_train, X_test, Y_test = data_generator()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

# # Avaliando rede neural
# scores = model.evaluate({'XZ_input': X_TEST[..., :2],
                         # 'U_entr': X_TEST[..., 0].reshape(X_TEST.shape[0], -1, 1)},
                        # {'Uxyz_Output': Y_TEST})
# print(f'Acurácia do modelo: {scores}')


