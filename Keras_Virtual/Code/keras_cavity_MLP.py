"""
Regressão Problema de cavidade
Usando Redes Neurais MLP para determinar propriedades em simulação
"""
import os
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from datetime import datetime

# Definindo seed
# np.random.seed(12213119)

# Extração de dados dos arquivos do OpenFoam
# Extraindo o dados de velocidade do último tempo de todos os 11 exemplos

# Local do arquivo
CAVITY_FOLDER_REMOTE = r'../Cavity_Neural_Networks/'

# Regex para extrair dados numéricos
search_pattern = "\([-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*)?(e-)?\d* [-]?(\d*.\d*)?\)"

NU = {}
TIME = {}

N = 0 # para fazer apenas os dois primeiros casos

# Iterando em cada um dos exemplos de cavidade
for sample in os.scandir(CAVITY_FOLDER_REMOTE):
    if N == 2:
        break
    N += 1
    # Definindo a pasta que contem o dado de transporte
    TRANSPORT_FILE_PATH = sample.path + r'/constant/transportProperties'
    TIME_FILE_PATH = max([
        folder.name for folder in os.scandir(sample)
        if re.match("[0-9.]", folder.name)
    ])
    print(f"Nu_path: {TRANSPORT_FILE_PATH}, Time_path: {TIME_FILE_PATH}")
    with open(TRANSPORT_FILE_PATH) as trans_prop:
        for line in trans_prop.readlines():
            if 'nu' in line.split():
                NU[sample.name] = [
                    float(num[:-1]) for num in line.split()
                    if re.match(r'(\d*.\d*);', num)
                ][0]
    # Abrindo arquivo da velocidade do último tempo
    with open(sample.path + "/" + TIME_FILE_PATH + '/U') as U_file:
        U_VALUES = []
        # Adicionando a posição da célula como informação extra
        # para que a rede neural possa avaliar a posição da
        # célula na malha
        U_POSIX = 0
        U_POSIY = 0
        for line in U_file.readlines():
            # A avaliação pelo índice da célula não foi suficiente
            # ao invés do ID, vou usar a posiçãoda célula em uma
            # maneira cartesiana (Ux, Uy)
            if re.match(search_pattern, line):
                if U_POSIX % 20 == 0:
                    U_POSIY += 1
                    U_POSIX = 0
                U_value = (U_POSIX, U_POSIY, line.split(' ')[0].strip('('),
                           line.split(' ')[1],
                           line.split(' ')[2][:-1].strip(')'))
                U_value = tuple([float(num) for num in U_value])
                U_VALUES.append(U_value)
                U_POSIX += 1
        TIME[sample.name + '_' + TIME_FILE_PATH] = U_VALUES

# Foram gerados dois dicionários com os dados de velocidade e Nu (Reynolds)
# Transformá-los em array com a dimensão:
# (Nª componentes (Ux e Uy) * Nº amostra * e vetor posição)

DATA_ARRAY = []

for t in TIME.keys():
    nu, T = t.split('_')[0], t.split('_')[1]
    for U in TIME[t]:
        DATA_ARRAY.append([NU[nu], U[0], U[1], U[2], U[3]])

DATA_ARRAY = np.array(DATA_ARRAY)
print(DATA_ARRAY.shape)
# print(DATA_ARRAY[:35])

# ETAPA DE PRÉ-PROCESSAMENTO: NORMALIZAÇÃO DOS INPUTS
# normalização utilizada média em R, [-1,1]

NEW_ARRAY = np.array([])

print(NEW_ARRAY.shape)
# Calculando os valores máximos, mínimos e médio
# Calculando à parte para reduzir a quantidade de acesso 
#   a memória
# TODO: implementar função para fazer a normalização
#       verificar frameworks prontos de preprocessamento
XMAX = [max(p) for p in [DATA_ARRAY[..., d]
                         for d, _ in enumerate(DATA_ARRAY[0])]]

XMIN = [min(p) for p in [DATA_ARRAY[..., d]
                         for d, _ in enumerate(DATA_ARRAY[0])]]

XMED = [sum(p)/len(p) for p in [DATA_ARRAY[..., d]
                                for d, _ in enumerate(DATA_ARRAY[0])]]

R = [max([XMAX[p] - XMED[p], XMED[p] - XMIN[p]]) for p, _ in enumerate(XMAX)]
# Normalização dos dados
print(R, XMED)
for linha, dado in enumerate(DATA_ARRAY):
    NORM_LIST = np.array([])
    # NORM_LIST = NORM_LIST[tuple(np.newaxis for _ in range(len(DATA_ARRAY[0])-1))]
    # print(f"shape de norm: {NORM_LIST.shape}")
    for col, num in enumerate(dado):
        NORM_LIST = np.append(NORM_LIST, np.array([(num - XMED[col])/R[col]]))
    NEW_ARRAY = np.append(NEW_ARRAY, NORM_LIST, axis=0)

NEW_ARRAY = NEW_ARRAY.reshape(DATA_ARRAY.shape)
print(NEW_ARRAY.shape)

X = NEW_ARRAY[..., :3]
Y = NEW_ARRAY[..., 3:]

# X = X.reshape(-1, 20, 20, 3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# print(X_train.shape, '\n\n', X_train[:3],'\n\n', Y_train[:3])

X_train = X_train[np.newaxis]
Y_train = Y_train[np.newaxis]
X_test = X_test[np.newaxis]
Y_test = Y_test[np.newaxis]
print(X_train.shape)

# Para gerar um diretório para visualização no tensoboard
NOW = datetime.now()
LOGDIR = NOW.strftime("%Y%m%d-%H%M%S") + "/"
os.mkdir(f'./Models/MLP/logs/{LOGDIR}')

CALLBACK = [EarlyStopping(monitor='loss', min_delta=0, patience=150,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30,
                              verbose=1, min_lr=1E-10),
            TensorBoard(log_dir=f'./Models/MLP/logs/{LOGDIR}')]

# Número de neuronios da rede neural
NN = [512, 256, 128, 64, 32]

MODEL = Sequential()

MODEL.add(Dense(NN[2], activation='tanh', input_shape=(None, 3)))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[1], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[1], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[0], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[1], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[2], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(NN[3], activation='tanh'))
# MODEL.add(Dropout(0.35))
MODEL.add(Dense(2, activation='tanh'))

MODEL.compile(optimizer='rmsprop', loss='mean_absolute_error',
              metrics=['mse'])

MODEL.fit(X_train, Y_train, batch_size=NN[2], epochs=4000,
          verbose=1, callbacks=CALLBACK)

scores = MODEL.evaluate(X_test, Y_test)
print(f"Acc: {scores}")

# Iterando sobre as configurações das redes neurais para salvar nomes
# para automatizar pesquisas de melhores hyperparametros
NET_CONFIG = MODEL.get_config()
NET_NAME = ""
for idx, layer in enumerate(NET_CONFIG['layers']):
    TYPE = layer['class_name'][:3]
    UNITS = layer['config']['units']
    NET_NAME += TYPE + str(UNITS) + '_'

# Salvando o modelo da rede neural de acordo com a estrutura
MODEL.save(f'./Models/MLP/cavityMLP_{NET_NAME[:-1]}')
