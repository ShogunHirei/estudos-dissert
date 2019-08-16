import modred as mr
import numpy as np
from pandas import read_csv
from keras.layers import Dense, Input, concatenate
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

df = read_csv(r'./Redução de Ordem Algs/DATA_FOLDER/cavity_U_0.1.csv')

Y = df['Y']
NU = df['NU']

# Fixando X em 0.05, e usando Y e Nu como input da rede para prever
# a Magnitude da velocidade

# Magnitude de U
U_mag = df[['Ux', 'Uy']].apply(lambda x: (x[0]**2 + x[1]**2)**0.5, axis=1)

U_MAG = np.array(U_mag).reshape(11, 20, 20)
U_MAG_FIXED_X = U_MAG[:, :, 10]

# Para obter dados de Y para inserir no modelo de rede
Y = np.array(Y).reshape(11, 400)
# Variou de 20 em 20 para extrair o valor de Y para um X fixo
Y_DATA = Y[:, 0::20]

# São 11 casos (diferentes NU) com 20x20 elementos (X e Y)
# Então para comparar com os elementos de entrada de Y
# foram extraídos do universo de 400 para cada caso
# 20 AMOSTRAS DOS VALORES DE NU, de 20 em 20 até o fim do CASO
NU_DATA = np.array(NU).reshape(11, 400)
NU_DATA = NU_DATA[:, ::20]

# Gerar instância de transformador
NU_scaler = StandardScaler().fit(NU_DATA)
U_scaler = StandardScaler().fit(U_MAG_FIXED_X)
# USANDO DADOS TRANSPOSTOS PARA QUE O SCALER PADRONIZE OS DADOS
# EM FUNÇÃO DO EIXO 1 (axis=1)
Y_scaler = StandardScaler().fit(Y_DATA.T)

# Padronizando os dados
U_SCALED = U_scaler.transform(U_MAG_FIXED_X)
NU_SCALED = NU_scaler.transform(NU_DATA)
####  USANDO OS DADOS TRANSPOSTOS DE Y PARA QUE O SCALER PADRONIZE OS DADOS
Y_SCALED = Y_scaler.transform(Y_DATA.T)

## CRIAR AGRUPAMENTO DE DADOS PARA DIVISÃO POSTERIOR DE GRUPOS DE TREINAMENTO E VALIDAÇÃO
DATA = np.array([list(zip(NU_SCALED[p], Y_SCALED.T[p], U_SCALED[p])) for p in range(len(NU_DATA))])
# Transpor novamente os dados de Y para recuperar a disposição dos dados originais
# Separando em grupos de treinamento e validação
X_train, X_test, Y_train, Y_test = train_test_split(DATA[...,:2], DATA[...,2], test_size=0.2)

Y_train = Y_train[np.newaxis].reshape(Y_train.shape[0], 20, 1)
Y_test = Y_test[np.newaxis].reshape(Y_test.shape[0], 20, 1)

print(X_train.shape, Y_train.shape)

# Grupos separados e em ordem!

# Gerando modelo de rede neural para predizer magnitude da velocidade
#
# Modelo de Regressão Simples utilizando redes Densas

model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1], X_train.shape[2],)))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32))
model.add(Dense(64, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# Definindo função de custo e otimizador
model.compile(optimizer='rmsprop', loss='mse')

# Definindo Callback para obter dados relevantes no Tensorboard
# CBCK = [TensorBoard(log_dir='../Virtual/estudos-dissert/Keras_Virtual/Code/\
                             # Models/MLP/logs/jupyter/')]

model.fit(X_train, Y_train, batch_size=1, epochs=3000)

scores = model.evaluate(X_test, Y_test)

print(f"Acc: {scores}")

# Testando previsão dos dados para a primeira amostra
TEST_DATA = np.array(list(zip(NU_SCALED[10], Y_SCALED.T[0])))
# Adicionando novo eixo para
PREDICTIONS = model.predict(TEST_DATA[np.newaxis], batch_size=2)
# Adequando os dados para o shape do scaler
PREDICTIONS = PREDICTIONS.reshape(1, U_MAG_FIXED_X.shape[1])
# Voltando para a escala original do problema
PREDICTIONS = U_scaler.inverse_transform(PREDICTIONS[0])

# Plotando os dados para comparação gráfica dos resultados
y1 = U_MAG_FIXED_X[10]
y2 = PREDICTIONS

plt.plot(Y_DATA[3], y1)
plt.plot(Y_DATA[3], y2, 'r*')
plt.show()


