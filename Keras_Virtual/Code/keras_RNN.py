# Estudos de Implementação de Redes Recorrentes pelo Keras

# ### -> Passos :
#    - Gerar dados com implementação de passo de tempo (similares ao OpenFoam)
#    - Criar modelo de Rede Recorrente
#    - Verificar avaliação da rede (precisão de predição)
#    - Verificar Visualização dos resultados

import re
import os
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping

# points = np.linspace(0, 2, num=100)

# x = points + 1.15 * np.random.random(100) - 0.15
# y = 2 * points**2 + 1.25 * np.random.random(100) - 0.25
# t = 2 * x**2 + 0.25 * y**3 + 0.25 * np.random.random(100)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.scatter3D(x, y, t, c=t)
# plt.show()

# Extrair dados dos timesteps nas simulações de cavity-flow
CAVITY_FOLDER_REMOTE = r'../Cavity_Neural_Networks/'
CAVITY_FOLDER_LOCAL = r'/home/lucashqr/OpenFOAM/lucashqr-6/run/'\
    'Cavity_Neural_Networks/'

print(CAVITY_FOLDER_LOCAL)

AMOSTRAS = os.listdir(CAVITY_FOLDER_REMOTE)

# Pegando a lista de tempos para o exemplo cavity
# DELTA_TS = [tempo for tempo in os.listdir(CAVITY_FOLDER+AMOSTRAS[0]) if re.match('\d.[\d]?', tempo)]
# DELTA_TS.sort()
# print(DELTA_TS)

# pattern_vector_3d = '\([-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*)?(e-)?\d* [-]?(\d*.\d*)?\)'

# Lendo os dados do campo vetorial da velocidade para os passos de tempo
# da primeira amostra

# TIME_DATA = {}


def time_steps_extractor(amostra, CAVITY_FOLDER):
    """
    função para extrair os dados de U
    """
    pattern_vector_3d = "\([-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*)?(e-)?\d* [-]?(\d*.\d*)?\)"

    DELTA_TS = [
        tempo for tempo in os.listdir(CAVITY_FOLDER + amostra)
        if re.match('\d.[\d]?', tempo)
    ]
    DELTA_TS.sort()

    TIME_DATA = {}

    for timestep in DELTA_TS:
        with open(CAVITY_FOLDER + amostra + '/' +
                  DELTA_TS[DELTA_TS.index(timestep)] + '/U') as U_file:
            data = U_file.readlines()
            U_values = [
                linhas for linhas in data
                if re.match(pattern_vector_3d, linhas)
            ]
            U_stripped = []
            # Para mudar os valores de string para float sem usar o eval
            for line in U_values:
                striped = line.strip()
                striped = striped.strip('(')
                striped = striped.strip(')')
                splited = striped.split(' ')
                U_stripped.append(tuple(float(number) for number in splited))
        TIME_DATA[timestep] = U_stripped

    TIME_ARRAY = {float(i): np.array(o) for i, o in TIME_DATA.items()}
    return TIME_ARRAY


cavity0 = time_steps_extractor(AMOSTRAS[0], CAVITY_FOLDER_REMOTE)

cavity1 = time_steps_extractor(AMOSTRAS[1], CAVITY_FOLDER_REMOTE)

cavity2 = time_steps_extractor(AMOSTRAS[2], CAVITY_FOLDER_REMOTE)

batch = [cavity0, cavity1, cavity2]

print(f"Time steps: {len(cavity2)}, No of CV's: {len(cavity2[0.005])}"
      f", Data for CV: {len(cavity2[0.005][0])}")

cavity0_array = np.array([p for p in cavity0.values()])

X_train, Y_train = cavity0_array[:15][..., :], cavity0_array[1:16][..., :]

print(X_train.shape, Y_train.shape)

print(cavity0_array.shape)

prediction_array = cavity0_array[16].reshape(-1, 400, 3)

print(prediction_array.shape)

# PARAMETROS DE REDE,
# Inserindo isso para facilitar em rastrear as melhores redes
num_neurons = 256
num_neurons_d = 60

MODEL = Sequential()

MODEL.add(GRU(num_neurons, return_sequences=True, input_shape=(400, 3)))
MODEL.add(LSTM(num_neurons, return_sequences=True))
MODEL.add(Dense(num_neurons_d, activation='tanh'))
MODEL.add(Dense(num_neurons_d, activation='tanh'))
MODEL.add(Dense(3, activation='tanh'))

MODEL.compile('rmsprop', loss='mean_squared_logarithmic_error')

CALLBACK_MODEL = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5,
                               restore_best_weights=True)

MODEL.fit(x=X_train, y=Y_train, batch_size=16,
          epochs=100, verbose=1, callbacks=[CALLBACK_MODEL])

# Salvar a rede neural gerada para uso no futuro
MODEL.save(f'Models/rede_{num_neurons}_{num_neurons_d}')

scores = MODEL.evaluate(cavity0_array[:8][..., :], y=cavity0_array[1:9][..., ])

print(f"Scores: {scores}")

print(list(cavity0.keys())[12])
# print(cavity0_array[12])

predictions = MODEL.predict(prediction_array)

print(predictions)


# print(DATA[0])
