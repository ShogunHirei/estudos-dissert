"""
Usar modelo de rede neural gerado para escrever
em formato interpretável pelo OpenFoam
"""

import sys
import numpy as np
from keras.models import load_model


# ETAPAS;
# -> Carregar modelo 
# -> Usar como dados de entrada NU e (x, y)
# -> Recuperar dados da rede
# -> Reescalar dados para as variáveis normais
# -> Abrir arquivo para escrita 
# -> Escrever dados no formato (ux uy 0)

######## CUIDADO COM OS VALORES PARA A RENORMALIZAÇÃO ############
R = [0.07165454545454514, 0.9974999999999956, 2.0816681711721685e-17, 0.8515943956457772, 0.4965164834260455]
xMED = [0.028345454545454868, 0.9974999999999956, 0.004999999999999979, 0.0008716043542227244, 0.0001354834260454543]

# Pegar nome do arquivo de modelo de rede neural a partir da linha de comando
FILENAME = sys.argv[1]
MODEL = load_model(FILENAME)

# Gerar input de dados
NU = 0.01
INPUT = []

# como os dados foram normalizados antes da inserção no modelo
# eles devem ser escalados para serem interpretados corretamente
norm = lambda x, i: (x - xMED[i])/(R[i])
re_norm = lambda x, i: x*R[i] + xMED[i]
NU = norm(NU, 0)
XY_input = []
x, y = 0, 0
for i in range(20):
    y += 0.1/20
    for j in range(20):
        x += 0.1/20
        XY_input.append([x, y])
        INPUT.append([norm(XY_input[j][0], 1), norm(XY_input[j][1], 2)])
    x = 0

[print(XY_input[p]) for p in range(25)]
INPUT = np.array(INPUT).reshape(-1, 400, 3)

PREDICTIONS = MODEL.predict(INPUT)

RE_NORM_PREDICTIONs = []

for data in PREDICTIONS[0]:
    RE_NORM_PREDICTIONs.append([re_norm(data[0], 3), re_norm(data[1], 4)])

print(RE_NORM_PREDICTIONs.__len__())

with open('./Generated Files/cavity_MLP_U', 'w') as results:
    for linha in RE_NORM_PREDICTIONs:
        results.write(f'({linha[0]:.6f} {linha[1]:.6f} 0)\n')

print('Finished!')


