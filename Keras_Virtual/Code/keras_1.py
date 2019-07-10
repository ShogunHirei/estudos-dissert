'''
Estudo do módulo Keras para a geração de redes neurais,
como o objetivo desse estudo é aprender a elaborar redes neurais mais complexas
utilizando esse módulo, pretendo criar pequenas aplicações durante o progresso
dos estudos para adquirir conhecimento sobre
'''

# Usando como referência a documentação do Keras disponível online

# Módulos: Sequential e Dense

from keras.models import Sequential
from keras.layers import Dense
from numpy import random as rd
from sklearn.decomposition import PCA

# Possível criar sequencias de camadas para a topologia  da rede

# Objeto inicializado,
model = Sequential()
# A rede pode ser criada como uma lista das camadas

# A primeira camada deve ter seu número de neuronios especificados
# pode ser realizado pelo função .add(), que recebe como argumento
# um objeto do tipo Layer (modulo keras.layers)
model.add(Dense(32, input_shape=(6,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Foram inseridas 32 camadas na rede, por ser do tipo Dense, elas
# são completamente conectadas.

# Aparentemente a última saída da última camada deve ser igual em dimensão
# aos targets

# Após criar a topologia e antes da etapa de treinamento,
# é necessário compilar a rede para configurar o processo de aprendizagem

# Isso pode ser feito pelo comando compile
model.compile(optimizer='sgd',
              loss='squared_hinge',
              metrics=['accuracy'])

# Criar valores Randomicos para simular
rd.seed(12213119)
# 270 linhas e 6 colunas
X_data = rd.rand(1500, 6)
# "features" 270 linhas e uma coluna
Y_data = rd.randn(1500, 1)

# Para poder treinar a rede usar model.fit
# Definir número de epochs e quantidade de dados na batelada
model.fit(X_data[:1200], Y_data[:1200], batch_size=10, epochs=200)

scores = model.evaluate(X_data[1200:], Y_data[1200:], batch_size=10)

print(f"O valor da {model.metrics_names[1]} é {scores[1]*100}")

print(scores)
