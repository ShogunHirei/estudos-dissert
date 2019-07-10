''' *Getting started to with the Keras Functional API*
    Segundo item da Documentação do Keras
    Alguns exemplos iniciais

'''
#  TODO: lembrar de alterar os cabeçalhos com a explicação dos exemplos
# realizados no arquivo, tipo índice

# Usar dados de compra de clientes de compra de bicicleta
# do curso "Gentle Introduction to Keras"
from pandas import read_csv
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
data = read_csv('BBCN.csv').values

# Dados tipo numpy array( slice de x, slice de y)
X_data = data[:, 0:11]
Y_data = data[:, 11]

# Retorna um tensor com os dados que pode ser chamado
inputs = Input(shape=(len(X_data[0]),))

# Uma instância de camada pode ser chamada em um tensor, e também retorna um
# tensor
x = Dense(8, activation='softmax')(inputs)
x = Dense(8, activation='sigmoid')(x)
x = Dense(8, activation='softmax')(x)
predic = Dense(1, activation='sigmoid')(x)

# Criar a rede com 3 camadas (predic) e usando as entradas iniciais
model = Model(inputs=inputs, outputs=predic)

# Configuração de processo de aprendizagem da rede
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Começando treinamento da rede
model.fit(X_data[:420], Y_data[:420], batch_size=12, epochs=125)

# Avaliação do Modelo elaborado pela rede
# Para um batch_size de 12 e 125 epochs o melhor batch size no evaluate é 4
scores = model.evaluate(x=X_data[420:], y=Y_data[420:], batch_size=4)

# Mostrando precisão do modelo
print(f"A {model.metrics_names[1]}: {scores[1]*100} ")

# OBSERVAÇÕES:
#
#       Os objetos 'models' são reutilizáveis, ou seja, após definir a camada
# interna da rede, possível chamar a variável 'x' como um novo model, da forma:
#
# >> x = Input(shape=(500,));
#
# >> y = model(x)
#
# assim é possível processar sequencias de inputs, como variações de espaço de
# tempo (time_steps) para a entrada de dados descrito em:
#
# https://keras.io/getting-started/functional-api-guide/#all-models-are-callable-just-like-layers
#
#       Também é possível integrar redes através dessa funcionalidade, exemplo
# na página do link acima
#

