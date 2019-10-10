#!/usr/bin/env python
# coding: utf-8

# In[1]:


import modred as mr
import numpy as np
from pandas import read_csv
from keras.layers import Dense, Input, concatenate
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split


# In[2]:


df = read_csv('/home/lucashqr/Documentos/Cursos/Keras Training/Virtual/estudos-dissert/Keras_Virtual/Code/Redução de Ordem Algs/DATA_FOLDER/cavity_U_0.1.csv')

X, Y = df['X'], df['Y']
Ux, Uy = df['Ux'], df['Uy']
NU = df['NU']

XY = list(zip(*[X, Y]))
DATA = list(zip(*[X, Y, NU]))

XY = np.array(XY).reshape(11, 400, 2)
print(XY.shape)

Ux = np.array(Ux).reshape(11,-1)
print(Ux.shape)

Uy = np.array(Uy).reshape(11,-1)
print(Uy.shape)
    
DATA = np.array(DATA).reshape(11, -1, 3)
print(DATA.shape)


# In[3]:


NU = np.array(NU).reshape(11, -1, 1)  # Para ficar em 3D
# X, Y = np.array(X).reshape((11, 20, -1)) , np.array(Y).reshape((11, 20, -1))
# Ux, Uy = np.array(Ux).reshape((-1, 400)) , np.array(Uy).reshape((-1, 400))
# Ux = Ux.reshape(11, )
# Separado os arrays para usar na entrada da rede


# In[4]:


# Criando modelo para prever Ux com dados de X e Y 
#                 X (shape 11,20,20)     Y(shape 11,20,20)
#                               \           /
#    NU (11,400,1)               XY (11, 400)
#    |______________________________|
#                       |
#                       |  concatenate inputs
#                       | 
#                     Conc1 
#                       | 
#                       | rede
#                       | 
#                     Output

XY_Input_Layer = Input(shape=(400, 2), dtype="float32", name='XY_input')
XY_output = Dense(256, activation=None)(XY_Input_Layer)

# Y_input_Layer = Input(shape=(20,), dtype="float32", name='Y_input')
# Y_output = Dense(256, activation=None)(Y_input_Layer)

NU_aux_input = Input(shape=(400,1), dtype="float32", name="NU_input")
NU_out = Dense(256, activation=None)(NU_aux_input)

# Conc1 = concatenate([X_output, Y_output, NU_out])
Conc1 = concatenate([XY_output, NU_out])
Conc1.shape


# In[5]:


# Criando rede (auto encoder)
x = Dense(256, activation='sigmoid')(Conc1)
x = Dense(64, activation=None)(x)
x = Dense(256, activation='sigmoid')(x)

# Output layer (obrigatoriamente depois)
Output_layer = Dense(2, activation=None, name='Ux_Output')(x)

# Criando modelo 
model = Model(inputs=[XY_Input_Layer, NU_aux_input], outputs=[Output_layer])


# In[6]:


# Etapa de Pré-processamento, 
NU = NU.reshape(11, -1)
print(NU.shape, sorted(list(set(list(NU.reshape(-1))))))

XY_scaler = StandardScaler().fit(XY[0])
Ux_scaler = StandardScaler().fit(Ux)
Uy_scaler = StandardScaler().fit(Uy)
NU_scaler = StandardScaler().fit(NU)

scaled_XY = np.array([XY_scaler.transform(XY[p]) for p in range(len(XY)) ])
scaled_Ux = Ux_scaler.transform(Ux)
scaled_Ux = scaled_Ux.reshape(11,-1,1)

scaled_Uy = Uy_scaler.transform(Uy)
scaled_Uy = scaled_Uy.reshape(11,-1,1)

NU_scaled = NU_scaler.transform(NU).reshape(11, -1, 1)
print('set', sorted(list(set(list(NU_scaled.reshape(-1))))))

NU_XY_scaled = np.concatenate((NU_scaled, scaled_XY), axis=2)
print(NU_XY_scaled.shape, '\n\n')

# Juntando as componentes da velocidade
U_xy_scaled = np.concatenate((scaled_Ux, scaled_Uy), axis=2)
print(U_xy_scaled.shape)

# Conjuntos de Teste e Treinamento
# X_train, X_test, Y_train, Y_test = train_test_split(scaled_XY, scaled_Ux, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(NU_XY_scaled, U_xy_scaled, test_size=0.2, random_state=42)

# Para a inserção na camada de entrada extra 
NU_train, NU_test = X_train[...,0], X_test[...,0]
NU_train, NU_test = NU_train.reshape(len(X_train), 400, 1), NU_test.reshape(len(X_test), 400, 1)
print('set', set(list(NU_test.reshape(-1))))
print(X_test.shape)


# In[7]:


# Compilando a rede
model.compile(optimizer='rmsprop', loss='mse')


# In[8]:


# TensorBoard K
CBCK = TensorBoard(log_dir='../Virtual/estudos-dissert/Keras_Virtual/Code/Models/MLP/logs/jupyter_Multi_input/')

# Rodando a rede com os dados de entrada X e Nu
model.fit({'XY_input':X_train[...,1:],'NU_input':NU_train}, {'Ux_Output':Y_train}, epochs=300, batch_size=16, 
         callbacks=[CBCK])


# In[16]:


# Avaliação do modelo para os casos restantes 
model.evaluate({'XY_input':X_test[..., 1:], 'NU_input':NU_test}, {'Ux_Output':Y_test})


# In[10]:


predictionsX = model.predict({'XY_input':X_test[..., 1:][2].reshape(1,-1,2),'NU_input':X_test[..., 0][2].reshape(1, -1, 1)})

# print(predictionsX[..., 0].shape) # resultados Ux e Uy, dimensão ok!

# Retornando os dados para a escala normal
norm_Ux = Ux_scaler.inverse_transform(predictionsX[..., 0])

norm_Uy = Uy_scaler.inverse_transform(predictionsX[..., 1])

re_NU = NU_scaler.inverse_transform(X_test[..., 0][2])
print(set(list(re_NU)))

print(norm_Uy.shape)

Final_U = np.array(list(zip(*[norm_Ux[0], norm_Uy[0], [0]*400 ])))
print(Final_U.shape)

with open('../Virtual/estudos-dissert/Keras_Virtual/Code/Generated Files/cavity_MULTI_2','w') as results:
    for linha in Final_U:
        results.write(f'({linha[0]:.6f} {linha[1]:.6f} {linha[2]})\n')


# In[ ]:




