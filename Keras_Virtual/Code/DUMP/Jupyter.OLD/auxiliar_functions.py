"""
File: auxiliar_functions.py
Author: ShogunHirei
Description: Funções utilizadas repetidamente durante a implementação.
"""

# Função utilizada em ciclone_ANN para obter a estrutura da rede no output
import re
import os
import numpy as np
from joblib import dump, load
from pandas import read_csv, concat, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense
import tensorflow as tf


class TrainingData:
    """
    File: training_data.py
    Author: ShogunHirei
    Description: Script para gerar dados de treinamento para facilitar
                 a scrita do código.
    """

    def __init__(self, data_folder, scaler=MinMaxScaler, scaler_dir='./'):
        self.data_folder = data_folder
        self.scaler = scaler
        self.N_SAMPLES = len(os.listdir(self.data_folder))
        self.scaler_folder = scaler_dir

    def data_gen(self, test_split=0.2, U_mag=False, load_sc=True, save_sc=False):
        """
            File: training_data.py
            Function Name: data_gen
            Summary: Gerar dados de treinamento
            Description: Usar pasta com arquivos .csv e gerar conjuntos
                         de treinamento e teste.

            U_mag -> Insere a magnitude da velocidade nos componentes
            load_sc -> Se for para carregar os scalers salvos
            save_sc -> não carregar, criar novos scaler e salvá-los


        """

        DF = [(read_csv(dado.path), re.findall(r'\d+\.?\d*_', dado.path)[0][:-1])
              for dado in os.scandir(self.data_folder)]

        XZ = np.array([np.array(sample) for sample in
                       [dado[0][['Points:0', 'Points:2']] for dado in DF]])

        # Valores de velocidade com o mesmo shape dos outros inputs
        INPUT_U = np.array([np.array(sample) for sample in
                            [[float(dado[1])] * len(XZ[0]) for dado in DF]])

        # Componentes de velocidade dos pontos (OUTPUT)
        U_xyz = np.array([np.array(sample) for sample in
                          [dado[0][['U:0', 'U:1', 'U:2']] for dado in DF]])

        # Convertendo shape das velocidades para ficarem de acordo input de posição
        INPUT_U = INPUT_U.reshape(XZ.shape[0], XZ.shape[1], 1)

        # Carregando padronizadores
        scaler_dic = self.return_scaler(load_sc=load_sc, save_sc=save_sc,
                                        data_input=[XZ, INPUT_U, U_xyz])

        # Cada valor do dicionário scaler_dic referencia seu padronizador
        # utilizando isso para escalonar os dados
        scaled_XZ = np.array([scaler_dic['XZ'].transform(sample) for sample in XZ])
        scaled_inputU = np.array([scaler_dic['U_in'].transform(sample)
                                  for sample in INPUT_U])
        scaled_Ux = np.array(scaler_dic['Ux_scaler'].transform(U_xyz[:, :, 0]))
        scaled_Uy = np.array(scaler_dic['Uy_scaler'].transform(U_xyz[:, :, 1]))
        scaled_Uz = np.array(scaler_dic['Uz_scaler'].transform(U_xyz[:, :, 2]))

        # Liberando espaço na memória
        del DF, scaler_dic

        # Mudando o shape para adequar ao formato de entrada de 3 dimensões
        # Concatenando arrays dos componentes de velocidade
        scaled_Uxyz = np.concatenate((scaled_Ux.reshape(self.N_SAMPLES, -1, 1),
                                      scaled_Uy.reshape(self.N_SAMPLES, -1, 1),
                                      scaled_Uz.reshape(self.N_SAMPLES, -1, 1)),
                                     axis=2)

        # Adicionar dados de magnitude caso necessário
        if U_mag:
            U_mag_scaled = self.U_mag_data_gen([DataFrame(sample)
                                                for sample in U_xyz]).reshape(self.N_SAMPLES, -1, 1)
            scaled_Uxyz = np.concatenate((scaled_Uxyz, U_mag_scaled), axis=2)

        del XZ, INPUT_U, U_xyz

        (X_TRAIN, X_TEST,
         Y_TRAIN, Y_TEST) = train_test_split(np.concatenate((scaled_XZ,
                                                             scaled_inputU),
                                                            axis=2),
                                             scaled_Uxyz,
                                             test_size=test_split)

        print("Shape of X_TRAIN: ", X_TRAIN.shape)
        print("Shape of Y_TRAIN: ", Y_TRAIN.shape)
        print("Shape of X_TEST: ", X_TEST.shape)
        print("Shape of Y_TEST: ", Y_TEST.shape)

        return (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    def U_mag_data_gen(self, U_xyz, scaler_ld_sv=[0, 0]):
        """
            File: training_data.py
            Function Name: U_mag_data_gen
            Summary: Gerar dados de magnitude da velocidade para concatenação
            Description: Função interna para agregar dados de magnitude
        """

        U_mag = [(sample**2).apply(np.sum, axis=1).apply(np.sqrt)
                 for sample in U_xyz]
        U_mag = np.array([np.array(sample) for sample in U_mag])

        if scaler_ld_sv[0]:
            U_mag_scaler = load(self.scaler_folder+'U_mag_scaler.joblib')
        else:
            U_mag_scaler = self.scaler().fit(U_mag)

        if scaler_ld_sv[1]:
            dump(U_mag_scaler, self.scaler_folder+'U_mag_scaler.joblib')

        U_mag_scaled = U_mag_scaler.transform(U_mag)

        return U_mag_scaled

    def return_scaler(self, load_sc=True, save_sc=False, data_input=None):
        """
            File: auxiliar_functions.py
            Function Name: return_scaler
            Summary: Retornar as funções utilizadas na padronização dos dados
            Description: Função que retorna os padronizadores para
                         serem reutilizados de outras formas além da geração
                         do conjunto de dados de treinamento e teste
        """

        if load_sc:
            XZ_scaler = load(self.scaler_folder+'points_scaler.joblib')
            INPUT_U_scaler = load(self.scaler_folder+'U_input_scaler.joblib')
            Ux_scaler = load(self.scaler_folder+'Ux_scaler.joblib')
            Uy_scaler = load(self.scaler_folder+'Uy_scaler.joblib')
            Uz_scaler = load(self.scaler_folder+'Uz_scaler.joblib')
        else:
            XZ = data_input[0]
            INPUT_U = data_input[1]
            U_xyz = data_input[2]
            XZ_scaler = self.scaler().fit(XZ[0])
            INPUT_U_scaler = self.scaler().fit(INPUT_U[:, 0])
            Ux_scaler = self.scaler().fit(U_xyz[..., 0])
            Uy_scaler = self.scaler().fit(U_xyz[..., 1])
            Uz_scaler = self.scaler().fit(U_xyz[..., 2])

        if save_sc:
            dump(XZ_scaler, self.scaler_folder+'points_scaler.joblib')
            dump(INPUT_U_scaler, self.scaler_folder+'U_input_scaler.joblib')
            dump(Ux_scaler, self.scaler_folder+'Ux_scaler.joblib')
            dump(Uy_scaler, self.scaler_folder+'Uy_scaler.joblib')
            dump(Uz_scaler, self.scaler_folder+'Uz_scaler.joblib')

        SCALER_DICT = {'XZ': XZ_scaler, 'U_in': INPUT_U_scaler,
                       'Ux_scaler': Ux_scaler, 'Uy_scaler': Uy_scaler,
                       'Uz_scaler': Uz_scaler}

        return SCALER_DICT

    def wall_data(self, XZ_DATA):
        """
            File: auxiliar_functions.py
            Function Name: wall_data
            Summary: Extrair dados do slice
            Description: Criar dados para a inserção na função zero_wall_mag
        """
        wall = []
        # Vetor A corresponde a diferença dos pontos entre o meio e extremo
        # para selecionar os pontos da parede
        # PNT1: (med1, med2) centro
        # PNT2: (max1, med2) extremo no eixo 1 e centro no eixo2
        A = np.array([np.mean(XZ_DATA[:, 0]) - np.max(XZ_DATA[:, 1]), 0])
        for pnt in XZ_DATA:
            # Vetor B, vetor do ponto analisado
            B = np.array(pnt) - np.array([np.mean(XZ_DATA[:, 0], axis=0),
                                          np.mean(XZ_DATA[:, 1])])
            B_mag = np.sqrt(np.sum(np.square(B)))
            A_mag = np.sqrt(np.sum(np.square(A)))
            if B_mag >= A_mag*0.99:  # 0.99 correção para pnts na parede
                tmp = [p for p in pnt]
                tmp.append(1)
                wall.append(np.array(tmp))
            else:
                wall.append(np.array([0, 0, 0]))
        wall = np.array(wall).reshape(-1, 3)
        return wall

    def organized_data(self, data):
        """
            File: auxiliar_functions.py
            Function Name: organized_data
            Summary: Extrair dados de forma organizada
            Description: Usa a magnitude de um vetor da origem à extremidade
                         e guarda os dados que estão em mesma condição (círculo)
                         manda para uma lista
        """
        # TODO: FALTA CONSERTAR MAGNITUDE PARA QUADRADO NO MEIO
        # NOTE: Adicionar etapa de verificação de dados, 'np.intersect1d'
        xz = np.concatenate((data, np.zeros((len(data), 1))), axis=1)
        # função de magnitude 'atalho'
        mag = lambda x: np.sqrt(np.sum(np.square(x)))

        # Vetor A de referência
        A = np.array([np.max(xz[:, 0]), np.mean(xz[:, 1]), 0]) - np.mean(xz, axis=0)
        arm2 = []
        MEAN_ORIG = np.mean(xz, axis=0)
        while mag(A) > 0.1455:
            arm1 = []
            for pnt in xz:
                # vetor B de comparação com A
                B = pnt - MEAN_ORIG
                if mag(B) >= 0.97 * mag(A):
                    # Adicionado um item a mais por causa do vetor velocidade
                    arm1.append([p for p in pnt])
                else:
                    # para manter as mesmas dimensões do mapeamento dos pontos '(868, 3)'
                    arm1.append([0, 0, 0])
            arm1 = np.array([np.array(p) for p in arm1])
            # Interseção dos dados no extremo e dados XZ, indices
            inter = np.intersect1d(xz[:, 0], arm1[:,0], return_indices=True)[1]
            # Para manter a dimensão dos arrays, mascarar dados com 0
            mask = np.ones(xz.shape, dtype=bool)
            # Mudando valores que identificados para 0
            mask[list(inter)] = 0
            NEW_XZ = xz * mask

            xz = NEW_XZ
            A = np.array([MEAN_ORIG[0], np.max(xz[:, 1:]), 0]) - MEAN_ORIG
            arm2.append(arm1)


    def data_prediction(self, MODEL, DATA_MAP, VEL=10):
        """
            File: auxiliar_functions.py
            Function Name: data_prediction
            Summary: Gravar dados preditos
            Description: Inserir mapeamento e velocidade de entrada para
                         a previsão dos dados.
        """
        scalers = self.return_scaler(load_sc=True)
        VEL_ARR = np.array([[VEL]*DATA_MAP.shape[-2]]).reshape(-1, 1)
        VEL_ARR = scalers['U_in'].transform(VEL_ARR).reshape(1, -1, 1)

        # TODO: Verificar a dimensão dos dados de predição
        #       Devido à futura inserção de novas variáveis
        PREDICs = MODEL.predict({'XZ_input': DATA_MAP, 'U_entr': VEL_ARR})

        # Retornando os dados para a escala anterior
        U_xyz = [scalers['Ux_scaler'].inverse_transform(PREDICs[..., 0]).reshape(-1),
                 scalers['Uy_scaler'].inverse_transform(PREDICs[..., 1]).reshape(-1),
                 scalers['Uz_scaler'].inverse_transform(PREDICs[..., 2]).reshape(-1)]

        # Transformando em DataFrames do panda para facilitar manipulação
        U_xyz = [DataFrame(dt, columns=f'U:{i}') for i, dt in enumerate(U_xyz)]

        # Dado Original de Velocidade
        ORIGIN_DATA = read_csv(self.data_folder+'SLICE_DATA_U_10_0.csv')

        # Coordenadas
        XYZ = ORIGIN_DATA[['Points:0', 'Points:1', 'Points:2']]

        # Geração de arquivo .CSV para leitura
        FILENAME = f'NEW_SLICE_10_Isolated.csv'

        SLICE_DATA = concat(U_xyz + XYZ, sort=True, axis=1)

        # Escrevendo o header no formato do paraview
        with open(self.scaler_dir+FILENAME, 'w') as filename:
            HEADER = ''
            for col in list(SLICE_DATA.columns):
                HEADER += '\"' + col + '\",'
            filename.write(HEADER[:-1])
            filename.write('\n')

        SLICE_DATA.to_csv(self.scaler_dir + FILENAME, index=False, header=False, mode='a')
        print("Dados de previsão copiados!")

        # Diferença do valor previsto e o caso original
        print("Calculando diferença...")

        DIFF = SLICE_DATA[['U:0', 'U:1', 'U:2']] - ORIGIN_DATA[['U:0', 'U:1', 'U:2']]

        RESULT_DATA = concat([DIFF, XYZ], axis=1)

        print('Escrevendo dados DIFERENÇA')
        RESULT_DATA.to_csv(self.scaler_dir + 'DIFF_SLICE_U_10.csv', index=False)
        print('Dados de diferença copiados!')

        return None




class NeuralTopology:
    """
    Classe para reduzir código, automatizar criação de estrutura de rede
    Atributos:
        --> DISTRIBUITION = 'autoencoder'
        --> INICIAL_LAYER = 64
        --> ACTIVATION = 'tanh'
    """
    DISTRIBUTION = 'autoenconder'
    ACTIVATION = 'tanh'

    def __init__(self, MODEL=Sequential(), lyr_type=Dense, num_lyrs=5,
                 init_lyr=64):
        self.model = MODEL
        self.type = lyr_type
        self.num_lyrs = num_lyrs
        self.layer0 = init_lyr
        if type(self.model) == type(Sequential()):
            self.DISTRIBUTION = 'linear'

    def create_sequential(self, inputs=(1,), outputs=1):
        """
            File: auxiliar_functions.py
            Function Name: create_sequential
            Summary: Criar rede simples
            Description: Criar modelo Sequential simples de acordo com os
                         parametros da classe.
        """

        self.model.add(self.type(self.layer0, input_shape=inputs,
                                 activation=self.ACTIVATION))

        num_neurons = self.layer0
        # Adicionando rede do tipo autoencoder
        if self.DISTRIBUTION == 'autoencoder':
            for layer in range(self.num_lyrs):
                # Reduzindo numeros de neuronios até cerca da metade
                if layer <= self.num_lyrs/2:
                    num_neurons = num_neurons/2 if num_neurons % 2 == 0 else (num_neurons+1)/2
                    num_neurons = int(num_neurons)
                else:
                    num_neurons *= 2
                # Adicionando camadas na rede
                self.model.add(self.type(num_neurons, activation=self.ACTIVATION))

            # Adicionando camada final antes do output, igual a primeira
            self.model.add(self.type(self.layer0, activation=self.ACTIVATION))
        elif self.DISTRIBUTION == 'linear':
            # Adicionando camadas com mesmo número de neuronios
            for layer in range(self.num_lyrs):
                self.model.add(self.type(num_neurons, activation=self.ACTIVATION))
        # Camada do output
        self.model.add(self.type(outputs, activation=self.ACTIVATION))

        return self.model


    def add_net_layers(self, qnt_layer=5, activation=['tanh'], dropout=0.0):
        """
            File: auxiliar_functions.py
            Function Name: add_net_layers
            Summary: Adicionar camadas em modelo
            Description: Para um modelo 'model' adicionar uma quantidade de
                         especificadas de camadas para determinadas
        """

    def set_result(self, filename):
        """
            File: auxiliar_functions.py
            Function Name: set_result
            Summary: Escrever topologia da rede
            Description: Usar a função recursiva para persistir a configuração
                         da rede.
        """
        with open(filename, 'w') as fn:
            rec_function(self.model.get_config(), fn)
        print('Done!')
        return None



def rec_function(dic, logfile):
    """
    Author: ShogunHirei
    Description: Função para iterar recursivamente por dicionário de
                 configuração de rede (model.get_config()) e obter as
                 características principais da topologia
    """
    if type(dic) == dict:
        for p in dic.keys():
            if type(dic[p]) == str:
                string = ''
                if len(re.findall('name', p)) >= 1:
                    if 'units' in dic.keys() or 'activation' in dic.keys():
                        string += ' '.join([str(p),  str(dic[p]),
                                            str(dic['units']),
                                            str(dic['activation'])])
                        logfile.write(string+'\n')
            elif type(dic[p]) == dict:
                rec_function(dic[p], logfile)
            elif type(dic[p]) == list:
                for y in range(len(dic[p])):
                    rec_function(dic[p][y], logfile)
    elif type(dic) == list:
        for p in range(len(dic)):
            rec_function(dic[p], logfile)


# Função loss Customizada para magnitude
def mag_diff_loss(y_pred, y_true):
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


def zero_wall_mag(y_pred, y_true, wall_val):
    """
        File: mag_isolated_prediction.py
        Function Name: mag_loss
        Summary: Função de custo para rede neural
        Description: Mudando o valor de y_pred para a condição de velocidade
                     nula na parede, e adicionando a diferença entre
                     as magnitudes da rede e experimentais.

        new_mag = functools.partial(zero_wall_mag, wall_val=U_arr, xz_dict=XZ')
        U_arr --> Array com os dados de velocidade na parede
        XZ --> dados mapeados com os pontos cartesianos
                (considerado plano cilíndrico, uma amostra)
    """
    #    # Magnitude dos valores reais
    M_t = K.sqrt(K.sum(K.square(y_true), axis=-1))
    # Magnitude dos valores previstos
    M_p = K.sqrt(K.sum(K.square(y_pred), axis=-1))

    # Mudar pontos da parede em Booleano, para transformar em binários
    # inverter o ponto binários, para que os pontos na parede seja iguais a 0
    # e os outros pontos iguais a 1, para multiplicar pelo tensor de vel
    wall_val = tf.cast(wall_val, dtype=tf.bool)
    wall_val = tf.expand_dims((tf.cast(wall_val, dtype=tf.float32) - 1) * -1,
                              axis=0)  # shape (1, 862, 3)
    tmp_tens = y_pred * wall_val
    # Tecnicamente 'tmp_tens'
    print(tmp_tens.get_shape())
    PNTY = K.mean(K.square(y_pred - tmp_tens), axis=-1)

    return K.mean(K.square(y_pred - y_true), axis=-1) + K.abs(M_p - M_t) + PNTY


# Classe para escrever dados de maneira organizada
class Writer:
    '''
    Classe para escrever dados de maneira organizada
    '''
    import os

    def __init__(self, DATA_ARRAY, Case, Props, Dir):
        self.data = DATA_ARRAY
        self.dim = self.data.shape
        self.name = Case
        self.props = Props
        self.folder = Dir
        self.samples = self.dim[0]

    def mk_path(self, dim, name_pattern, folder_name='', ind=''):
        '''
        Criação de pastas de forma recursiva
        '''
        if len(dim[:-2]) == 1:
            path = folder_name + f"{name_pattern[0]}/"
            self.os.makedirs(path)
            for folder_num in range(dim[:-2][0]):
                index = [int(p) for p in str(ind+f"{folder_num}").split(',')]
                index = tuple(index)
                name = self.props + "_" + str(folder_num)
                self.record(index, path, name)
        else:
            folder_name += f"{name_pattern[0]}_"
            for num in range(dim[:-2][0]):
                folder_name += f"{num+1}/"
                ind += f"{num},"
                new_dim = dim[1:]
                new_names = name_pattern[1:]
                self.mk_path(new_dim, new_names, folder_name, ind)
                folder_name = folder_name[:-2]
                ind = ind[:-2]

    def record(self, ind, DIR, NAME):
        "Function to record the values in array"
        with open(f'{DIR}/{NAME}', 'w') as datafile:
            for dataline in self.data[ind]:
                datafile.write(str(tuple(dataline)))
        # Inserir local e indices do array
