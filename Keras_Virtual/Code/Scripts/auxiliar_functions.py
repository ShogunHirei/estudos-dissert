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
                    if 'units' in dic.keys():
                        string += str(p) + ' ' + str(dic[p])
                        string += ' ' + str(dic['units'])
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


def zero_wall_mag(y_pred, y_true, wall_val, xz_dict):
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
    # Magnitude dos valores reais
    M_t = K.sqrt(K.sum(K.square(y_true), axis=-1))
    # Magnitude dos valores previstos
    M_p = K.sqrt(K.sum(K.square(y_pred), axis=-1))
    for indx, xz in enumerate(xz_dict):
        if xz_dict[indx] == wall_val[indx]:
            # y_pred[indx] = tf.zeros(3)
            tf.Session().run(tf.assign(y_pred[indx], tf.zeros(3)))
    return K.mean(K.square(y_pred - y_true), axis=-1) + K.abs(M_p - M_t)


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
