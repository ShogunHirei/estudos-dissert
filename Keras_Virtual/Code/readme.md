# Predição de Perfis de escoamento em separadores 
Estudos em progresso do uso de redes neurais para predição de perfis de 
escoamento. 

## Branches 
## 1. cavity 
### 1.1 *Lid-Driven Cavity Flow*

    Uso de *AutoEncoder* para prever o perfil de velocidade.
    Arquivos: keras_cavity_MLP.py -> rede neural MLP genérica para testes iniciais.

## 2. ciclone

### 2.1 ciclone_ANN.py
    O arquivo que será utilizado na otimização dos hiperparametros de 'ciclone_ANN_old'.
    Utilizando Hyperas (HyperOpt) como modulo de otimização.

### 2.2 ciclone_ANN_old.py

    Arquivo com configuração de rede utilizada no processo de treinamento inicial.

### 2.3 isolated_prediction

    Rede neural elaborada com base no script de 'ciclone_ANN_old.py', mas para predizer
    as 3 componentes de velocidade.


## Estrutura de Arquivos
### 1. Models

    Modelos aplicados no problema em cada branch
### 2. Order Reduction

    Algoritmos de redução de ordem, ou seus estudos, para uso no problema, caso necessário
### 3. Generated Files

    Arquivos Gerados, por extração de dados, ou modelos salvos pelo Keras.

### 4. Scripts

    Scripts auxiliares ao problema, extração de dados ou alteração de arquivos envolvidos
