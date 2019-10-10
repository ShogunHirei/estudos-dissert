"""
File: ciclone_ANN.py
Author: ShogunHirei
Description: Uso de redes neurais para predizer componentes de velocidade em 
             valor fixo de y (Slice em -0.2).
"""

import numpy as np
import modred as md
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from datetime import datetime
















































