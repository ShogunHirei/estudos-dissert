"""
File: hypTunningTest.py
Author: ShogunHirei
Description: Hyperparameters tunning test with the keras-tuner package
"""

# Python Builtins
import sys, os
import numpy

# Packaeg imports
import kerastuner as kt
from tensrflow.keras.layers import Dense
from tensorflow_docs import EpochDots

# Add top-level packaeg to PATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import auxiliar_functions

