"""
File: automatic_data_extraction.py
Author: ShogunHirei
Description: Generate data for comparison in ParaView
"""

# Python packages imports
import os, re, sys, argparse
import numpy as np
import pandas as pd

# Tensorflow and Keras
from tensorflow.keras.models import load_model

# Adding parent folder to PATH, to work properly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from auxiliar_functions import TrainingData, tf_less_verbose

# Seting up parser of arguments
parser = argparse.ArgumentParser()

# Case folder 
parser.add_argument("samples_path", help="The relative folder path", )

# Model Path
parser.add_argument('model_path', help="The path for the model to evaluate", )

# Scaler Folder
parser.add_argument('-sf', '--scaler_folder', help="""Scaler path to load scalers of  
                    variables, defaults to the `model_path` folder, if fails 
                    changes to current working folder""") 

# Base fold
parser.add_argument("-bf", "--base_fold", help="""The folder where will be stored
                    the information, defaults to `model_path` base folder""")

# Number of SAMPLES to produce output
parser.add_argument("-s", "--samples", type=int, help="""Number of Samples to 
                    Show Results""", default=3)

# Variables to write
parser.add_argument('--vars',  nargs='+', default=['U'], help="""The Variables 
                    which will be added to the generate .CSV and will have the 
                    OF variable file generated""")

# If unidimensional Reynods number
parser.add_argument('-R1', '--Reynolds', action='store_false', 
                    help='''Unidimensional Reynods''')

# To enable verbose from Tensorflow warnings
parser.add_argument('-q', '--quiet', action='store_true', help='''Supress TF 
                    warnings''')

args = parser.parse_args()

# DEFINING INPUTS 
CASE = args.samples_path
MODEL_PATH = args.model_path
SAMPLES = args.samples
VARs = args.vars
Re_un = args.Reynolds

if args.base_fold:
    BASE_FOLD = args.base_fold
else:
    BASE_FOLD = os.path.dirname(MODEL_PATH) + '/'

if args.scaler_folder:
    SCALER_FOLDER = args.scaler_folder
elif os.path.exists(BASE_FOLD + 'Scalers/'):
    SCALER_FOLDER = BASE_FOLD + 'Scalers/'
else:
    SCALER_FOLDER = './Scalers/'

print(f'Variable projected {VARs} and 1-D Re: {Re_un}')

# Main function to call
def main(CASE, MODEL_PATH, BASE_FOLD, SCALER_FOLD, SAMPLES = 3,
         Outputs = ['U'], factor=5283.80102, Re_un = False):

    """
    Create OF files for folder of data
    
    """
    print('=='*20, '\nStarting Data Generation!', '++'*20, sep='\n')
    # Create instance of training with especified data
    TEST_CASE = TrainingData(CASE, FACTOR=factor)

    # Scaler used to normalize data
    TEST_CASE.scaler_folder = SCALER_FOLD


    # Try to load the custom objects of a model
    try:
        model = load_model(MODEL_PATH)
    except:
        # Add some way to integrate the results generation with custom objects
        print('Custom object functions must be declared!')
        CUSTOM_OBJ = {}
        model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJ )


    # Model compiled and loaded
    # Data generation (just input matters for comparison)
    X, _ = TEST_CASE.data_gen(out_labels=Outputs, EVAL=True)


    # Comparison between names in CASES and values in DX[Inlet]
    # To ensure the index of the written data
    
    # Load the scaler to compare data in original scale
    INLET_SCLR  = TEST_CASE.return_scaler()['Inlet_U']

    # Generate a dictionary for each of the samples in CASE folder 
    # Get "Inlet_U" idex in the whole data 
    INLT_IDX= TEST_CASE.ORDER[0]['Inlet_U'][0]

    for DATA_VAL in X:
        # To add the axe removed by the for-loop
        DATA_VAL = DATA_VAL[np.newaxis, ...]

        # Get the inlet sample value
        INLET_VAL = INLET_SCLR.inverse_transform(DATA_VAL[:, :, INLT_IDX])[0, 0]
        # Obtaining the velocity         
        INLET_VAL = round(INLET_VAL/factor, 4)

        print("Creating Data Output For array of: ", INLET_VAL)
        
        # Create the input dictionary to data
        DX = TEST_CASE.training_dict(DATA_VAL, 0)


        # Check for dimension of Inlet
        if Re_un:
            DX['Inlet_U'] = DX['Inlet_U'][:, 0, :]
            print('Loaded unidimensional Re for input')


        # Creating folder for each sample
        DIRPATH = BASE_FOLD + f'SAMPLE_{INLET_VAL}/'
        if not os.path.exists(DIRPATH):
            os.mkdir(DIRPATH)


        # Input data and obtain results
        FILEPATH = DIRPATH+f'ANN_{INLET_VAL}_variables.csv'
        TEST_CASE.predict_data_generator(model, DX, FILEPATH)


        # Read results with pandas and format data in OpenFOAM variable layout
        RR = pd.read_csv(FILEPATH)

        for var_name in Outputs:
            # Check for variables that can be vectors
            VARS = []
            for name in list(RR.columns):
                if re.findall(f'^{var_name}', name):
                    VARS.append(str(name))

        
            # Check if the formatting method is to scalar or vector variables
            print('The variables added are: ', *VARS)
            if len(VARS) > 1:
                VECTOR_KEY = True
            elif len(VARS) == 1:
                VECTOR_KEY = False
            else:
                print('Zero Varibles in Outputs!\nCheck Code and Inputs!')
                break


            # Formatting...
            print("Writing OF variable file for: ", var_name)
            FILENAME = DIRPATH + f'ANN_{var_name}'
            TEST_CASE.U_for_OpenFOAM(RR[VARS], FILENAME, VECTOR=VECTOR_KEY)

        if SAMPLES < 0:
            print("Limit of required samples achieved")
            break
        else:
            SAMPLES -= 1

    print('=='*20, '\nFinished!')
    return None


if __name__ == '__main__':
    # Define Tensorflow verbose level
    if args.quiet:
        tf_less_verbose()

    # do main function
    main(CASE, MODEL_PATH, BASE_FOLD, SCALER_FOLDER, SAMPLES=SAMPLES,
         Outputs=VARs, Re_un=Re_un)
