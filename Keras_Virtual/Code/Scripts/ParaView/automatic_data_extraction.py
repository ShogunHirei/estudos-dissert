"""
File: automatic_data_extraction.py
Author: ShogunHirei
Description: Generate data for comparison in ParaView
"""

import argparse
import sys
# Seting up parser of arguments
parser = argparse.ArgumentParser()

# Case folder 
parser.add_argument("samples_path", help="The relative folder path", )

# Model Path
parser.add_argument('model_path', help="The path for the model to evaluate", )

# Scaler Folder
parser.add_argument('-sf', '--scaler_folder', help="""Scaler path to load scalers of  
                    variables, defaults to the `model_path` folder, if fails 
                    changes to current working folder""", default=False) 

# Base fold
parser.add_argument("-bf", "--base_fold", help="""The folder where will be stored
                    the information, defaults to `model_path` base folder""")

# Output File to store runtime
parser.add_argument("-of", "--outfile", help="""The file path that will 
                    store runtime information""")

# Number of SAMPLES to produce output
parser.add_argument("-s", "--samples", type=int, help="""Number of Samples to 
                    Show Results""", default=3)

# Variables to write
parser.add_argument('--vars',  nargs='+', default=['U'], help="""The Variables 
                    which will be added to the generate .CSV and will have the 
                    OF variable file generated""")

# If unidimensional Reynods number
parser.add_argument('-R1', '--Reynolds', action='store_false', 
                    help='''Multidimensional Reynolds input''')

# To enable verbose from Tensorflow warnings
parser.add_argument('-q', '--quiet', action='store_true', help='''Supress TF 
                    warnings''')


# If the inlet tangential velocity must be specified
parser.add_argument('-iv', '--inlet_velocity', nargs='+', type=float, default=False,
                    help='''If specified, use the values as tangential in script.
                         Can receive one or more values. implies the limiting 
                         of --samples.''')

parser.add_argument('-f', '--factor', help= ''' The factor that will multiply
                    the inlet velocity to obtain the Reynolds number''', default=5283.80102)

# Just to test script operation
parser.add_argument('--dryrun', action='store_true', help='''Do not write 
                    or make any process. Just for debug purposes''')

args = parser.parse_args()

# Python packages imports
import os
import re
import numpy as np
import pandas as pd

# Tensorflow and Keras
from tensorflow.keras.models import load_model

# Adding parent folder to PATH, to work properly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from auxiliar_functions import TrainingData, tf_less_verbose

# DEFINING INPUTS 
CASE = args.samples_path
MODEL_PATH = args.model_path
SAMPLES = args.samples
VARs = args.vars
Re_un = args.Reynolds
OUTPUT_FILE = args.outfile
FORCED_INLET = args.inlet_velocity
dry_run = args.dryrun
FACTOR = float(args.factor)

if bool(FORCED_INLET):
    SAMPLES = len(FORCED_INLET) 
    print('There will be determined ', SAMPLES, 'results')

if args.base_fold:
    BASE_FOLD = args.base_fold
else:
    BASE_FOLD = os.path.dirname(MODEL_PATH) + '/'


# Scaler folder path
if not bool(args.scaler_folder):

    if os.path.exists(BASE_FOLD + 'Scalers/'):
        SCALER_FOLDER = BASE_FOLD + 'Scalers/'

    # Scalers in model path are preferred 
    if os.path.exists(os.path.dirname(MODEL_PATH) + '/Scalers/'):
        # print('Here')
        SCALER_FOLDER = os.path.dirname(MODEL_PATH) + '/Scalers/'

    else:
        SCALER_FOLDER = './Scalers/'

else:
    SCALER_FOLDER = args.scaler_folder + '/'


print(f'Variable projected {VARs} and 1-D Re: {Re_un}')

# Main function to call
def main(CASE, MODEL_PATH, BASE_FOLD, SCALER_FOLD, SAMPLES = 3,
         Outputs = ['U'], factor=5283.80102, Re_un = False, **kwargs):

    """
    Create OF files for folder of data
    
    """
    print('=='*20, '\nStarting Data Generation!', '++'*20, sep='\n')
    print('SCALER_FOLDER is', SCALER_FOLD)
    # Create instance of training with especified data
    TEST_CASE = TrainingData(CASE, FACTOR=factor)

    # Scaler used to normalize data
    TEST_CASE.scaler_folder = SCALER_FOLD

    # Pass the values to function
    INLETS = kwargs.get('INLETS', False)
    if INLETS:
        inlet_idx = 0

    # Check if argument is used
    OUTFILE = kwargs.get('dumpfile', False)

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
    X, _ = TEST_CASE.data_gen(out_labels=Outputs, 
                              EVAL=True, BATCH=True, SAMPLES=SAMPLES)


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
        INLET_SHAPE = DATA_VAL[:, :, INLT_IDX].shape

        # Get the inlet sample value
        if not bool(INLETS):
            INLET_VAL = INLET_SCLR.inverse_transform(DATA_VAL[:, :, INLT_IDX])[0, 0]
            # Obtaining the velocity         
            INLET_VAL = round(INLET_VAL/factor, 4)
        else:
            INLET_VAL = INLETS[inlet_idx]
        

        print("Creating Data Output For array of: ", INLET_VAL)

        # Create the input dictionary to data
        DX = TEST_CASE.training_dict(DATA_VAL, 0)

        # Check for dimension of Inlet
        if Re_un:
            if not bool(INLETS):
                # If the inlet it wasn't specified by FORCED_INLET
                # use the loaded before
                DX['Inlet_U'] = DX['Inlet_U'][:, 0, :]
            else:
                DX['Inlet_U'][...] = INLETS[inlet_idx] * FACTOR

                SHAPE = DX['Inlet_U'].shape

                print(DX['Inlet_U'][:, 0, :], 'and shape ->', SHAPE)

                # If inlets are enabled the data is being generated
                # So it is needed to obtain the Reynolds Number to 
                # the dict which will be used as input

                DX['Inlet_U'] = INLET_SCLR.transform(DX['Inlet_U'][:, :, 0]).reshape(SHAPE)
                DX['Inlet_U'] = DX['Inlet_U'][:, 0, :]
                # DX['Inlet_U'] = np.array(INLETS[0]).reshape(SHAPE)

            print('Loaded unidimensional Re for input')

        print(DX['Inlet_U'], '<---Inlet_u Reyn')

        # Creating folder for each sample
        DIRPATH = BASE_FOLD + f'SAMPLE_{INLET_VAL}/'
        if not os.path.exists(DIRPATH):
            os.mkdir(DIRPATH)


        # Input data and obtain results
        FILEPATH = DIRPATH+f'ANN_{INLET_VAL}_variables.csv'
        TEST_CASE.predict_data_generator(model, DX, FILEPATH,
                                         inlet=INLET_VAL, outfile=OUTFILE,
                                         VARNAME = str(Outputs))


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
            TEST_CASE.U_for_OpenFOAM(RR[VARS], FILENAME, VECTOR=VECTOR_KEY, )

        if bool(INLETS):
            inlet_idx += 1


    print('=='*20, '\nFinished!')
    return None


if __name__ == '__main__':
    # Define Tensorflow verbose level
    if args.quiet:
        tf_less_verbose()

    if dry_run:
        print('Dry run! Check variables')
        print(f'''CASE: {CASE}\nMODEL_PATH: {MODEL_PATH}\nBASE_FOLD: {BASE_FOLD}
                  \rSCALER_FOLDER: {SCALER_FOLDER}\nSAMPLES: {SAMPLES}
                  \rVARIABLES: {VARs}\nReynold1D: {Re_un}\nOUTPUT_FILE: {OUTPUT_FILE}
                  \rINLETS: {FORCED_INLET}\nFACTOR: {FACTOR}''') 
    else:
        # do main function
        main(CASE, MODEL_PATH, BASE_FOLD, SCALER_FOLDER, SAMPLES=SAMPLES,
             Outputs=VARs, Re_un=Re_un, factor=FACTOR, dumpfile=OUTPUT_FILE, INLETS=FORCED_INLET)
