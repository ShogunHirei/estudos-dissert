import sys, os
import argparse

parser = argparse.ArgumentParser()

# Case folder 
parser.add_argument("samples_path", help="The relative folder path")


# model Path
parser.add_argument('model_path', help="The path for the model to evaluate")

# Base fold
parser.add_argument("-bf", "--base_fold", help="The folder where will be stored\
                    the information, defaults to `model_path` base folder")


# SAMPLES 
parser.add_argument("-s", "--samples", type=int, help="Number of Samples to Show Results",
                    default=3)
# Variables to write
parser.add_argument('--vars',  nargs='+', default=['U'],
                    help="The Variables wich will be added to the generate .CSV and will have the OF variable file generated")

# If unidimensional Reynods number
parser.add_argument('-R1', '--Reynolds', action='store_false')

args = parser.parse_args()

print(args._get_args())
print('++'*20)

CASE = args.samples_path
MODEL_PATH = args.model_path

if args.base_fold:
    BASE_FOLD = args.base_fold
else:
    BASE_FOLD = os.path.dirname(MODEL_PATH)
SAMPLES = args.samples
VARs = args.vars
RE_un = args.Reynolds

print(CASE, MODEL_PATH, BASE_FOLD, SAMPLES, VARs, RE_un)


