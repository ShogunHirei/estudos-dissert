from os import path

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
    from . import auxiliar_functions

else:
    import auxiliar_functions

