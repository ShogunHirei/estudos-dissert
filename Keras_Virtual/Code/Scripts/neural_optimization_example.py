from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import TensorBoard
from datetime import datetime
import os


def data_generator():
    data = np.random.random(size=(139, 2))
    x_train, x_test, y_train, y_test = train_test_split(
        data[:, 0], data[:, 1], test_size=0.2)
    return x_train, y_train, x_test, y_test


def function_model(x_train, y_train, x_test, y_test):
    '''Docstring
    '''
    model = Sequential()
    model.add(Dense({{choice([5, 10, 25])}},
                    activation={{choice(['sigmoid', 'relu'])}}, input_shape=(1,)))
    model.add(Dense({{choice([5, 10, 25])}},
                    activation={{choice(['sigmoid', 'relu'])}}))

    if {{choice(['three', 'four', 'five'])}} == 'four':
        model.add(Dense({{choice([5, 10, 15])}},
                        activation={{choice(['sigmoid', 'relu'])}}))
    if {{choice(['four', 'five'])}} == 'five':
        model.add(Dense({{choice([5, 10, 15])}},
        activation={{choice(['sigmoid', 'relu'])}}))

    model.add(Dense(1,
                    activation={{choice(['sigmoid', 'relu'])}}))

    model.compile(loss={{choice(['mse', 'mae'])}},
                  optimizer='rmsprop', metrics=['accuracy'])

    FOLDER = './test_dir/'
    NOW = datetime.now()
    LOGDIR = FOLDER + NOW.strftime("%Y%m%d-%H%M%S") + "/"
    os.mkdir(LOGDIR)

    TB = TensorBoard(log_dir=LOGDIR)
    result = model.fit(x_train, y_train,
                       batch_size={{choice([10, 20])}},
                       validation_data=(x_test, y_test),
                       epochs={{choice([30, 75, 100])}},
                       verbose=0,
                       callbacks=[TB])
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=function_model,
                                          data=data_generator,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials(),
                                          eval_space=True)
    X_train, Y_train, X_test, Y_test = data_generator()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
