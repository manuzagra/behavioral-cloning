from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import save_data, load_data

from get_data_info import get_data_info
from data_generator import get_car_data_generators
from model_arch import get_model
from train_model import train

if __name__ == '__main__':

    # It defines the flow of the program
    program = {'train': True,
               'load_model': False,
               'model_path': './saved_models/nvidia_1586195186.h5',
               'load_data': False,
               'data_path': './data.p'}

    # it defines the training
    params = {'model_name': 'nvidia',
              'batch_size': 250,
              'epochs': 5,
              'loss': 'mse',
              'optimizer': 'adam',
              'verbose': 1}

    params['callbacks'] = [ModelCheckpoint('./saved_models/'+params['model_name']+'_ckeckpoint.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=False,
                                           save_weights_only=False,
                                           mode='auto',
                                           period=1),
                          EarlyStopping(monitor='val_loss',
                                        min_delta=0.0005,
                                        patience=2,
                                        verbose=1,
                                        mode='auto',
                                        baseline=None,
                                        restore_best_weights=False)]

    # load the metadata of get it readong the .csv files
    if not program.get('load_data'):
        print('Getting data info...')
        data_info = get_data_info('/opt/sim_data')
        print('Saving data to ' + program['data_path'])
        save_data({'data_info':data_info}, program['data_path'])
        print('Data saved.')
    else:
        print('Loading data from ' + program['data_path'])
        d = load_data(program['data_path'])
        data_info = d['data_info']
        del d
        print('Data loaded.')

    # create the generators for the data
    print('Creating generators...')
    train_data_gen, valid_data_gen = get_car_data_generators(data_info, params['batch_size'])
    print(len(train_data_gen), ' training batches of ', params['batch_size'])
    print(len(valid_data_gen), ' validation batches of ', params['batch_size'])

    # load the model or create it from scratch
    if not program.get('load_model'):
        print('Creating new model...')
        model = get_model(params['model_name'])
        print('Model created.')
    else:
        print('Loading model from ' + program.get('model_path'))
        model = load_model(program.get('model_path'))
        print('Model loaded.')

    # train the model
    if program.get('train'):
        hist = train(model, train_data_gen, valid_data_gen, **params)
        print('hist')
        print(hist.history)
