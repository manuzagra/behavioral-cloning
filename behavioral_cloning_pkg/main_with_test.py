from keras.models import load_model

from utils import save_data, load_data

from get_data_info import get_data_info
from data_generator import get_car_data_generators_with_test
from model_arch import get_model
from train_model import train
from test_model import test

if __name__ == '__main__':
    
    program = {'train': True,
               'test': True,
               'load_model': True,
               'model_path': './saved_models/model.h5',
               'load_data': True,
               'data_path': './data.p'}
               
    
    params = {'model_name': 'model',
              'batch_size': 200,
              'epochs': 1,
              'loss': 'mse',
              'optimizer': 'adam',
              'validation_steps': 1,
              'verbose': 0}
    
    if not program.get('load_data'):
        print('Getting data info...')
        data_info = get_data_info('/opt/sim_data')
        print('Creating generators...')
        train_data_gen, valid_data_gen, test_data_gen = get_car_data_generators_with_test(data_info, params['batch_size'])
        print('Saving data to ' + program['data_path'])
        save_data({'data_info':data_info,
                   'train_data_gen':train_data_gen,
                   'valid_data_gen':valid_data_gen,
                   'test_data_gen':test_data_gen},
                  program['data_path'])
        print('Data saved.')
    else:
        print('Loading data from ' + program['data_path'])
        d = load_data(program['data_path'])
        data_info = d['data_info']
        train_data_gen = d['train_data_gen']
        valid_data_gen = d['valid_data_gen']
        test_data_gen = d['test_data_gen']
        print('Data loaded.')
        
        
    if not program.get('load_model'):
        print('Creating new model...')
        model = get_model()
        print('Model created.')
    else:
        print('Loading model from ' + program.get('model_path'))
        model = load_model(program.get('model_path'))
        print('Model loaded.')
            
    if program.get('train'):
        hist = train(model, train_data_gen, valid_data_gen, **params)
        print('hist')
        print(hist.history)
        
    if program.get('test'):
        loss = test(model, test_data_gen, **params)
        print('loss')
        print(loss)
        
    