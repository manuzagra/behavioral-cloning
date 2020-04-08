import time
import json

def train(model, train_data_gen, valid_data_gen, **kwargs):
    """
    Train and save a model
    """
    print('Training model...')
    # compile the model qith the given optimizer and loss
    model.compile(loss=kwargs.get('loss', 'mse'), optimizer=kwargs.get('optimizer', 'adam'), metrics=['accuracy'])
    # https://keras.io/models/model/#fit_generator
    hist = model.fit_generator(train_data_gen,
                        steps_per_epoch=len(train_data_gen),
                        validation_data=valid_data_gen,
                        validation_steps=len(valid_data_gen),
                        epochs=kwargs.get('epochs', 1),
                        verbose=kwargs.get('verbose', 1), # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                        callbacks=kwargs.get('callbacks', None),
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=False)
    # save the model afer training
    if kwargs.get('model_name'):
        fname = kwargs.get('model_name') + '_' + str(int(time.time())) + '.h5'
    else:
        fname = 'model_' + str(int(time.time())) + '.h5'
    model.save('saved_models/' + fname)
    print('Model saved as: ' + fname)

    # save the training hyper parameters and the metrics
    # it can be usefull to choose the best model and the best hyper parameters
    with open('saved_models/' + fname + '.txt', 'w') as f:
        for key, value in kwargs.items():
            if key is not 'callbacks':
                f.write(key + ' = ' + str(value) + '\n')
        f.write('training_samples = ' + str(2*len(train_data_gen.data_info)) + '\n')
        f.write('validation_samples = ' + str(2*len(valid_data_gen.data_info)) + '\n')
        f.write('\n--------------------------------------------\n')
        for key, value in hist.history.items():
            f.write(key + ' = ' + str(value) + '\n')

    return hist
