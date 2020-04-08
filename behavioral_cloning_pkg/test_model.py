def test(model, test_data_gen, **kwargs):
    """
    Test a model with a data generator
    """
    print('Testing model...')
    print(model.metrics_names)
    loss = model.evaluate_generator(test_data_gen,
                   steps=len(test_data_gen),
                   verbose=kwargs.get('verbose', 1), # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                   max_queue_size=10,
                   workers=1,
                   use_multiprocessing=False)
    return loss
