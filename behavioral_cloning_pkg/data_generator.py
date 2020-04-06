from keras.utils import Sequence
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

from get_data_info import get_field_as_list

# https://keras.io/preprocessing/image/    if more augmentation is needed

# https://keras.io/utils/#sequence
class car_data_generator(Sequence):
    def __init__(self, data_info, batch_size):
        self.data_info = shuffle(data_info)
        self.augmentation_info = shuffle(data_info)
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(2*len(self.data_info) / float(self.batch_size)))

    def __getitem__(self, idx):
        size = int(self.batch_size/2)
        data = self.data_info[idx * size:(idx + 1) * size]
        augmentation = self.augmentation_info[idx * size:(idx + 1) * size]
        
        # images are read in BGR scheme but drive.py uses RGB.
        x = np.vstack([np.array([cv2.cvtColor(cv2.imread(d['img_center']), cv2.COLOR_BGR2RGB) for d in data]),
                   np.array([self.flip_horizontally(
                       cv2.cvtColor(cv2.imread(d['img_center']), cv2.COLOR_BGR2RGB)) for d in augmentation])])
        y = np.hstack([np.array([d['steering_angle'] for d in data]),
                  np.array([self.flip_horizontally(d)['steering_angle'] for d in augmentation])]).T
        
        # print('len(data) = ', len(data))
        # print('len(augmentation) = ', len(augmentation))
        # print('x.shape = ', x.shape)
        # print('y.shape = ', y.shape)
        return x, y
    
    def on_epoch_end(self):
        self.data_info = shuffle(self.data_info)
        self.augmentation_info = shuffle(self.augmentation_info)
    
    @staticmethod
    def flip_horizontally(x):
        if isinstance(x, np.ndarray):
            x = np.fliplr(x)
        elif isinstance(x, dict):
            x['steering_angle'] = -x['steering_angle']
        return x
        
        
def get_car_data_generators(data, batch_size):
    train_data, validation_data = train_test_split(data, test_size=0.2)
    return car_data_generator(train_data, batch_size), car_data_generator(validation_data, batch_size)
        
        
def get_car_data_generators_with_test(data, batch_size):
    train_data, validation_data = train_test_split(data, test_size=0.2)
    train_data, test_data = train_test_split(train_data, test_size=0.1)
    return car_data_generator(train_data, batch_size), car_data_generator(validation_data, batch_size), car_data_generator(test_data, batch_size)


if __name__ == '__main__':
    from get_data_info import get_data_info
    
    d = get_data_info('/opt/sim_data/')
    train, valid = get_car_data_generators(d, 300)
    print(len(train))
    print(len(valid))
    for i in range(len(valid)):
        print('---------------')
        print('i: ', i)
        epoch = valid[i]
        print('epoch[0].shape: ', epoch[0].shape)
        print('epoch[1].shape: ', epoch[1].shape)
<<<<<<< HEAD
    
=======
    
>>>>>>> c70989b47b0e69db88dba746966f0c887c4b70b8
