from keras.utils import Sequence
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

from get_data_info import get_field_as_list

# https://keras.io/preprocessing/image/    if more augmentation is needed

# https://keras.io/utils/#sequence
class car_data_generator(Sequence):
    """
    A sequence to feed the model with data. It can be use with training, validation and test data.
    """
    def __init__(self, data_info, batch_size):
        self.data_info = shuffle(data_info)
        self.batch_size = batch_size

    def __len__(self):
        # each element of the generator is a batch
        return int(np.ceil(len(self.data_info) / float(self.batch_size)))

    def __getitem__(self, idx):
        metadata = self.data_info[idx * self.batch_size:(idx + 1) * self.batch_size]

        # random augmentation
        # 0 - central normal
        # 1 - central flipped
        # 2 - left
        # 3 - left flipped
        # 4 - right
        # 5 - right flipped
        rand_choice = np.random.random_integers(0, 5, len(metadata))
        # the early choice of the process for the data is needed because we need to do the same for x and y
        x = np.array([self.augment_image(metadata[i], rand_choice[i]) for i in range(len(metadata))])
        y = np.array([self.augment_label(metadata[i], rand_choice[i]) for i in range(len(metadata))])
        return x, y

    def on_epoch_end(self):
        """
        This methood is called at the end of each epoch
        """
        self.data_info = shuffle(self.data_info)

    @staticmethod
    def augment_image(metadata, num):
        # random augmentation
        # 0 - central normal
        # 1 - central flipped
        # 2 - left
        # 3 - left flipped
        # 4 - right
        # 5 - right flipped
        # select image
        if num < 2:
            # images are read in BGR scheme but drive.py uses RGB.
            img = cv2.cvtColor(cv2.imread(metadata['img_center']), cv2.COLOR_BGR2RGB)
        elif num < 4:
            img = cv2.cvtColor(cv2.imread(metadata['img_left']), cv2.COLOR_BGR2RGB)
        elif num < 6:
            img = cv2.cvtColor(cv2.imread(metadata['img_right']), cv2.COLOR_BGR2RGB)
        # flip
        if num % 2:
            img = np.fliplr(img)
        return img

    @staticmethod
    def augment_label(metadata, num):
        # random augmentation
        # 0 - central normal
        # 1 - central flipped
        # 2 - left
        # 3 - left flipped
        # 4 - right
        # 5 - right flipped

        angle_correction = 0.2
        angle = metadata['steering_angle']
        if num < 2:
            pass
        elif num < 4:
            angle += angle_correction
        elif num < 6:
            angle -= angle_correction
        # flip
        if num % 2:
            angle = -angle
        return angle


def get_car_data_generators(data, batch_size):
    """
    Returns 2 generators, one for training and another for validation
    """
    train_data, validation_data = train_test_split(data, test_size=0.2)
    return car_data_generator(train_data, batch_size), car_data_generator(validation_data, batch_size)


def get_car_data_generators_with_test(data, batch_size):
    """
    Returns 3 generators, for training, validation and testing
    """
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
