import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3

from keras.models import Sequential, Model
from keras.layers import Input, Cropping2D, Dense, Flatten, Lambda, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate


def expand_InceptionV3_01():
    print('Model - Inception')
    ### Sequential model to preprocess the images before the model
    input_layer = Input(shape=(160,320,3))
    # set up lambda layer
    # input images are 160x320x3
    # set up cropping2D layer 50 rows from the top and 20 rows from the bottom
    crop = Cropping2D(cropping=((50,20), (0,0)))(input_layer)
    # normalize
    norm = Lambda(lambda x: (x / 255.0) - 0.5)(crop)
    # here the image is 90x320x3 and normalized [-0.5, 0.5]

    # im going to use InceptionV3 as the core of my model
    # the input shape is this because im going to crop the images  input_shape=(90,320,3)
    inp_input = Input(shape=(90,320,3))
    inception = InceptionV3(input_tensor=norm, weights='imagenet', include_top=False)
    # im going to train all the layers
    for layer in inception.layers:
        layer.trainable = True

    # put the model in the middle
    output_inp = inception.get_layer(index=-1).output

    # GlobalAveragePooling2D
    gavpoll = GlobalAveragePooling2D()(output_inp)
    # Fully connected
    dense = Dense(200)(gavpoll)
    # output
    output_layer = Dense(1)(dense)

    # generate the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def nvidia():
    """
    https://arxiv.org/pdf/1604.07316v1.pdf
    """
    print('Model - nvidia')
    inp = Input(shape=(160,320,3))
    
    crop = Cropping2D(cropping=((50,20), (0,0)))(inp)
    # convert to yuv colour scheme
    #def hsv_conversion(x):
        #return tf.image.rgb_to_yuv(x)

    #yuv = Lambda(hsv_conversion)(crop)
    # normalize
    norm = Lambda(lambda x: (x / 127.5) - 1.0)(crop)

    conv1 = Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid")(norm)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid")(conv1)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="valid")(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid")(conv3)
    conv5 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid")(conv4)
    flat = Flatten()(conv5)
    fully1 = Dense(1164, activation="relu")(flat)
    drop1 = Dropout(0.5)(fully1)
    fully2 = Dense(100, activation="relu")(drop1)
    drop2 = Dropout(0.5)(fully2)
    fully3 = Dense(50, activation="relu")(drop2)
    drop3 = Dropout(0.5)(fully3)
    fully4 = Dense(10, activation="relu")(drop3)
    drop4 = Dropout(0.5)(fully4)
    outp = Dense(1)(drop4)

    model = Model(inputs=inp, outputs=outp)
    return model


def double_nvidia():
    print('Model - double nvidia')
    inp = Input(shape=(160,320,3), name='input')
    # crop the image, im cropping less because im using images from the track 2 with slopes
    crop = Cropping2D(cropping=((30,20), (0,0)), name='crop')(inp)
    # convert to yuv colour scheme
    def hsv_conversion(x):
        return tf.image.rgb_to_hsv(x)

    yuv = Lambda(hsv_conversion, name='yuv')(crop)
    # normalize
    norm = Lambda(lambda x: (x / 255.0/2) - 1.0, name='normalize')(yuv)

    x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", padding="valid")(norm)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", padding="valid")(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", padding="valid")(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="elu", padding="valid")(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="elu", padding="valid")(x)
    x = Flatten()(x)
    x = Dense(1164, activation="elu")(x)
    x = Dense(100, activation="elu")(x)
    x = Dense(50, activation="elu")(x)
    x = Dense(10, activation="elu")(x)

    z = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", padding="valid")(norm)
    z = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", padding="valid")(z)
    z = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", padding="valid")(z)
    z = Conv2D(64, (3, 3), strides=(1, 1), activation="elu", padding="valid")(z)
    z = Conv2D(64, (3, 3), strides=(1, 1), activation="elu", padding="valid")(z)
    z = Flatten()(z)
    z = Dense(1164, activation="elu")(z)
    z = Dense(100, activation="elu")(z)
    z = Dense(50, activation="elu")(z)
    z = Dense(10, activation="elu")(z)

    con = Concatenate()([x, z])

    out = Dense(1)(con)

    model = Model(inputs=inp, outputs=out)
    return model


def dummy_model():
    print('Model - dummy')
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(100))
    model.add(Dense(1))
    return model


def get_model(name='nvidia'):
    if name == 'double_nvidia':
        return double_nvidia()
    elif name == 'nvidia':
        return nvidia()
    elif name == 'inception':
        return expand_InceptionV3_01()
    elif name == 'dummy':
        return dummy_model()
    raise ValueError('Model name not found.')


if __name__ == '__main__':
    model = get_model()
    model.summary()
