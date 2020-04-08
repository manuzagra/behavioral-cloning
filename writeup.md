# **Behavioral Cloning**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

##### Files

My project includes a python package, "Behavioral_cloning_pkg", with the following files:
* **model.py** is the main script that goes through all the steps
* **utils.py** a couple of useful functions
* **get_data_info.py** gets the path to all the images and the labels
* **data_generator.py** contains the data generators needed to train, evaluate and test the model
* **model_arch.py** create the model, there are some models architectures here, one of them is the chosen at the end
* **train_model.py** contains the function to train the model
* **test_model.py** contains the function to test the model, it is a fast alternative to the simulation.
* **drive.py** for driving the car in autonomous mode
* **video.py** it creates a video from frames
* **saved_models/nvidia_complete.h5** containing a trained convolution neural network

And some extra files:
* **get_data.sh** to download the data and extract it
* **video.mp4** video showing the results
* **writeup.md** summarizing the results

##### Usage
To download the data from google drive and unzip it in /opt/sim_data
```sh
./get_data.sh
```

Now you can train the model running:
```sh
python model.py
```
Inside model.py you can find the training parameters. The model will be saved in the folder "saved_models".


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:
```sh
python drive.py saved_models/nvidia_complete.h5
```

---
### Model Architecture and Training Strategy

##### Model architecture
The model is copied from https://arxiv.org/pdf/1604.07316v1.pdf. The first 2 layers are preprocessing, cropping the image to discard sky and car, and normalizing the pixels. It contains 5 convolutional layers followed by 5 fully connected layers, the final one is the output. I am using relu as activation in all the layers and dropout between the fully connected layers.

```python
def nvidia():
    """
    https://arxiv.org/pdf/1604.07316v1.pdf
    """
    print('Model - nvidia')
    inp = Input(shape=(160,320,3))

    crop = Cropping2D(cropping=((50,20), (0,0)))(inp)

    norm = Lambda(lambda x: (x / 127.5) - 1.0)(crop)

    conv1 = Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid")(norm)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid")(conv1)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="valid")(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), activatUsageion="relu", padding="valid")(conv3)
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
```

##### Training parameters
The chosen optimizer was "adam" with the default parameters and the loss function was "mse". The training was performed for 5 epochs with a batch size of 250 samples.

##### Training data
The used data was the provided by Udacity and extra data taken from the simulator, I tried to stay centred on the road all the time. The total number of samples is 36712.

##### Strategy
At the beginning I was using just the central images.
The first model I tried was InceptionV3, I tried to train it completely but there was no memory enough. Then I started reading the mentioned papers and the forums and I ended up copying the model from nvidia.
It did not perform well. I tried different training parameters and I did not get good performance, I could see the car did not turn enough.
At the end I tried adding the left and right images adapting the label, I also flipped images and angles. This was a critical step, after I diversify the used data the model performed much better driving around the circuit without problems.

---
### Reflection

There are a lot of possibilities, different architectures, hyper parameters and data. In my case to use left and right images was decisive and changed completely the performance of the model.
