# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
"""
 python3 train_mask_detector.py --dataset dataset
"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

LR = 1e-4
EPOCHS = 20
BS = 32


def load_data():
    image_paths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]

        image = load_img(image_path, target_size=(224, 244))
        image = img_to_array(image)
        image = preprocess_input(image) # scales the pixel intensities in the image to the range [-1,1]

        data.append(image)
        labels.append(label)
    # convert the data and labels to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    #split data
    (train_X, test_X, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state = 42)

    # Generate batches of tensor image data with real-time data augmentation
    # apply on-the-flu mutations to our images in an effort to improve generalization
    # random rotation, zoom, shear, shift....
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    return train_X, test_X, train_y, test_y, aug, lb


def create_model():

    # Load MobileNetV2 network
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
	    input_tensor=Input(shape=(224, 224, 3))) # include_top = False => leaves off the head of network

    # construct the head that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        """
            The weights of these base layes will not be updated during the process
            of thse backpropagation, whereas the head layer weights will be tuned
        """
        layer.trainable = False
    return model


def train_model():
    train_X, test_X, train_y, test_y, aug, lb = load_data()
    model = create_model()

    opt = Adam(lr=LR, decay=LR/EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) # binary_crossentropy => 2 classes

    # train the head of the network
    H = model.fit(
        aug.flow(train_X, train_y, batch_size=BS),
        steps_per_epoch=len(train_X) // BS,
        validation_data=(test_X, test_y),
        validation_steps=len(test_X) // BS,
        epochs=EPOCHS
    )

    # MAKE TEST PREDICTIONS

    predIdxs = model.predict(test_X, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(test_y.argmax(axis=1), predIdxs,
        target_names=lb.classes_))

    # serialize the model to disk
    model.save(args["model"], save_format="h5")

    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


if __name__ == "__main__":
    train_model()