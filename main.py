import os
import wx
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pickle

import definitions
from classifing.image_classifing import Model
from image_processing.image_processing import ImageProcessing
from image_processing.webcam import WebcamFeed


# image = preprocessing.load_image(definitions.ROOT_DIR + '\\resources\\images\\image-1.jpg')
# image_processed = preprocessing.preprocess_image(image, visualize=True)

class App:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.preprocessing = ImageProcessing()
        self.webcam = WebcamFeed()
        self.model = Model()
        """Create a pretrained model"""
        self.model_nn = self.model.create_model()
        self.encoder = LabelEncoder()
        self.classifier = None

    def load_image(self, path):
        return self.preprocessing.load_image(path)

    def capture_image(self):
        return self.webcam.get_image(self.window_width, self.window_height)

    def preprocess_image(self, image):
        return self.preprocessing.preprocess_image(image)

    def load_identities(self):
        return self.model.load_identities(definitions.ROOT_DIR + '\\resources\\images')

    def load_training_data(self, idendities):
        """Load the training data"""
        idendities_images = []
        for index, value in enumerate(idendities):
            image = self.preprocessing.load_image(value.get_image_path())
            image = self.preprocessing.preprocess_image(image)
            idendities_images.append(image)

        """Create the embeded vectors from the training images"""
        embeded_data = self.model.create_embeded_data_from_identities(idendities_images, self.model_nn)
        return embeded_data

    def train(self, identities, embeded_identities, show_accuracy=False):
        classifier_path = definitions.ROOT_DIR + 'resources\\models\\face_classifier.pkl'
        if os.path.exists(classifier_path) is True:
            with open(classifier_path, 'rb') as fid:
                self.classifier = pickle.load(fid)
        else:
            targets = np.array([identity.name for identity in identities])

            self.encoder.fit(targets)

            targets_encoded = self.encoder.transform(targets)
            train_idx = np.arange(identities.shape[0]) % 2 != 0
            test_idx = np.arange(identities.shape[0]) % 2 == 0

            data_train = embeded_identities[train_idx]
            data_test = embeded_identities[test_idx]

            labels_train = targets_encoded[train_idx]
            labels_test = targets_encoded[test_idx]

            self.classifier = LinearSVC()
            self.classifier.fit(data_train, labels_train)

            with open(classifier_path, 'wb') as fid:
                pickle.dump(self.classifier, fid)

            if show_accuracy is True:
                acc_svc = accuracy_score(labels_test, self.classifier.predict(data_test))
                print(f'SVM accuracy = {acc_svc}')

    def predict(self, image):
        embeded_identity = self.model.create_embeded_data_from_identity(image, self.model_nn)
        prediction = self.classifier.predict([embeded_identity])
        identity = self.encoder.inverse_transform(prediction)[0]
        return identity


class AppGUI(wx.Frame):
    def __init__(self, parent, title):
        self.window_width = 600
        self.window_height = 480
        super(AppGUI, self).__init__(parent, title=title,
                                     size=(self.window_width, self.window_height))
        self.init()
        self.initUI()

    def init(self):
        app = App(self.window_width, self.window_height)
        identities = app.load_identities()
        identities_embeded = app.load_training_data(identities)
        app.train(identities=identities, embeded_identities=identities_embeded, show_accuracy=False)

        """
        example_image = app.load_image(
            definitions.ROOT_DIR + '\\resources\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0009.jpg')
        example_image = app.preprocess_image(example_image)
        example_identity = app.predict(example_image)
        print('Example identity:', example_identity)
        """

    def initUI(self):
        self.Centre()
        self.Show()


if __name__ == '__main__':
    app = wx.App()
    AppGUI(None, title='Center')
    app.MainLoop()
