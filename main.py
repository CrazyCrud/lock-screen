import os
import time
import wx
import cv2
import re
import pathlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pickle
from glob import glob

import definitions
from classifing.image_classifing import Model
from image_processing.image_processing import ImageProcessing
from image_processing.webcam import WebcamFeed


# image = preprocessing.load_image(definitions.ROOT_DIR + '\\resources\\images\\image-1.jpg')
# image_processed = preprocessing.preprocess_image(image, visualize=True)

class RecognitionSystem:
    def __init__(self, window_width, window_height):
        self.FORCE_TRAIN = True
        self.window_width = window_width
        self.window_height = window_height

        self.preprocessing = ImageProcessing()
        self.model = Model()
        """Create a pretrained model"""
        self.model_nn = self.model.create_model()
        self.encoder = LabelEncoder()
        self.classifier = None

    def load_image(self, path):
        return self.preprocessing.load_image(path)

    def capture_image(self, bpm_frame, file_name):
        image_name = time.strftime('image_%H_%M_%S')
        file_name = file_name.strip()
        file_name = re.sub('\s', '_', file_name)
        file_dir = definitions.ROOT_DIR + '\\resources\\images\\' + file_name
        if not os.path.isdir(file_dir):
            pathlib.Path(file_dir).mkdir(parents=False, exist_ok=True)
            # TODO: scan folders for list in combobox or just add it
        bpm_frame.SaveFile(file_dir + '\\' + image_name + '.jpg',
                           wx.BITMAP_TYPE_JPEG)

        print("Save image file as {}".format(file_dir + '\\' + image_name))

    def preprocess_image(self, image):
        return self.preprocessing.preprocess_image(image)

    def load_identities(self):
        return self.model.load_identities(definitions.ROOT_DIR + '\\resources\\images')

    def load_training_data(self, idendities):
        """Load the training images and try to detect a face"""
        idendities_images = []
        identities_mod = []
        for index, value in enumerate(idendities):
            print("Get image path {}".format(value.get_image_path()))
            image = self.preprocessing.load_image(value.get_image_path())
            print("Image could not be loaded: {}".format(True if image is None else False))
            image = self.preprocessing.preprocess_image(image)
            print("Face could not be detected: {}".format(True if image is None else False))
            if image is not None:
                idendities_images.append(image)
                identities_mod.append(value)

        """Create the embeded vectors from the training images"""
        embeded_data = self.model.create_embeded_data_from_identities(idendities_images, self.model_nn)
        return np.array(identities_mod), embeded_data

    def train(self, identities, embeded_identities, show_accuracy=False):
        classifier_path = definitions.ROOT_DIR + '\\resources\\models\\face_classifier.sav'
        encoder_path = definitions.ROOT_DIR + '\\resources\\models\\face_encoder.sav'
        if os.path.exists(classifier_path) is True and os.path.exists(
                encoder_path) is True and self.FORCE_TRAIN is False:
            self.classifier = pickle.load(open(classifier_path, 'rb'))
            self.encoder = pickle.load(open(encoder_path, 'rb'))
        else:
            targets = np.array([identity.name for identity in identities])
            print("Targets: {}".format(targets))
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

            pickle.dump(self.classifier, open(classifier_path, 'wb'))
            pickle.dump(self.encoder, open(encoder_path, 'wb'))

            if show_accuracy is True:
                acc_svc = accuracy_score(labels_test, self.classifier.predict(data_test))
                print(f'SVM accuracy = {acc_svc}')

    def predict(self, image):
        embeded_identity = self.model.create_embeded_data_from_identity(image, self.model_nn)
        prediction = self.classifier.predict([embeded_identity])
        identity = self.encoder.inverse_transform(prediction)[0]
        return identity


class AppFrame(wx.Frame):
    def __init__(self, parent, fps=15):
        self.title = "Lock Screen"
        self.LAYOUT_ONLY = True

        self.app_backend = None
        self.webcam = None

        self.timer = None
        self.bmp_frame = self.image_static = None
        self.text_control = None
        self.button_capture = None
        self.button_train = None
        self.button_activate = None
        self.combobox = None

        self.fps = fps
        self.window_width = 800
        self.window_height = 650
        self.image_width = 800
        self.image_height = 450
        self.training_folders = []
        super(AppFrame, self).__init__(parent, id=-1, title=self.title,
                                       size=(self.window_width, self.window_height))
        self.init()
        self.init_ui()

    def init(self):
        if self.LAYOUT_ONLY is not True:
            self.app_backend = RecognitionSystem(self.image_width, self.image_height)

            identities = self.app_backend.load_identities()
            print("Identities: {}".format(identities))
            identities, identities_embeded = self.app_backend.load_training_data(identities)
            self.app_backend.train(identities=identities, embeded_identities=identities_embeded, show_accuracy=False)
        self.webcam = WebcamFeed()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.GridBagSizer(0, 0)

        image_frame = self.webcam.get_image(self.image_width, self.image_height)
        self.bmp_frame = wx.Bitmap.FromBuffer(self.image_width, self.image_height, image_frame)
        self.image_static = wx.StaticBitmap(panel, bitmap=self.bmp_frame, size=(self.image_width, self.image_height))
        self.text_control = wx.TextCtrl(panel)
        self.button_capture = wx.Button(panel, label="Capture Image")
        self.button_capture.Bind(wx.EVT_BUTTON, self.on_capture)
        self.button_train = wx.Button(panel, label="Train")

        self.scan_train_folder()
        self.combobox = wx.ComboBox(panel, value=self.training_folders[0], choices=self.training_folders, style=wx.CB_DROPDOWN)

        self.button_activate = wx.ToggleButton(panel, label="Activate Lock Screen")
        # image_placeholder = \
        # wx.StaticText(panel, label="Image goes here...", size=(self.image_width, self.image_height))

        sizer.Add(self.image_static, pos=(0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL)
        sizer.Add(self.text_control, pos=(1, 0), flag=wx.ALL | wx.CENTER)
        sizer.Add(self.button_capture, pos=(2, 0), flag=wx.ALL | wx.CENTER)
        sizer.Add(self.combobox, pos=(1, 1), flag=wx.ALL | wx.CENTER)
        sizer.Add(self.button_train, pos=(2, 1), flag=wx.ALL | wx.CENTER)
        sizer.Add(self.button_activate, pos=(3, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL | wx.CENTER)

        panel.SetSizerAndFit(sizer)

        self.Centre()
        self.Show()

        self.timer = wx.Timer(self)
        self.timer.Start(1000. // self.fps)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_TIMER, self.next_frame)

    def scan_train_folder(self):
        root_folder = definitions.ROOT_DIR + '\\resources\\images\\'
        sub_folders = [x[0] for x in os.walk(root_folder)]
        sub_folders = sub_folders[1:]
        sub_folders = list(map(lambda x: self.get_person_name_from_folder(x), sub_folders))
        self.training_folders = sub_folders
        print("Sub folders of {}: \n{}".format(root_folder, sub_folders))

    def get_person_name_from_folder(self, absolute_path):
        absolute_path = re.sub(r"\s+", "", absolute_path, flags=re.UNICODE)  # hack because whitespace in path
        person_name = absolute_path[absolute_path.rfind('\\') + 1:]
        return person_name

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp_frame, 0, 0)

    def next_frame(self, event):
        frame = self.webcam.get_image(self.window_width, self.window_height)
        if frame is not None:
            self.bmp_frame.CopyFromBuffer(frame)
            # self.image_static.SetBitmap(self.bmp_frame)
            self.Refresh()

            if self.LAYOUT_ONLY is not True:
                face_image = self.app_backend.preprocess_image(frame)
                if face_image is not None:
                    identity = self.app_backend.predict(face_image)
                    print('Identity:', identity)

    def on_capture(self, event):
        name = self.text_control.GetLineText(0)
        if name:
            self.app_backend.capture_image(self.bmp_frame, name)


class App(wx.App):
    def OnInit(self):
        app_frame = AppFrame(None)
        app_frame.Show(True)
        return True


if __name__ == '__main__':
    app = App()
    app.MainLoop()
