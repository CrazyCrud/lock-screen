import os
import time
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

class RecognitionSystem:
    def __init__(self, window_width, window_height):
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

    def capture_image(self, bpm_frame):
        image_name = time.strftime('image_%H_%M_%S')
        bpm_frame.SaveFile(definitions.ROOT_DIR + '\\resources\\Constantin_Lehenmeier\\' + image_name + '.jpg',
                           wx.BITMAP_TYPE_JPEG)
        print("Save image file as %s" % image_name)

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
        classifier_path = definitions.ROOT_DIR + '\\resources\\models\\face_classifier.sav'
        if os.path.exists(classifier_path) is True:
            self.classifier = pickle.load(open(classifier_path, 'rb'))
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

            pickle.dump(self.classifier, open(classifier_path, 'wb'))

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
        self.fps = fps
        self.window_width = 800
        self.window_height = 650
        self.image_width = 800
        self.image_height = 450
        super(AppFrame, self).__init__(parent, id=-1, title=self.title,
                                       size=(self.window_width, self.window_height))
        self.init()
        self.init_ui()

    def init(self):
        if self.LAYOUT_ONLY is not True:
            self.app_backend = RecognitionSystem(self.image_width, self.image_height)

            identities = self.app_backend.load_identities()
            identities_embeded = self.app_backend.load_training_data(identities)
            self.app_backend.train(identities=identities, embeded_identities=identities_embeded, show_accuracy=False)
        self.webcam = WebcamFeed()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.GridBagSizer(0, 0)

        image_frame = self.webcam.get_image(self.image_width, self.image_height)
        self.bmp_frame = wx.Bitmap.FromBuffer(self.image_width, self.image_height, image_frame)
        self.image_static = wx.StaticBitmap(panel, bitmap=self.bmp_frame, size=(self.image_width, self.image_height))
        text_control = wx.TextCtrl(panel)
        button_capture = wx.Button(panel, label="Capture Image")
        button_capture.Bind(wx.EVT_BUTTON, self.on_capture)
        button_train = wx.Button(panel, label="Train")
        combobox = wx.ComboBox(panel, style=wx.CB_DROPDOWN)
        button_activate = wx.ToggleButton(panel, label="Activate Lock Screen")
        # image_placeholder = \
        # wx.StaticText(panel, label="Image goes here...", size=(self.image_width, self.image_height))

        sizer.Add(self.image_static, pos=(0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL)
        sizer.Add(text_control, pos=(1, 0), flag=wx.ALL | wx.CENTER)
        sizer.Add(button_capture, pos=(2, 0), flag=wx.ALL | wx.CENTER)
        sizer.Add(combobox, pos=(1, 1), flag=wx.ALL | wx.CENTER)
        sizer.Add(button_train, pos=(2, 1), flag=wx.ALL | wx.CENTER)
        sizer.Add(button_activate, pos=(3, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL | wx.CENTER)

        panel.SetSizerAndFit(sizer)

        self.Centre()
        self.Show()

        self.timer = wx.Timer(self)
        self.timer.Start(1000. // self.fps)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_TIMER, self.next_frame)

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
        print("Take photo")
        self.app_backend.capture_image(self.bmp_frame)


class App(wx.App):
    def OnInit(self):
        app_frame = AppFrame(None)
        app_frame.Show(True)
        return True


if __name__ == '__main__':
    app = App()
    app.MainLoop()
