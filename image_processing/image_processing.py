import cv2
import definitions
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from image_processing.alignment import AlignDlib


class ImageProcessing:
    def __init__(self):
        self.alignment = AlignDlib(definitions.ROOT_DIR + '\\resources\\models\\face_landmarks_68.dat')

    def load_image(self, path):
        if path is None:
            return None
        else:
            image_bgr = cv2.imread(path)
            image_rgb = self.convert_image_to_rgb(image_bgr)
            return image_rgb

    def preprocess_image(self, image, visualize=False):
        if image is None:
            return None
        else:
            face_bounding_box = self.alignment.getLargestFaceBoundingBox(image)
            if face_bounding_box is None:
                return None
            else:
                image_face = self.alignment.align(96, image, face_bounding_box,
                                                  landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

                if visualize is True:
                    self._visualize(image, face_bounding_box, image_face)

                return image_face

    @staticmethod
    def convert_image_to_rgb(self, image_bgr):
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _visualize(self, image_raw, face_bounding_box, image_face):
        # print("Visualize image")
        plt.subplot(131)
        plt.imshow(image_raw)

        plt.subplot(132)
        plt.imshow(image_raw)
        plt.gca().add_patch(
            patches.Rectangle((face_bounding_box.left(), face_bounding_box.top()), face_bounding_box.width(),
                              face_bounding_box.height(), fill=False, color='red'))

        plt.subplot(133)
        plt.imshow(image_face)

        plt.show()
