from keras.models import Model
from keras.layers import Input
import numpy as np

import definitions
from classifing.models.triplet_loss_layer import TripletLossLayer
from classifing.models.model import create_model
from classifing.identities.identities_processing import IdentityData


class Model:
    def __init__(self):
        self.identity_data = IdentityData()
        self.embeded_data = None

    def create_model(self, pretrained=True):
        if pretrained is True:
            return self._create_model_pretrained()
        else:
            return self._create_model_raw()

    def create_embeded_data_from_identities(self, idendity_images, model):
        idendity_images = np.array(idendity_images)
        embeded_data = np.zeros((idendity_images.shape[0], 128))

        for index, image in enumerate(idendity_images):
            embeded_data[index] = self.create_embeded_data_from_identity(image, model)

        return embeded_data

    def create_embeded_data_from_identity(self, identity_image, model):
        # scale RGB values to interval [0,1]
        identity_image = (identity_image / 255.).astype(np.float32)
        # obtain embedding vector for image
        return model.predict(np.expand_dims(identity_image, axis=0))[0]

    def load_identities(self, path):
        return self.identity_data.load_identities(path)

    def _create_model_pretrained(self):
        nn4_small2_pretrained = create_model()
        nn4_small2_pretrained.load_weights(definitions.ROOT_DIR + '\\resources\\models\\nn4.small2.v1.h5')
        return nn4_small2_pretrained

    def _create_model_raw(self):
        # Input for anchor, positive and negative images
        in_a = Input(shape=(96, 96, 3))
        in_p = Input(shape=(96, 96, 3))
        in_n = Input(shape=(96, 96, 3))

        nn4_small2 = create_model()

        # Output for anchor, positive and negative embedding vectors
        # The nn4_small model instance is shared (Siamese network)
        emb_a = nn4_small2(in_a)
        emb_p = nn4_small2(in_p)
        emb_n = nn4_small2(in_n)

        # Layer that computes the triplet loss from anchor, positive and negative embedding vectors
        triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

        # Model that can be trained with anchor, positive negative images
        nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)

        return nn4_small2_train
