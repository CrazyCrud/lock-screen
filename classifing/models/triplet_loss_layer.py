from keras import backend as KerasBackend
from keras.layers import Layer


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = KerasBackend.sum(KerasBackend.square(a - p), axis=-1)
        n_dist = KerasBackend.sum(KerasBackend.square(a - n), axis=-1)
        return KerasBackend.sum(KerasBackend.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
