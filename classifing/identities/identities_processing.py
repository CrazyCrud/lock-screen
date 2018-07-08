import os
import numpy as np
from classifing.identities.identity import Identity


class IdentityData:
    def __init__(self):
        pass

    def load_identities(self, path):
        identities = []
        for dir_identity in os.listdir(path):
            if os.path.isdir(os.path.join(path, dir_identity)):
                for file in os.listdir(os.path.join(path, dir_identity)):
                    identities.append(Identity(path, dir_identity, file))
        return np.array(identities)

    def load_identity(self, path):
        identities = []
        if os.path.isdir(path):
            head, tail = os.path.split(path)
            for file in os.listdir(path):
                identities.append(Identity(head, tail, file))
        return np.array(identities)