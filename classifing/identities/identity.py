import os


class Identity():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.get_image_path()

    def get_image_path(self):
        return os.path.join(self.base, self.name, self.file)
