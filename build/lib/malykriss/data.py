
from helchriss.dsl.dsl_types import TypeBase

class ImageType(TypeBase):
    def __init__(self, resolution = (64,64), channel = 3):
        super().__init__(f'Image[{resolution[0]}x{resolution[1]}x{channel}]')
        self.resolution = resolution
        self.channel = channel