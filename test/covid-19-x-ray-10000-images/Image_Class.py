from enum import Enum
class Image_Class:
    class type(Enum):
        NORMAL = 0
        COVID = 1
    def __init__(self, file_path, type):
        self.file_path = file_path
        self.type = type

