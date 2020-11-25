cdef extern from "conversion.h":
    cdef cppclass NDArrayConverter:
        NDArrayConverter()

cdef extern from "IncrementalBoW.h" namespace "PlaceRecognizer":
    cdef cppclass IncrementalBoW:
        IncrementalBoW()


class iBoW():
    
    def __cinit__(self):
        pass
    
    def addImage(self, imagId, descriptors):
        pass
    
    def addImage2(self, imageId, descriptors):
        pass
    
    def search(self):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass