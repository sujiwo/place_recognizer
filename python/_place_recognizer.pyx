from iBoW cimport IncrementalBoW as iBoW, KeyPoint, Mat, vector, bool, objectToVectorKeyPoint, NDArrayConverter, DMatch

cdef Mat fromPyObject():
    pass


cdef class IncrementalBoW():
    cdef iBoW bow
    cdef NDArrayConverter npconv
    
    def __cinit__(self):
        self.bow = iBoW()
        self.npconv = NDArrayConverter()
        
    def addImage (self, imageId, _keypoints, _descriptors):
        cdef vector[KeyPoint] kpts
        objectToVectorKeyPoint(_keypoints, kpts)
        cdef Mat descriptors = self.npconv.toMat(_descriptors)
        return self.bow.addImage(imageId, kpts, descriptors)

    def addImage2 (self, imageId, _keypoints, _descriptors):
        cdef vector[KeyPoint] kpts
        objectToVectorKeyPoint(_keypoints, kpts)
        cdef Mat descriptors = self.npconv.toMat(_descriptors)
        return self.bow.addImage2(imageId, kpts, descriptors)
    
    def search (self, _descriptors):
        pass
        