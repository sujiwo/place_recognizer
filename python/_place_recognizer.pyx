from libc.stdint cimport *
from iBoW cimport IncrementalBoW, KeyPoint, Mat, vector, bool, objectToVectorKeyPoint, NDArrayConverter, DMatch, ImageMatch

cdef Mat fromPyObject():
    pass


cdef class PyIncrementalBoW():
    cdef IncrementalBoW bow
    cdef NDArrayConverter npconv
    
    def __cinit__(self):
        self.bow = IncrementalBoW()
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
    
    def search (self, _descriptors, _numOfYields, _knn, _checks):
        cdef Mat descriptors = self.npconv.toMat(_descriptors)
        cdef vector[vector[DMatch]] featureMatches
        cdef uint32_t knn = _knn
        cdef uint32_t checks = _checks
        cdef uint32_t numOfYields = _numOfYields
        self.bow.searchDescriptors(descriptors, featureMatches, knn, checks)
        cdef vector[DMatch] realMatches
        for m in range(len(featureMatches)):
            if (featureMatches[m][0].distance < featureMatches[m][1].distance * 0.65):
                realMatches.push_back(featureMatches[m][0]);
        cdef vector[ImageMatch] imgMatches
        self.bow.searchImages(descriptors, realMatches, imgMatches)
        
        retval = []
        for i in range(min(numOfYields, len(imgMatches))):
            retval.append(imgMatches[i].image_id)
        return retval