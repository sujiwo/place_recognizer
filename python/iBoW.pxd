from libcpp cimport bool
from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "opencv2/core/core.hpp" namespace "cv":
    ctypedef struct KeyPoint:
        pass
    ctypedef struct Mat:
        pass
    ctypedef struct DMatch:
        int queryIdx
        int trainIdx
        int imgIdx
        float distance

    

cdef extern from "conversion.h":
    cdef bool objectToVectorKeyPoint(obj, vector[KeyPoint] vec)
    cdef cppclass NDArrayConverter:
        NDArrayConverter()
        Mat toMat(npyObj)


cdef extern from "IncrementalBoW.h" namespace "PlaceRecognizer":
    ctypedef struct ImageMatch:
        int image_id
        double score

    cdef cppclass IncrementalBoW:
        IncrementalBoW()
        void addImage  (uint32_t, vector[KeyPoint], Mat)
        void addImage2 (uint32_t, vector[KeyPoint], Mat)
        void searchImages(Mat, vector[DMatch], vector[ImageMatch])
        void searchDescriptors(Mat, vector[vector[DMatch]], uint32_t, uint32_t)
        void saveToDisk(string)
        void loadFromDisk(string)


# class iBoW():
#     
#     cpdef __init__(self):
#         self.ibow = IncrementalBoW()
#     
#     def addImage(self, imagId, descriptors):
#         pass
#     
#     def addImage2(self, imageId, descriptors):
#         pass
#     
#     def search(self):
#         pass
#     
#     def save(self, path):
#         pass
#     
#     def load(self, path):
#         pass