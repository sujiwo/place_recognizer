from __future__ import print_function
import cv2
from sklearn.neighbors import KDTree
from scipy.spatial.distance import hamming
import mlcrate as mlc
import pickle
import numpy as np
import itertools
from numpy import dtype
from copy import copy

np.seterr(all='raise')
# XXX: Check VLfeat source code

class VisualDictionary():
    """
    VisualDictionary is a class that represents codebook to translate image features
    to VLAD
    """
    
    dtype = np.float32
    
    def __init__ (self, numWords=256, numFeaturesOnImage=3000):
        self.numWords = numWords
        self.numFeatures = numFeaturesOnImage
        self.featureDetector = cv2.ORB_create(numFeaturesOnImage)
        self.descriptors = []
        self.cluster_centers = []
        
    def train(self, image):
        keypts, descrs = self.featureDetector.detectAndCompute(image, None)
        self.descriptors.append(descrs)
    
    def build(self):
        print("Clustering... ", end="")
        self.descriptors = np.array(list(itertools.chain.from_iterable(self.descriptors))).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        compactness, self.bestLabels, self.cluster_centers = cv2.kmeans(self.descriptors, self.numWords, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        print("Done")
    
    # Outputs the index of nearest center using single feature
    def predict1row(self, descriptors):
        assert(descriptors.shape[0]==self.cluster_centers.shape[1])
#         dist = []
#         for ci in range(len(self.cluster_centers)):
#             c = self.cluster_centers[ci]
#             d = np.linalg.norm(c - descriptors.astype(np.float32))
#             dist.append(d)
        dist = np.linalg.norm(self.cluster_centers - descriptors.astype(np.float32), axis=1)
        return np.argmin(dist)
    
    def predict(self, X):
        indices = []
        for r in range(X.shape[0]):
            ix = self.predict1row(X[r,:])
            indices.append(ix)
        return np.array(indices, dtype=np.int)
    
    def adapt(self, newDescriptors, dryRun=False):
        assert(newDescriptors.dtype==self.cluster_centers.dtype)
        descCenters = self.predict(newDescriptors)
        descSums = np.zeros(self.cluster_centers.shape, dtype=np.float64)
        descCount = np.zeros((self.cluster_centers.shape[0],), dtype=np.uint64)
        for i in range(newDescriptors.shape[0]):
            c = descCenters[i]
            f = newDescriptors[i]
            descSums[c] += f
            descCount[c] += 1
        for c in range(self.cluster_centers.shape[0]):
            descSums[c] /= float(descCount[c])
        descSums = descSums.astype(np.float32)
        if dryRun==True:
            return descSums
        else:
            self.cluster_centers = descSums
    
    def save(self, path):
        fd = open(path, "wb")
        pickle.dump(self.numWords, fd)
        pickle.dump(self.numFeatures, fd)
        pickle.dump(self.bestLabels, fd)
        pickle.dump(self.cluster_centers, fd)
        fd.close()
        
    @staticmethod
    def fromClusterCenters(_cluster_centers):
        vd = VisualDictionary()
        vd.numWords = _cluster_centers.shape[0]
        vd.cluster_centers = _cluster_centers
        return vd
        
    
    @staticmethod
    def load(path):
        fd = open(path, "rb")
        vd = VisualDictionary()
        vd.numWords = pickle.load(fd)
        vd.numFeatures = pickle.load(fd)
        vd.featureDetector = cv2.ORB_create(vd.numFeatures)
        vd.bestLabels = pickle.load(fd)
        vd.cluster_centers = pickle.load(fd)
        fd.close()
        return vd
    
    
class VisualDictionaryBinaryFeature(VisualDictionary):
    def __init__ (self, numWords=256, numFeaturesOnImage=3000):
        self.numWords = numWords
        self.numFeatures = numFeaturesOnImage
        self.featureDetector = cv2.ORB_create(numFeaturesOnImage)
        self.descriptors = []
        self.cluster_centers = []
        self.bestLabels = []
        
    def train(self, image):
        keypts, descrs = self.featureDetector.detectAndCompute(image, None)
        self.descriptors.append(descrs)
    
    def build(self):
        # Not running, really.
        # We turn to Matlab for computing cluster centers     
        print("Training done")
        
    @staticmethod
    def hamming_distance(f1, f2):
        assert(len(f1)==len(f2) and f1.dtype==np.uint8 and f2.dtype==np.uint8)
        return cv2.norm(f1, f2, cv2.NORM_HAMMING)
    
    # Outputs the index of nearest center using single feature
    def predict1row(self, descriptors):
        assert(descriptors.shape[0]==self.cluster_centers.shape[1])
        dist = [self.hamming_distance(descriptors, c) for c in self.cluster_centers ]
        return np.argmin(dist)
    
#     def predict(self, X):
#         indices = []
#         for r in range(X.shape[0]):
#             ix = self.predict1row(X[r,:])
#             indices.append(ix)
#         return np.array(indices, dtype=np.int)
    
#     def save(self, path):
#         pass
#     
    @staticmethod
    def load(path):
        fd = open(path, "rb")
        vd = VisualDictionaryBinaryFeature()
        vd.numWords = pickle.load(fd)
        vd.numFeatures = pickle.load(fd)
        vd.featureDetector = cv2.ORB_create(vd.numFeatures)
        vd.descriptors = pickle.load(fd)
        vd.bestLabels = pickle.load(fd)
        vd.cluster_centers = pickle.load(fd)
        fd.close()
        return vd


class VLADDescriptor:
    def __init__(self, imageDescriptors, dictionary):
        assert(isinstance(dictionary, VisualDictionary))
        self.descriptors, self.centroid_counters = VLADDescriptor.compute(imageDescriptors, dictionary)
        
    def shape(self):
        return self.descriptors.shape
    
    def dtype(self):
        return self.descriptors.dtype
    
    @staticmethod
    def compute(imgDesc, dictionary):
        predictedLabels = dictionary.predict(imgDesc)
        centers = dictionary.cluster_centers
        k=dictionary.cluster_centers.shape[0]
        m,d = imgDesc.shape
        V=np.zeros([k,d])
        C=np.zeros(k, dtype=np.int)
        #computing the differences
        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            C[i] = np.sum(predictedLabels==i)
            if C[i]>0:
                # add the diferences
                # XXX: what's the best formula?
                V[i]=np.sum(imgDesc[predictedLabels==i,:] - centers[i], axis=0)
                l2 = np.linalg.norm(V[i])
        return V, C
    
    def normalized(self):
        V = copy(self.descriptors)
        for r in range(V.shape[0]):
            l2 = np.linalg.norm(V[r])
            if (l2<1e-3):
                V[r] = 0
            else:
                V[r] = V[r] / l2
        V = V / np.linalg.norm(V)
        return V
    
    def flattened(self):
        return self.normalized().flatten()
    
    def adaptNewCentroids(self, dictionary, old_centroids):
        for i in range(self.descriptors.shape[0]):
            self.descriptors[i] = self.centroid_counters[i]*(dictionary.cluster_centers[i] - old_centroids[i]) + self.descriptors[i]

   
class VLADLoadError(IOError):
    pass
    
    
class VLAD2():
    """
    VLAD Indexing and query Class
    """
    
    def __init__ (self, D, _blank=False):
        """
        Initialize VLAD class
        
        Parameters
        ----------
        @param D: VisualDictionary
        @param _blank: bool, unused
        """
        if (_blank==False):
            assert(isinstance(D, VisualDictionary))
            self.dictionary = D
        else:
            self.dictionary = None
        self.descriptors = None
        self.imageIds = []
        
    def save(self, path):
        """
        Save the map to disk
        
        Parameters
        ----------
        @param path: file name to save map to
        """
        fd = open(path, "wb")
        fd.write('VLAD')
        pickle.dump(self.leafSize, fd)
        pickle.dump(self.tree, fd)
        pickle.dump(self.imageIds, fd)
        pickle.dump(self.descriptors, fd)
        pickle.dump(self.dictionary.cluster_centers, fd)
        fd.close()
    
    @staticmethod
    def load(path):
        """
        Load a map from disk
        
        Parameters
        ----------
        @param path: file name to load map from
        """
        mvlad = VLAD2(None, True)
        fd = open(path, "rb")
        if (fd.read(4) != "VLAD"):
            raise VLADLoadError("Not a VLAD map file")
        mvlad.leafSize = pickle.load(fd)
        print("Tree 1")
        mvlad.tree = pickle.load(fd)
        print("Tree 2")
        mvlad.imageIds = pickle.load(fd)
        mvlad.descriptors = pickle.load(fd)
#         Cluster centers
        cc = pickle.load(fd)
        mvlad.dictionary = VisualDictionary.fromClusterCenters(cc)
        
        fd.close()
        return mvlad
    
    def initTrain(self, leafSize=40):
        """
        Start a new training session
        
        Parameters
        ----------
        @param leafSize: integer, number of leafs for index tree. Changing this number
        may affect query time and/or performance
        """
        self.newDatasetDescriptors = None
        self.leafSize = leafSize
        self.trainDescriptorPtr = []
        
    # XXX: Insert cartesian coordinate of the image when adding
    # imageId is actually vestigial data that can be replaced with other datatypes which will be
    # returned upon query, eg. geographic coordinates
    def addImage(self, imageId, descriptors, keypoints=None):
        """
        Add image descriptors acquired from a single image during a training session
        
        Parameters
        ----------
        @param imageId: int, Image Id. Can be numbering of image (start from 0) at the original bag
        @param descriptors: numpy.ndarray. Image descriptors generated from feature detector
        @param keypoints: list of keypoints
        """
        if (self.newDatasetDescriptors is None):
            self.newDatasetDescriptors = np.zeros((0,descriptors.shape[1]), dtype=descriptors.dtype)
        curPtr = self.newDatasetDescriptors.shape[0]
        self.trainDescriptorPtr.append((curPtr, curPtr+descriptors.shape[0]))
        self.newDatasetDescriptors = np.append(self.newDatasetDescriptors, descriptors.astype(VisualDictionary.dtype), axis=0)
        self.imageIds.append(imageId)
        
    # query() should return cartesian coordinates
    def query(self, imgDescriptors, numOfImages=5):
        queryDescriptors = VLADDescriptor(imgDescriptors, self.dictionary).flattened().reshape(1,-1)
        dist, idx = self.tree.query(queryDescriptors, numOfImages)
        
        # We got the candidates. Let's check each of them
        
        return [self.imageIds[i] for i in idx[0]]

    @staticmethod
    def normalizeVlad(vDescriptors):
        for r in range(vDescriptors.shape[0]):
            l2 = np.linalg.norm(vDescriptors[r])
        vDescriptors = vDescriptors / np.linalg.norm(vDescriptors)
        return vDescriptors
    
    def rebuildVladDescriptors(self, oldDictionary):
        for vd in self.descriptors:
            vd.adaptNewCentroids(self.dictionary, oldDictionary)
            
    def flatNormalDescriptors(self):
        shp = self.descriptors[0].shape()
        flatDescriptors = np.zeros((len(self.descriptors), shp[0]*shp[1]), dtype=self.descriptors[0].dtype())
        for i in range(len(self.descriptors)):
            d = self.descriptors[i].flattened()
            flatDescriptors[i] = d
        return flatDescriptors
    
    def stopTrain(self):
        """
        Ends a training session
        """
        hasTrained = True
        if (self.descriptors is None):
            self.descriptors=[]
            hasTrained = False
        print("Cluster center adaptation")
        oldDictionary = copy(self.dictionary.cluster_centers)
        self.dictionary.adapt(self.newDatasetDescriptors)
        
        if hasTrained==True:
            print("Adapting old VLAD descriptors to new centroids")
            self.rebuildVladDescriptors(oldDictionary)
        
        print("Build VLAD from data stream")
        # XXX; This loop is amenable to parallelization
        for ptr in self.trainDescriptorPtr:
            imgDescriptors = self.newDatasetDescriptors[ptr[0] : ptr[1]]
            newvd = VLADDescriptor(imgDescriptors, self.dictionary)
            self.descriptors.append (newvd)

        D = self.flatNormalDescriptors()
        # XXX: Switch to KDTree ?
        # Implementation choices: SKlearn vs OpenCV
        self.tree = KDTree(D, leaf_size=self.leafSize)
        
    

