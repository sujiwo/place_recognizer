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
# Compatibility between Python2 & 3
_pickleProtocol = 2
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
#         self.centerCounts = np.zeros((self.numWords,), dtype=np.uint64)
        
    def train(self, image):
        keypts, descrs = self.featureDetector.detectAndCompute(image, None)
        self.descriptors.append(descrs)
    
    def build(self):
        print("Clustering... ", end="")
        self.descriptors = np.array(list(itertools.chain.from_iterable(self.descriptors))).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        compactness, bestLabels, self.cluster_centers = cv2.kmeans(self.descriptors, self.numWords, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        
        # Put sum of all descriptors
#         print("Building summation of all centers")
#         for i in range(len(bestLabels)):
#             lbl = bestLabels[i]
#             self.centerCounts[lbl] += 1
        print("Done")
    
    # Outputs the index of nearest center using single feature
    def predict1row(self, descriptors):
        assert(descriptors.shape[0]==self.cluster_centers.shape[1])
        dist = np.linalg.norm(self.cluster_centers - descriptors.astype(np.float32), axis=1)
        return np.argmin(dist)
    
    def predict(self, X):
        """
        Search nearest cluster centers for each row of X
        @param X: numpy.ndarray    Image descriptors
        """
        indices = []
        for r in range(X.shape[0]):
            ix = self.predict1row(X[r,:])
            indices.append(ix)
        return np.array(indices, dtype=np.int)
    
    def _adapt(self, newDescriptors, dryRun=False):
        """
        Adjust cluster centers to new set of descriptors as moving averages
        @param newDescriptors: numpy.ndarray    Set of new descriptors
        @param dryRun: bool                     If True, returns new cluster center but do not change it
        @return: numpy.ndarray or None
        """
        assert(newDescriptors.dtype==self.cluster_centers.dtype)
        descCenters = self.predict(newDescriptors)
        movingAverage = copy(self.cluster_centers)
        descCount = np.zeros((self.cluster_centers.shape[0],), dtype=np.uint64)
        for i in range(newDescriptors.shape[0]):
            c = descCenters[i]
            movingAverage[c] += (1.0/float(self.centerCounts[c]+i+1)) * (newDescriptors[i] - movingAverage[c])
            descCount[c] += 1
        movingAverage = movingAverage.astype(np.float32)
        if dryRun==True:
            return movingAverage
        else:
            self.cluster_centers = movingAverage
            self.centerCounts += descCount
            
    def adapt(self, newDescriptors, dryRun=False):
        assert(newDescriptors.dtype==self.cluster_centers.dtype)
        descCenters = self.predict(newDescriptors)
        movingAverage = np.zeros(self.cluster_centers.shape, dtype=np.float64)
        descCount = np.zeros(self.numWords, dtype=np.uint64)
        for i in range(newDescriptors.shape[0]):
            c = descCenters[i]
            descCount[c] += 1
            movingAverage[c] += newDescriptors[i]
        for i in range(self.numWords):
            movingAverage[i] /= float(descCount[i])
        movingAverage = (self.cluster_centers + movingAverage.astype(np.float32)) /2
        if dryRun==True:
            return movingAverage
        else:
            self.cluster_centers = movingAverage
#             self.centerCounts += descCount
        
    
    def save(self, path):
        """
        Save Visual dictionary to file
        """
        fd = open(path, "wb")
        pickle.dump(self.numWords, fd, protocol=_pickleProtocol)
        pickle.dump(self.numFeatures, fd, protocol=_pickleProtocol)
        pickle.dump(self.cluster_centers, fd, protocol=_pickleProtocol)
#         pickle.dump(self.centerCounts, fd, protocol=_pickleProtocol)
        fd.close()
        
    @staticmethod
    def fromClusterCenters(_cluster_centers):
        """
        Create a VisualDictionary from precomputed cluster centers
        @param _cluster_centers: numpy.ndarray
        @return: VisualDictionary
        """
        vd = VisualDictionary()
        vd.numWords = _cluster_centers.shape[0]
        vd.cluster_centers = _cluster_centers
        return vd
        
    @staticmethod
    def load(path):
        """
        Load dictionary from file
        """
        fd = open(path, "rb")
        vd = VisualDictionary()
        vd.numWords = pickle.load(fd)
        vd.numFeatures = pickle.load(fd)
        vd.cluster_centers = pickle.load(fd)
        vd.featureDetector = cv2.ORB_create(vd.numFeatures)
#         vd.centerCounts = pickle.load(fd)
        fd.close()
        return vd
    
    def __getstate__(self):
        dct = self.__dict__.copy()
        if dct.has_key("featureDetector"): del(dct["featureDetector"])
        if dct.has_key("descriptors"): del(dct["descriptors"])
        return dct
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.featureDetector = cv2.ORB_create(self.numFeatures)
    
    


class VLADDescriptor:
    '''
    Aggregated descriptors from an image
    
    Parameters
    ----------
    - descriptors: computed VLAD descriptor of an image, unnormalized
    - centroid_counters: keeps counts of related cluster centers over all descriptors
      of this image
    '''
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
    
    def _adaptNewCentroids(self, dictionary, old_centroids):
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
        self.placeIds = []
        
    @staticmethod
    def is_file_str(filename, open_mode):
        if isinstance(filename, str):
            fd = open(filename, open_mode)
        elif hasattr(filename, "write") or hasattr(filename, "read"):
            fd = filename
        else:
            raise ValueError("Supplied file name is neither path nor file descriptor")
        return fd
        
    def save(self, filetarget):
        """
        Save the map to disk
        
        Parameters
        ----------
        @param filetarget: file name to save map to, or file descriptor from open()
        """
        fd = VLAD2.is_file_str(filetarget, "wb")
        fd.seek(0)
            
        fd.write('VLAD')
        pickle.dump(self.leafSize, fd, protocol=_pickleProtocol)
        pickle.dump(self.tree, fd, protocol=_pickleProtocol)
        pickle.dump(self.placeIds, fd, protocol=_pickleProtocol)
        pickle.dump(self.descriptors, fd, protocol=_pickleProtocol)
        pickle.dump(self.dictionary.cluster_centers, fd, protocol=_pickleProtocol)
        
        if isinstance(filetarget, str):
            fd.close()
    
    @staticmethod
    def load(filesource):
        """
        Load a map from disk
        
        Parameters
        ----------
        @param filesource: file name to load map from
        """
        mvlad = VLAD2(None, True)
        fd = VLAD2.is_file_str(filesource, "rb")
        fd.seek(0)
            
        if (fd.read(4) != "VLAD"):
            raise VLADLoadError("Not a VLAD map file")
        mvlad.leafSize = pickle.load(fd)
        mvlad.tree = pickle.load(fd)
        mvlad.placeIds = pickle.load(fd)
        mvlad.descriptors = pickle.load(fd)
#         Cluster centers
        cc = pickle.load(fd)
        mvlad.dictionary = VisualDictionary.fromClusterCenters(cc)
        
        if isinstance(filesource, str):
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
    def addImage(self, descriptors, keypoints=None, placeId=-1):
        """
        Add image descriptors acquired from a single image during a training session
        
        Parameters
        ----------
        @param placeId: int, Image Id. Can be numbering of image (start from 0) at the original bag
        @param descriptors: numpy.ndarray. Image descriptors generated from feature detector
        @param keypoints: list of keypoints
        """
        if (self.newDatasetDescriptors is None):
            self.newDatasetDescriptors = []
        curPtr = len(self.newDatasetDescriptors)
        self.trainDescriptorPtr.append((curPtr, curPtr+descriptors.shape[0]))
        self.newDatasetDescriptors.extend(descriptors)
#         self.placeIds.append(placeId)
        
    def lastImageId(self):
        if (self.descriptors is None):
            return 0
        else:
            return len(self.descriptors)
        
    # query() should return cartesian coordinates
    def query(self, imgDescriptors, numOfImages=5):
        """
        Search VLAD database for set of image descriptors
        @param imgDescriptors: numpy.ndarray    Image descriptors generated from feature detector
        @param numOfImages: int    Number of images returned from database
        """
        queryDescriptors = VLADDescriptor(imgDescriptors, self.dictionary).flattened().reshape(1,-1)
        dist, idx = self.tree.query(queryDescriptors, k=numOfImages)
        
        # Only return index of image, which can be matched to Metadata (see GenericTrainer)
        return idx[0]
        
        # We got the candidates. Let's check each of them
        
        return [self.placeIds[i] for i in idx[0]]
    
    def _scoreVector(self, candidateIndex, vladImgDescriptor):
        """
        Return vector of score for a candidate index, starting from -5 to +5.
        Initial observation shows that good candidates provide low standard deviation in this vector
        """
        startSeq = max(0, min(candidateIndex-5, len(self.descriptors)))
        stopSeq = min(len(self.descriptors), candidateIndex+5)
        scoresVec = [np.linalg.norm(vladImgDescriptor.flattened() - self.descriptors[v].flattened()) 
            for v in range(startSeq, stopSeq)]
        return np.array(scoresVec)
    
    def _scoreDiffVector(self, candidateIndex, vladImgDescriptor):
        scoreVect = self._scoreVector(candidateIndex, vladImgDescriptor)
        myscore = np.linalg.norm(vladImgDescriptor.flattened()-self.descriptors[candidateIndex].flattened())
        return scoreVect-myscore
        

    @staticmethod
    def normalizeVlad(vDescriptors):
        for r in range(vDescriptors.shape[0]):
            l2 = np.linalg.norm(vDescriptors[r])
        vDescriptors = vDescriptors / np.linalg.norm(vDescriptors)
        return vDescriptors
    
    def rebuildVladDescriptors(self, oldDictionary):
        for vd in self.descriptors:
            vd._adaptNewCentroids(self.dictionary, oldDictionary)
            
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
        self.newDatasetDescriptors = np.array(self.newDatasetDescriptors, dtype=self.dictionary.cluster_centers.dtype)
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
        print("Build Index Tree")
        self.tree = KDTree(D, leaf_size=self.leafSize)
        
    

