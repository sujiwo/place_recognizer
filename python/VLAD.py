import cv2
from sklearn.neighbors import BallTree
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer 
import mlcrate as mlc
import pickle
import numpy as np
import itertools
from numpy import dtype


# XXX: Check VLfeat source code

class VisualDictionary():
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
        self.descriptors = np.array(list(itertools.chain.from_iterable(self.descriptors))).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        compactness, self.bestLabels, self.cluster_centers = cv2.kmeans(self.descriptors, self.numWords, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        print("Training done")
    
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
    
    def save(self, path):
        fd = open(path, "wb")
        pickle.dump(self.numWords, fd)
        pickle.dump(self.numFeatures, fd)
        pickle.dump(self.descriptors, fd)
        pickle.dump(self.bestLabels, fd)
        pickle.dump(self.cluster_centers, fd)
        fd.close()
    
    @staticmethod
    def load(path):
        fd = open(path, "rb")
        vd = VisualDictionary()
        vd.numWords = pickle.load(fd)
        vd.numFeatures = pickle.load(fd)
        vd.featureDetector = cv2.ORB_create(vd.numFeatures)
        vd.descriptors = pickle.load(fd)
        vd.bestLabels = pickle.load(fd)
        vd.cluster_centers = pickle.load(fd)
        fd.close()
        return vd
    
    
class VisualDictionaryBinaryFeature():
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
        self.descriptors = np.array(self.descriptors, dtype=self.descriptors[0].dtype)
        
        # Step 1: Initialize cluster centers using K-Means++
        self.cluster_centers = kmeans_plusplus_initializer(self.descriptors, self.numWords).initialize()
        
        # Step 2: Run K-Means with Hamming distance
        clusters = kmeans(self.descriptors, self.cluster_centers, tolerance, ccore)
        
        print("Training done")
    
    # Outputs the index of nearest center using single feature
    def predict1row(self, descriptors):
        pass
    
    def predict(self, X):
        pass
    
    def save(self, path):
        pass
    
    @staticmethod
    def load(path):
        pass


class VLAD():
    def __init__ (self, numFeatures=3000):
        self.numFeatures = numFeatures
        self.orb = cv2.ORB_create(numFeatures)
        self.dictionary = None
        pass
    
    def loadDictionary(self, dictPath):
        self.dictionary = VisualDictionary.load(dictPath)
        
    def initTrain(self, leafSize=40):
        self.datasetDescriptors = []
        self.imageIds = []
        self.leafSize = leafSize
        
    # train() produces VLAD descriptor of an input image
    def train(self, imageId, image, mask=None):
        keypoints, descriptors = self.orb.detectAndCompute(image, mask)
        V = self.computeVlad2(descriptors)
        self.imageIds.append(imageId)
        self.datasetDescriptors.append(V)
        
    def stopTrain(self):
        self.descriptors = np.asarray(self.datasetDescriptors)
        del(self.datasetDescriptors)
        
        # Create index ball-tree
        self.tree = BallTree(self.descriptors, leaf_size=self.leafSize)
        print("Done training")
        
    def computeVlad(self, descriptors, raw=False):
        predictedLabels = self.dictionary.predict(descriptors)
        centers = self.dictionary.cluster_centers
        k=self.dictionary.cluster_centers.shape[0]
        m,d = descriptors.shape
        V=np.zeros([k,d])

        #computing the differences
        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels==i)>0:
                # add the diferences
                V[i]=np.sum(descriptors[predictedLabels==i,:]-centers[i],axis=0)
        
        if (raw==True):
            return V
        
        V = V.flatten()
        # power normalization, also called square-rooting normalization
        V = np.sign(V)*np.sqrt(np.abs(V))

        # L2 normalization
        V = V/np.sqrt(np.dot(V,V))
        return V
    
    # VLAD with intra-normalization
    def computeVlad2(self, descriptors):
        predictedLabels = self.dictionary.predict(descriptors)
        centers = self.dictionary.cluster_centers
        k=self.dictionary.cluster_centers.shape[0]
        m,d = descriptors.shape
        V=np.zeros([k,d])

        #computing the differences
        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels==i)>0:
                # add the diferences
                V[i]=np.sum(descriptors[predictedLabels==i,:]-centers[i],axis=0)
                l2 = np.linalg.norm(V[i])
                V[i] = V[i] / l2
        V = V.flatten()
        V = V / np.linalg.norm(V)
        return V

    def query(self, image, numOfImages):
        kp, descs = self.orb.detectAndCompute(image, None)
        vl = self.computeVlad(descs)
        vl = vl.reshape((1, vl.shape[0]))
        dist, idx = self.tree.query(vl, numOfImages)
        res = [self.imageIds[i] for i in idx[0]]
        return res

    def save(self, path):
        fd = open(path, "wb")
        pickle.dump(self.numFeatures, fd)
        pickle.dump(self.leafSize, fd)
        pickle.dump(self.tree, fd)
        pickle.dump(self.imageIds, fd)
        fd.close()
    
    @staticmethod
    def load(path):
        vld = VLAD()
        fd = open(path, "rb")
        vld.numFeatures = pickle.load(fd)
        vld.leafSize = pickle.load(fd)
        vld.tree = pickle.load(fd)
        vld.imageIds = pickle.load(fd)
        fd.close()
        return vld
    
if __name__ == '__main__':
    vd = VLAD()
    vd.loadDictionary("/tmp/test_visual_dict.dat")
    
    pass

