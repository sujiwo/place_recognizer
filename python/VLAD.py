import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import mlcrate as mlc
import pickle
import numpy as np
import itertools


class VisualDictionary():
    def __init__ (self, numWords=64, numFeaturesOnImage=3000):
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
        dist = []
        for ci in range(len(self.cluster_centers)):
            c = self.cluster_centers[ci]
            d = np.linalg.norm(c - descriptors.astype(np.float32))
            dist.append(d)
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
        numWords = pickle.load(fd)
        numFeatures = pickle.load(fd)
        vd = VisualDictionary()
        vd.descriptors = pickle.load(fd)
        vd.bestLabels = pickle.load(fd)
        vd.cluster_centers = pickle.load(fd)
        fd.close()
        return vd

class VLAD():
    def __init__ (self, numFeatures=3000):
        self.numFeatures = numFeatures
        self.orb = cv2.ORB_create(numFeatures)
        self.dictionary = None
        pass
    
    def loadDictionary(self, dictPath):
        self.dictionary = mlc.load(dictPath)
        
    def initTrain(self, leafSize=40):
        self.datasetDescriptors = []
        self.imageIds = []
        self.leafSize = leafSize
        
    # train() produces VLAD descriptor of an input image
    def train(self, imageId, image, mask=None):
        keypoints, descriptors = self.orb.detectAndCompute(image, mask)
        V = self.computeVlad(descriptors)
        self.imageIds.append(imageId)
        self.datasetDescriptors.append(V)
        
    def stopTrain(self):
        self.descriptors = np.asarray(self.datasetDescriptors)
        del(self.datasetDescriptors)
        
        # Create index ball-tree
        self.tree = BallTree(self.descriptors, leaf_size=self.leafSize)
        print("Done training")
        
        
    def computeVlad(self, descriptors):
        predictedLabels = self.dictionary.predict(descriptors)
        centers = self.dictionary.cluster_centers_
        k=self.dictionary.n_clusters
        m,d = descriptors.shape
        V=np.zeros([k,d])

        #computing the differences
        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels==i)>0:
                # add the diferences
                V[i]=np.sum(descriptors[predictedLabels==i,:]-centers[i],axis=0)
        
        V = V.flatten()
        # power normalization, also called square-rooting normalization
        V = np.sign(V)*np.sqrt(np.abs(V))

        # L2 normalization
        V = V/np.sqrt(np.dot(V,V))
        return V
    
    def query(self, image, numOfImages):
        kp, descs = self.orb.detectAndCompute(image, None)
        vl = self.computeVlad(descs)
        vl = vl.reshape((1, vl.shape[0]))
        dist, idx = self.tree.query(vl, numOfImages)
        res = [self.imageIds[i] for i in idx[0]]
        return res

    def save(self, path):
        return mlc.save(self, path)
    
    @staticmethod
    def load(path):
        pass
    
    
