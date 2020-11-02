import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import mlcrate as mlc
import numpy as np


class VisualDictionary():
    def __init__ (self, numWords=64, numFeaturesOnImage=3000):
        self.numWords = numWords
        self.featureDetector = cv2.ORB_create(numFeaturesOnImage)
        self.descriptors = []
        
    def train(self, image):
        keypts, descrs = self.featureDetector.detectAndCompute(image, None)
        self.descriptors.append(descrs)
    
    def build(self):
        self.descriptors = np.array(descriptors).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        compactness, self.bestLabels, self.centers = cv2.kmeans(self.descriptors, self.numWords, None, criteria)
    
    def predict(self, descriptors):
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
    
    
if __name__=='__main__':

    from RandomAccessBag import ImageBag
    
    trainBag=ImageBag('/media/sujiwo/PlaceRecognition/sources/train.bag', '/front_rgb/image_raw')
    sampleList = [0, 14, 17, 20, 21, 24, 27, 30, 115, 134, 147, 275, 440, 460, 467, 520, 526, 541, 557, 588, 787, 985, 1182, 1184, 1186, 1188, 1190, 1192, 1221, 1225, 1227, 1229, 1232, 1236, 1280, 1320, 1412, 1503, 1538, 1542, 1589, 1745, 1753, 1791, 1919, 1967, 2004, 2026, 2041, 2117, 2154, 2292, 2428, 2554, 2642, 2714, 2752, 2875, 3014, 3199, 3355, 3370, 3379, 3386, 3392, 3411, 3414, 3415, 3419, 3435, 3447, 3454, 3464, 3467, 3470, 3473, 3476, 3479, 3545, 3547, 3549, 3551, 3552, 3554, 3557, 3559, 3560, 3561, 3567, 3578, 3585, 3662, 3669, 3686, 3702, 3805, 3872, 3886, 3912, 3941, 3942, 3951, 3955, 3974, 4107, 4121, 4134, 4166, 4276, 4372, 4467, 4570, 4805, 5060, 5384, 5401, 5448, 5495, 5542, 5570, 5608, 6474, 6677, 6799, 6809, 6813, 6818, 7050, 7089, 7120, 7150, 7289, 7594, 7657, 7708, 8011, 8036, 8049, 8199, 8285, 8365, 8445, 8520, 8594, 8668, 8739, 8814, 8887, 8961, 9064, 9175, 9577, 9717, 9827, 9902, 9927, 10002, 10077, 10090, 10107, 10184, 10379, 10686, 10733, 10780, 10828, 10831, 10878, 10925, 10935, 11029, 11054, 11078, 11082, 11116, 11254, 11326, 11419, 11484, 11569, 11651, 11743, 11848, 11880, 11977, 12054, 12126, 12205, 12285, 12315, 12344, 12396, 12553, 12685, 12818, 12899, 13043, 13160, 13264, 13373, 13486, 13611, 13755, 13874, 13988, 14085, 14171, 14250, 14321, 14391, 14482, 14595, 14622, 14650, 14690, 14822, 14925, 15020, 15110, 15199, 15291, 15383, 15467, 15545, 15625, 15712, 15836, 15889, 15929, 16033, 16154, 16343, 16360, 16368, 16376, 16386, 16608, 16733, 16765, 16772, 16781, 16962, 17037, 17149, 17239, 17319, 17396, 17474, 17555, 17635, 17711, 17789, 17842, 17924, 18001, 18076, 18157, 18237, 18311, 18387, 18464, 18539, 18615, 18688, 18768, 18889, 19055, 19194, 19316, 19334, 19355, 19365, 19455, 19554, 19638, 19716, 19798, 19911, 20054, 20084, 20182, 20461, 20476, 20488, 20654, 20747, 20826, 20900, 20978, 21079, 21148, 21172, 21201, 21325, 21413, 21494, 21572, 21646, 21717, 21785, 21851, 21915, 21978, 22044, 22120, 22205, 22285, 22362, 22428, 22492, 22560, 22630, 22701, 22772, 22854, 23243, 23771, 23810, 23954, 23988, 24025, 24136, 24219, 24293, 24366, 24437, 24506, 24573, 24641, 24710, 24778, 24846, 24912, 24978, 25046, 25114, 25182, 25253, 25324, 25393, 25469, 25596, 25747, 25873, 25915, 25942, 25974, 26004, 26031, 26157, 26252, 26335, 26410, 26482, 26568, 26699, 26745, 26766, 26786, 26790, 26793, 26796, 26797, 26798, 26799, 26800, 26869, 26938, 26942, 26956, 26990, 27117, 27230, 27329, 27387, 27400, 27433, 27442, 27451, 27456, 27480, 27487, 27493, 27529, 27571, 27645, 27851, 27957, 28057, 28165, 28233, 28256, 28290, 28367, 28662, 28874, 29287, 29436, 29564, 29674, 29795, 30016, 30038, 30063, 30095, 30329, 30389, 30601, 30784, 30910, 31008, 31158, 31283, 31384, 31514, 31770, 31846, 31894, 31924, 32011, 32014, 32026, 32038, 32107, 32141, 32270, 32358, 32452, 32553, 32594, 32618, 32621, 32633, 32720, 32737, 32747, 32753, 32782, 32802, 32808, 32811, 32816, 32883, 32901, 33052, 33057, 33073, 33082, 33088, 33102, 33104, 33106, 33111, 33114, 33117, 33118, 33137, 33143]
    mvlad = VLAD()
    mvlad.loadDictionary("/home/sujiwo/visualDictionary.pickle")
    mvlad.initTrain()
    for s in sampleList:
        img=trainBag[s]
        mvlad.train(s, img)
        print(s)
    mvlad.stopTrain()
    imageQuery = cv2.imread("/home/sujiwo/ouster-1.png")
    az = mvlad.query(imageQuery, 5)
    print(az)
    
    pass