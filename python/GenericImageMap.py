#!/usr/bin/env python2
import sys
import cv2
from .VLAD import *
from ._place_recognizer import *
import pickle
import tarfile
import uuid
import os
from io import BytesIO
from numpy.random import randint
from place_recognizer._place_recognizer import IncrementalBoW


_numOrbFeatures = 6000

class GenericTrainer(object):
    '''
    Base Class for training/building map file for IncrementalBoW and VLAD
    
    Attributes
    ----------
    numFeatures: number of features to be extracted by ORB descriptor
    initialMask: Image mask to be used by ORB descriptor
    resize_factor: Resize image frame by this factor
    show_image_frame: whether to show incoming image frame after preprocessed 
    
    Parameters
    ----------
    method: str, "ibow" or "vlad"
    mapfile_output: str, path to resulting map file
    mapfile_load: str, path to existing map file to be loaded
    vdictionaryPath: str, path to visual dictionary (only for vlad)
    '''

    # Number of features to be extracted from single image
    numFeatures = _numOrbFeatures
    # Initial masks for feature extraction
    mask = None
    initialMask = None
    # Reduce image size with this factor
    resize_factor = 0.53333
    # Whether to show image frames
    show_image_frame = True
    
    def __init__(self, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        """
        Initialization
        
        """
        self.method = method
        self.mapfile_output = mapfile_output
        self.imageMetadata = []
        
        # Prepare the map
        if mapfile_load is not None:
            self.mapper, self.imageMetadata = GenericTrainer.loadMap(mapfile_load)
        else:
            if method=="vlad":
                try:
                    vdict = VisualDictionary.load(vdictionaryPath)
                except:
                    raise ValueError("Unable to load visual dictionary for VLAD method")
                self.mapper = VLAD2(vdict)
            elif method=="ibow":
                self.mapper = IncrementalBoW()
                
        self.createFeatureExtractor()
        
        # Image ID
        self.imageIdNext = self.mapper.lastImageId()
        
    def createFeatureExtractor(self):
        # Feature extractor
        self.extractor = cv2.ORB_create(self.numFeatures)
    
    def preprocess(self, image):
        """
        Preprocess input image frame prior to train. You may need to override this function
        in order to customize training process; eg. add segmentation or enhance the contrast
        
        This function may do two things:
        - modify image prior to feature extraction
        - generate masks for feature extraction
        """
        imgprep = cv2.resize(image, (0,0), None, fx=self.resize_factor, fy=self.resize_factor)
        return imgprep
    
    def addImage(self, image, imageMetadata=None):
        '''
        Add an image to the map, with additional metadata
        
        Parameters
        ----------
        image: np.ndarray, image frame
        imageMetadata: metadata about the image that will be returned upon query (must be pickleable)
        '''
        image_prep = self.preprocess(image)
        keypoints, descriptors = self.extractor.detectAndCompute(image_prep, self.mask)
        
        self.mapper.addImage(descriptors, keypoints, self.imageIdNext)
        self.imageMetadata.append(imageMetadata)
        self.imageIdNext += 1

        if self.show_image_frame:
            cv2.imshow("Image", image_prep)
            cv2.waitKey(1)

    def initTrain(self):
        """
        Initialize a new training session
        """
        self.mapper.initTrain()
    
    def stopTrain(self):
        """
        Stop current training session and save resulting map to disk
        """
        self.mapper.stopTrain()
        self.save(self.mapfile_output)
        
    def save(self, filepath):
        
        '''
        Map and metadata is saved separately but joined in a TAR archive
        '''
        prtar = tarfile.TarFile(filepath, "w")
        
        # Header
        header = {
                'method': self.method,
                'initialMask': self.initialMask
            }
        headerIo = BytesIO()
        headerInfo = tarfile.TarInfo(name="header.dat")
        pickle.dump(header, headerIo, protocol=2)
        headerInfo.size = int(headerIo.tell())
        headerIo.seek(0)
        prtar.addfile(tarinfo=headerInfo, fileobj=headerIo)
        
        if self.method=='vlad':
            mapIo = BytesIO()
            mapInfo = tarfile.TarInfo(name="map.dat")
            self.mapper.save(mapIo)
            mapInfo.size = int(mapIo.tell())
            mapIo.seek(0)
            prtar.addfile(tarinfo=mapInfo, fileobj=mapIo)
            ibowMapTmpName = None
        else:
            # IncrementalBoW does not support saving to file descriptor
            randInt = str(randint(10000, 99999))
            ibowMapTmpName = os.path.join(os.path.dirname(os.path.realpath(filepath)), 'map'+randInt+'.int')
            self.mapper.save(ibowMapTmpName)
            prtar.add(ibowMapTmpName, arcname='map.dat')
            print("IBoW map saved")
        
        metadataIo = BytesIO()
        metadataInfo = tarfile.TarInfo(name="metadata.dat")
        pickle.dump(self.imageMetadata, metadataIo, protocol=2)
        metadataInfo.size = int(metadataIo.tell())
        metadataIo.seek(0)
        prtar.addfile(tarinfo=metadataInfo, fileobj=metadataIo)
        
        prtar.close()
        if (ibowMapTmpName is not None):
            os.remove(ibowMapTmpName)
        
    @staticmethod
    def loadMap(filepath):
        '''
        Load map from file on disk
         
        Parameters
        ----------
        :param path: str, path to file on disk
         
        Returns
        -------
        mapper: Mapper object
        imageMetadata: Image stream Metadata (typically geographic positions)
        '''
        prtar = tarfile.TarFile(filepath, "r")
        
        headerInfo, mapInfo, metadataInfo = prtar.getmembers()
        
        headerIo = prtar.extractfile(headerInfo)
        header = pickle.load(headerIo)

        if (header['method']=='vlad'):
            mapIo = prtar.extractfile(mapInfo)
            mapObj = VLAD2.load(mapIo)
        else:
            import tempfile
            mapObj = IncrementalBoW()
            mapInfo.name = str(uuid.uuid4())
            ibowMapTmpName = os.path.join(tempfile.gettempdir(), mapInfo.name)
            prtar.extract(mapInfo, path=tempfile.tempdir)
            mapObj.load(ibowMapTmpName)
            os.remove(ibowMapTmpName)
            pass
        
        metadataIo = prtar.extractfile(metadataInfo)    
        metadata = pickle.load(metadataIo)
        
        prtar.close()
        return mapObj, metadata
        

class GenericImageDatabase(GenericTrainer):
    def __init__(self, mapfile_load):
        self.mapper, self.imageMetadata = GenericImageDatabase.loadMap(mapfile_load)
        self.createFeatureExtractor()
        
    def initTrain(self):
        raise RuntimeError("Not implemented")
    
    def stopTrain(self):
        raise RuntimeError("Not implemented")
    
    def addImage(self, image, imageMetadata=None):
        raise RuntimeError("Not implemented")

    def query(self, image, numOfImages=5, indicesOnly=False):
        '''
        Search the map using an image
        '''
        imageprep = self.preprocess(image)
        keypoints, descriptors = self.extractor.detectAndCompute(imageprep, self.mask)
        indices = self.mapper.query(descriptors, numOfImages)
        
        if indicesOnly: return indices
        else: return [self.imageMetadata[i] for i in indices]
        
        
