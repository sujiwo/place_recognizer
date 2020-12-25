#!/usr/bin/python


from RandomAccessBag import RandomAccessBag
from tqdm import tqdm
from geodesy import utm
import numpy as np


class GeographicTrajectory:
    """
    This class represents position of vehicle in time, acquired from GNSS.
    It supports two types of message:
    - nmea_msgs/Sentence
    - sensor_msgs/NavSatFix
    """
    
    def __init__ (self):
        pass
    
    @staticmethod
    def parseFromBag(randomBag):
        pass
    
    @staticmethod
    def _parseFromNmea(randomBag):
        try:
            import pynmea2
        except ImportError:
            raise RuntimeError("NMEA support is not available; install pynmea2")
        
        parsedCoordinates = []
        for rawmsg in tqdm(randomBag):
            try:
                m = pynmea2.parse(rawmsg.sentence)
                coord = utm.fromLatLong(float(m.lat), float(m.lon), float(m.altitude))
                parsedCoordinates.append([coord.easting, coord.northing, coord.altitude])
            except:
                continue
        parsedCoordinates = np.array(parsedCoordinates)
        return parsedCoordinates
                
    @staticmethod
    def _parseFromNavSatFix(randomBag, eastingShift=0.0, northingShift=0.0, heightShift=0.0):
        assert(randomBag.type()=='sensor_msgs/NavSatFix')
        parsedCoordinates = np.zeros((len(randomBag),3), dtype=np.float)
        i = 0
        for rawmsg in tqdm(randomBag):
            coord = utm.fromLatLong(rawmsg.latitude, rawmsg.longitude, rawmsg.altitude)
            parsedCoordinates[i,:] = [coord.easting+eastingShift, coord.northing+northingShift, coord.altitude+heightShift]
            i+=1
        return parsedCoordinates
        


if __name__=='__main__' :
    pass