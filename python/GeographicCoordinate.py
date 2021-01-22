#!/usr/bin/python


from RandomAccessBag import RandomAccessBag
from tqdm import tqdm
from geodesy import utm
import numpy as np
import rospy
from rospy import Time, Duration
from bisect import bisect_left
import math
from tf import transformations as tfx


_startTime0 = rospy.Time(0)
_stopTime_1 = rospy.Time(0)


class GeographicTrajectory:
    """
    This class represents position of vehicle in time, acquired from GNSS.
    It supports two types of message:
    - nmea_msgs/Sentence
    - sensor_msgs/NavSatFix
    
    Attributes
    ----------
    timestamps : list of rospy.Time
        List of discrete timestamp at which pose is recorded
    coordinates : np.ndarray
        Position of vehicle at each timestamp
    duration : rospy.Duration
        Length of position recording
    frame_id : str
        Frame ID of coordinate sensor
    """
    
    supportedMsgTypes = ['sensor_msgs/NavSatFix', 'nmea_msgs/Sentence']
    
    def __init__ (self, randomBag=None, eastingShift=0.0, northingShift=0.0, heightShift=0.0):
        if (randomBag is None):
            self.timestamps = []
            self.coordinates = []
            self.duration = Duration(0)
            return
        
        if (randomBag.type()=='sensor_msgs/NavSatFix'):
            self.timestamps, self.coordinates = GeographicTrajectory._parseFromNavSatFix(randomBag, eastingShift, northingShift, heightShift)
        elif (randomBag.type()=='nmea_msgs/Sentence'):
            self.timestamps, self.coordinates = GeographicTrajectory._parseFromNmea(randomBag, eastingShift, northingShift, heightShift)
        else:
            raise ValueError("Input bag is of unknown type")
        self.duration = self.timestamps[-1] - self.timestamps[0]
        self.frame_id = randomBag[0].header.frame_id
        
    def __len__(self):
        return len(self.timestamps)
        
    def positionAt(self, time):
        """
        Returns position at requested time using interpolation
        """
        if (isinstance(time, float)):
            if (time < 0 or time > self.duration.to_sec()):
                raise ValueError("Offset value is outside bag timestamp range")
            time = self.timestamps[0] + Duration.from_sec(time)
        elif (isinstance(time, Time)):
            if (time < self.timestamps[0] or time > self.timestamps[-1]):
                raise ValueError("Timestamp value is outside the bag range")
        
        _t1 = bisect_left(self.timestamps, time)
        if _t1==len(self.timestamps)-1:
            _t1 = len(self.timestamps)-2
        t1 = self.timestamps[_t1]
        t2 = self.timestamps[_t1+1]
        r = ((time-t1).to_sec()) / (t2-t1).to_sec()
        return self.coordinates[_t1] + (self.coordinates[_t1+1] - self.coordinates[_t1])*r
        
    def buildFromTimestamps(self, timestampList):
        """
        Creates new trajectory based on current one using interpolation of each timestamp
        """
        track = GeographicTrajectory()
        for t in tqdm(timestampList):
            if t<=self.timestamps[0]:
                track.coordinates.append(self.coordinates[0])
            elif t>=self.timestamps[-1]:
                track.coordinates.append(self.coordinates[-1])
            else:
                pos = self.positionAt(t)
                track.coordinates.append(pos)
            track.timestamps.append(t)
        track.duration = self.timestamps[-1] - self.timestamps[0]
        track.coordinates = np.array(track.coordinates)
        track.frame_id = self.frame_id
        return track
            
    
    @staticmethod
    def parseFromBag(randomBag):
        pass
    
    @staticmethod
    def _parseFromNmea(randomBag, eastingShift=0.0, northingShift=0.0, heightShift=0.0):
        try:
            import pynmea2
        except ImportError:
            raise RuntimeError("NMEA support is not available; install pynmea2")
        
        parsedCoordinates = []
        timestamps = []
        i = 0
        for rawmsg in tqdm(randomBag):
            i += 1
            try:
                m = pynmea2.parse(rawmsg.sentence)
                coord = utm.fromLatLong(float(m.latitude), float(m.longitude), float(m.altitude))
                parsedCoordinates.append([coord.easting, coord.northing, coord.altitude])
                timestamps.append(rawmsg.header.stamp)
            except:
                continue
        parsedCoordinates = np.array(parsedCoordinates)
        return timestamps, parsedCoordinates
                
    @staticmethod
    def _parseFromNavSatFix(randomBag, eastingShift=0.0, northingShift=0.0, heightShift=0.0, startTime=_startTime0, stopTime=_stopTime_1):
        assert(randomBag.type()=='sensor_msgs/NavSatFix')
        timestamps = []
        i = 0
        if startTime==_startTime0 and stopTime==_stopTime_1:
            msgSamples = range(len(randomBag))
        else:
            msgSamples = randomBag.desample(-1, True, startTime, stopTime)
        parsedCoordinates = np.zeros((len(msgSamples),7), dtype=np.float)
        
        for s in tqdm(msgSamples):
            rawmsg = randomBag[s]
            coord = utm.fromLatLong(rawmsg.latitude, rawmsg.longitude, rawmsg.altitude)
            parsedCoordinates[i,0:3] = [coord.easting+eastingShift, coord.northing+northingShift, coord.altitude+heightShift]
            if i>=1:
                quat = GeographicTrajectory.orientationFromPositionOnlyYaw(parsedCoordinates[i], parsedCoordinates[i-1])
                parsedCoordinates[i,3:7] = quat
                if i==1:
                    parsedCoordinates[0,3:7] = quat
            timestamps.append(rawmsg.header.stamp)
            i+=1
        return timestamps, parsedCoordinates
        
    @staticmethod
    def orientationFromPositionOnlyYaw(curPosition, prevPosition):
        yaw = math.atan2(curPosition[1]-prevPosition[1], curPosition[0]-prevPosition[0])
        return tfx.quaternion_from_euler(0, 0, yaw)
        

if __name__=='__main__' :
    tgbag = RandomAccessBag('/Data/MapServer/Logs/log_2016-12-26-13-21-10.bag', '/nmea_sentence')
    ps = GeographicTrajectory._parseFromNmea(tgbag)
    pass
    
    
    