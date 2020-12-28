#!/usr/bin/python


from RandomAccessBag import RandomAccessBag
from tqdm import tqdm
from geodesy import utm
import numpy as np
import rospy
from rospy import Time, Duration
from bisect import bisect_left


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
    def _parseFromNavSatFix(randomBag, eastingShift=0.0, northingShift=0.0, heightShift=0.0):
        assert(randomBag.type()=='sensor_msgs/NavSatFix')
        parsedCoordinates = np.zeros((len(randomBag),3), dtype=np.float)
        timestamps = []
        i = 0
        for rawmsg in tqdm(randomBag):
            coord = utm.fromLatLong(rawmsg.latitude, rawmsg.longitude, rawmsg.altitude)
            parsedCoordinates[i,:] = [coord.easting+eastingShift, coord.northing+northingShift, coord.altitude+heightShift]
            timestamps.append(rawmsg.header.stamp)
            i+=1
        return timestamps, parsedCoordinates
        


if __name__=='__main__' :
    tgbag = RandomAccessBag('/Data/MapServer/Logs/log_2016-12-26-13-21-10.bag', '/nmea_sentence')
    ps = GeographicTrajectory._parseFromNmea(tgbag)
    pass
    
    
    