How to Build Place Recognition Maps and Use Them
===

Overview
--------

`place_recognizer` is a package and C++/Python library for building visual-based place database. It can also be integrated to your own projects; for example, as loop detection assistance in SLAM methods.

This package implements two major methods of visual place recognition: Incremental Bag-of-Words (IBoW) and Vector of Locally Aggregated Descriptors (VLAD). Both methods allow multi-session map building.

Building visual place database requires two ingredients; first is image stream itself, and second, per-frame image metadata. This image metadata will be returned upon query. An example of relevant image metadata is GPS coordinate, but you can supply other types metadata such as LIDAR scans.

Installation
------------

1. Create a new ROS Workspace, and cd to that workspace's directory.

2. Clone the required repositories inside the `src` directory of ROS workspace:

   - rosbag_viewer
   - im_enhance
   - place_recognizer

3. Run `catkin_make install`. 
4. Before running any programs, update the environment variables from `install` directory by:
<pre>
source install/setup.bash
</pre>

## Examples
Except stated otherwise, all examples in this file are written using Python syntax.

### Creating Map File from ROS Bag

The easiest way to create map file from ROS Bag files is by using script `train_from_bag.py` that must be run from bash command prompt. This script requires a ROS bag file that contains at least two data stream:

- Image stream in either sensor_msgs/Image or sensor_msgs/CompressedImage

- GNSS fix in either nmea_msgs/Sentence or sensor_msgs/NavSatFix

An example of script execution (from ROS workspace directory) to create a new map file and stored in `/tmp/result-map.dat` is as follows. The image topic in the bag file is /camera1/image_raw.

```bash
$ ./install/lib/place_recognizer/train_from_bag.py --method ibow /media/user/source.bag /camera1/image_raw /tmp/result-map.dat
```


For more information on script's parameters, call train_from_bag with `-h` parameter.

### Creating Map File from Custom Data Sources

In principle, user must create GenericTrainer object that specifies compression and indexing methods, whether to load previous sessions and path for saving map file. Then this object must initialize mapping using initTrain() function, subsequently add more images and stop the mapping process by using stopTrain().

```python
from place_recognizer import *
import cv2
import glob
import numpy as np

mymap = GenericTrainer(method='ibow', mapfile_output='/tmp/result-map.dat')
mymap.initTrain()

# We assume that images are located in files in same directory, while their positions 
# are placed inside a CSV file.
# You are responsible for determining image metadata that will be supplied and returning 
# upon query
imagePoses = np.loadtxt('/Datasource/coordinates.txt')

i = 0
for file in glob.glob('/Datasource/*.png'):
    image = cv2.imread(file)
    position = imagePoses[i, 0:3]
    mymap.addImage(image, position)
    
mymap.stopTrain()
```

### Loading and Querying Database

To query the database, first create the image database object. 

```python
mymap = GenericImageDatabase('/tmp/result-map.dat')
qImage = cv2.imread('test-image.jpg')
imagePosCandidates = mymap.query(qImage)
```

