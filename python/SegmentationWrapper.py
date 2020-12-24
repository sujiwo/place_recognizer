#!/usr/bin/python


_hasSegment = False

try:
    import caffe
    _hasSegment = True
    caffe.set_mode_gpu()
except ImportError:
    print("Unable to import Caffe")
    
def CreateMask(image):
    pass
    
    
if __name__=="__main__" and _hasSegment==True:
    print(caffe.__version__)
    print("Caffe available")