#!/usr/bin/python

from PyQt5 import QtWidgets, uic
import rospkg
from os import path
import sys


class RatioLayoutedFrame(QtWidgets.QFrame):
    pass


class OxfordDatasetViewer(QtWidgets.QMainWindow):
    
    def __init__(self, dataset):
        self.dataset = dataset
        ui_file = path.join(rospkg.RosPack().get_path('rosbag_viewer'), 'GenericImagesetViewer.ui')
        super(OxfordDatasetViewer, self).__init__()
        uic.loadUi(ui_file, self)
        self.show()
        self.app = QtWidgets.QApplication(sys.argv)
        self.app.exec_()