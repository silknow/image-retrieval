# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:20:00 2020

@author: clermont
"""

import sys
sys.path.insert(0,'./src') 
import silknow_image_retrieval as sir

if __name__ == '__main__':
    configFile = sys.argv[1]
    sir.get_kNN_configfile(configFile)