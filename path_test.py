# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:27:42 2024

@author: Maxwell
"""
import os
import glob

home = True
if home:
    path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks"
else:
    path = os.getcwd().replace("\\", "/") + "/twin_data"
spectrum_paths_rot0 = glob.glob(path + "/data/isic12_95_gTrue_rot0_/*[0-9].ds")
spectrum_paths_rot0 = [i.replace("\\", "/") for i in spectrum_paths_rot0]
spectrum_paths_rot1 = glob.glob(path + "/data/isic12_95_gTrue_rot1_/*[0-9].ds")
spectrum_paths_rot1 = [i.replace("\\", "/") for i in spectrum_paths_rot1]
input_path = path + "/source_images/isic12_95.ds"

print("spectrum rot0")
for path in spectrum_paths_rot0:
    print(path)
    
print("\nspectrum rot1")
for path in spectrum_paths_rot1:
    print(path)