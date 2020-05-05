# !/usr/local/biotools/python/3.4.3/bin/python3
__author__ = "Naresh Prodduturi"
__email__ = "prodduturi.naresh@mayo.edu"
__status__ = "Dev"

import openslide
import tensorflow as tf
import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
import glob
# import numpy as np
# from PIL import Image, ImageDraw
# import tempfile
# import math
# import io
# import re
# import matplotlib
# from skimage.filters import threshold_otsu
# from skimage.color import rgb2lab,rgb2hed
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from dataset_utils import *
# from shapely.geometry import Polygon, Point, MultiPoint
# from shapely.geometry import geo
# from descartes.patch import PolygonPatch

gene="BRCA2"
level_svs=2
level_tiff=5
tcga_col=41
mayo_col=26
#svs:(1.0, 4.000079681274901, 16.00159387950271, 32.012071534663015)
#tiff:(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)

#reading the TCGA file
fobj=open("FINAL_TCGA_SVS.xls")
header=fobj.readline()
header_list = header.split("\t")
if not gene in header_list[tcga_col-1]:
    print("gene,",gene,"column header",header_list[tcga_col-1])
    sys.exit(0)
#gene_status="0"
dict_tcga={}
for i in fobj:
    i =i.strip()
    arr = i.split("\t")
    if arr[tcga_col-1] == "1":
        gene_status="1"
    else:
        gene_status="0"
    dict_tcga[arr[0]]=gene_status
#print(dict_tcga["TCGA-B6-A0RG"])
#sys.exit(0)    
#reading the Mayo file
fobj=open("Merge.xls")
header=fobj.readline()
header_list = header.split("\t")
#if not gene in header_list[mayo_col-1]:
    #print("gene,",gene,"column header",header_list[mayo_col-1])
    #sys.exit(0)
num=0
gene_status="0"
dict_mayo={}
for i in fobj:
    i =i.strip()
    arr = i.split("\t")
    if arr[mayo_col-1] == gene:
        num=num+1
        gene_status="1"
    else:
        gene_status="0"
    dict_mayo[arr[6]]=gene_status
    dict_mayo[arr[14]]=gene_status
if num==0:
    print("Mayo num:",num)
    sys.exit(0)
#print(dict_mayo['e70cab296d7044c2bb9412b391b0e389'])
#sys.exit(0)
#svs_file=sys.argv[1]
fobj=open("file_names.txt")
for i in fobj:
    i =i.strip()
    OSobj = openslide.OpenSlide(i)
    ld = OSobj.level_dimensions
    ds = OSobj.level_downsamples
    bn=os.path.basename(i)
    if bn.startswith("TCGA"):
        list = bn.split("-")
        bn1 = list[0]+'-'+list[1]+'-'+list[2]
        genestatus = dict_tcga[bn1]
    else:
        if '.svs' in bn:
            bn1 = bn.replace(".svs","")
            genestatus = dict_mayo[bn1]
        else:
            bn1 = bn.replace(".tiff","")
            bn1 = bn1.replace("_BIG","")
            genestatus = dict_mayo[bn1]
    #print(i,genestatus,ld[0][0],ld[0][1],ds[0],ds[1])
    print(i+"\t"+genestatus)