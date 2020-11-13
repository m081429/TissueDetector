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
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import math
import io
import re
import matplotlib
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab,rgb2hed
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dataset_utils import *
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import geo
from descartes.patch import PolygonPatch
import xml.etree.ElementTree as ET
from xml.dom import minidom

'''function to check if input files exists and valid'''


def input_file_validity(file):
    '''Validates the input files'''
    if os.path.exists(file) == False:
        raise argparse.ArgumentTypeError('\nERROR:Path:\n' + file + ':Does not exist')
    if os.path.isfile(file) == False:
        raise argparse.ArgumentTypeError('\nERROR:File expected:\n' + file + ':is not a file')
    if os.access(file, os.R_OK) == False:
        raise argparse.ArgumentTypeError('\nERROR:File:\n' + file + ':no read access ')
    return file


def argument_parse():
    '''Parses the command line arguments'''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-p", "--patch_dir", help="Patch dir", required="True")
    parser.add_argument("-i", "--input_file", help="input file", required="True")
    parser.add_argument("-o", "--tf_output", help="output tf dir", required="True")
    parser.add_argument("-s", "--patch_size", help="Patch_size", required="True")
    parser.add_argument("-t", "--threshold", help="background Threshold pixel cutoff", required="True")
    parser.add_argument("-a", "--threshold_area_percent", help="background Threshold pixel cutoff area percent",required="True")
    parser.add_argument("-m", "--threshold_mean", help="background Threshold mean cutoff", required="True")
    parser.add_argument("-d", "--threshold_std", help="background Threshold std cutoff", required="True")
    parser.add_argument("-l", "--level", help="level", required="True")
    parser.add_argument("-x", "--rgb2lab_thresh", help="level", required="True")
    parser.add_argument("-z", "--patch_byte_cutoff", help="patch_byte_cutoff", required="True")
    parser.add_argument("-c", "--mut_type", help="mut type", required="True")
    parser.add_argument("-A", "--ann", help="annotation", required="True")
    
    return parser




def create_patch_deprecated(svs, patch_sub_size, patch_dir, samp, tf_output, patch_level, threshold, threshold_area_percent,
                 threshold_mean, threshold_std):
    # print(svs+' '+str(patch_sub_size)+' '+patch_dir+' '+samp+' '+str(patch_level)+' '+tf_output)
    tf_writer = tf.python_io.TFRecordWriter(tf_output + '/' + samp + '.tfrecords')
    # threshold=240
    # level=2
    # threshold=242
    level = patch_level
    OSobj = openslide.OpenSlide(svs)
    minx = 0
    miny = 0
    tmp = OSobj.level_dimensions[level]
    maxx = tmp[0]
    maxy = tmp[1]
    # this factor if required to convert level0 start coordinatess to level2 start coordinates (this is required for OSobj.read_region function)
    multi_factor = OSobj.level_downsamples[level]
    # print(svs+' '+str(patch_sub_size)+' '+patch_dir+' '+str(maxx))
    start_x = minx
    '''creating sub patches'''
    '''Iterating through x coordinate'''
    current_x = 0
    filenames = []
    # num=0
    # while start_x < maxx:
    while start_x + patch_sub_size < maxx:
        '''Iterating through y coordinate'''
        current_y = 0
        start_y = miny
        # while start_y < maxy:
        while start_y + patch_sub_size < maxy:
            tmp_start_x = int(round(start_x * multi_factor, 0))
            tmp_start_y = int(round(start_y * multi_factor, 0))
            try:
                img_patch = OSobj.read_region((tmp_start_x, tmp_start_y), level, (patch_sub_size, patch_sub_size))
            except:
                sys.exit(0)
            # img_patch = OSobj.read_region((start_x,start_y), level, (maxx, maxy))
            # num=num+1
            # img_patch.save(patch_dir+'/'+str(num)+'.png', "png")
            # sys.exit(1)
            np_img = np.array(img_patch)
            # max_min_channels = [np.amax(np_img[:,:,0]), np.amax(np_img[:,:,1]), np.amax(np_img[:,:,2]), np.amin(np_img[:,:,0]), np.amin(np_img[:,:,1]), np.amin(np_img[:,:,2])]
            im_sub = Image.fromarray(np_img)
            width, height = im_sub.size
            '''Change to grey scale'''
            grey_img = im_sub.convert('L')
            '''Convert the image into numpy array'''
            np_grey = np.array(grey_img)
            patch_mean = round(np.mean(np_grey), 2)
            patch_std = round(np.std(np_grey), 2)
            # patch_max=round(np.amax(np_grey),2)
            # patch_min=round(np.amin(np_grey),2)
            '''Identify patched where there are tissues'''
            '''tuple where first element is rows, second element is columns'''
            idx = np.where(np_grey < threshold)
            # print(len(idx[0]))
            # print(np_grey.shape[0]*np_grey.shape[1])
            patch_area = len(idx[0]) / (np_grey.shape[0] * np_grey.shape[1])

            '''proceed further only if patch has non empty values'''
            # if len(idx[0])>0 and len(idx[1])>0 and width==patch_sub_size and height==patch_sub_size:
            # num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)+"_mean_"+str(patch_mean)+"_patch_std_"+str(patch_std)+"_patch_area_"+str(patch_area)
            num_patch = samp + "_X_" + str(start_x) + "_" + str(start_x + patch_sub_size) + "_Y_" + str(
                start_y) + "_" + str(start_y + patch_sub_size)
            if patch_area > threshold_area_percent and patch_mean < threshold_mean and patch_std > threshold_std and width == patch_sub_size and height == patch_sub_size:
                # if	threshold_area> threshold_area_percent and patch_std>5:
                # if width==patch_sub_size and height==patch_sub_size:
                # print("sucess")
                '''creating patch name'''
                # num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)
                num_patch = num_patch + "_included"
                # filenames.append(num_patch)
                # tmp_png=patch_dir+'/'+num_patch+'.png'
                # tmp_png=patch_dir+'/'+"included_mean_"+str(patch_mean)+"patch_std_"+str(patch_std)+"_"+num_patch+'.png'
                # else:
                # '''creating patch name'''
                # num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)
                # filenames.append(num_patch)
                # tmp_png=patch_dir+'/'+"excluded_mean_"+str(patch_mean)+"patch_std_"+str(patch_std)+"_"+num_patch+'.png'

                # '''saving image'''
                # im_sub.save(patch_dir+'/'+num_patch+".png", "png")
                # sys.exit(1)
                image_format = "png"
                height = patch_sub_size
                width = patch_sub_size
                image_name = num_patch

                sub_type = 2
                # if p[1] == "TCGA-LUAD":
                # sub_type=0
                # if	p[1] == "TCGA-LUSC":
                # sub_type=1
                # sub_type=p[1]
                if 'BRAF' in samp:
                    sub_type = 1
                else:
                    sub_type = 0
                mut_type = ""
                if 'WT' in samp:
                    mut_type = 'WT'
                elif 'BRAF_V600E' in samp:
                    mut_type = 'BRAF_V600E'
                elif 'BRAF_V600K' in samp:
                    mut_type = 'BRAF_V600K'
                elif 'BRAF_V600NENK' in samp:
                    mut_type = 'BRAF_V600NENK'
                elif 'BRAF_V600X' in samp:
                    mut_type = 'BRAF_V600X'
                else:
                    mut_type = 'Normal'
                imgByteArr = io.BytesIO()
                im_sub.save(imgByteArr, format='PNG')
                imgByteArr = imgByteArr.getvalue()
                record = image_to_tfexample_braf(imgByteArr, image_format, int(height), int(width), image_name,
                                                 sub_type, mut_type)
                tf_writer.write(record.SerializeToString())
            filenames.append(num_patch)
            start_y = start_y + patch_sub_size
            current_y = current_y + patch_sub_size
        start_x = start_x + patch_sub_size
        current_x = current_x + patch_sub_size
    # sys.exit(1)
    tf_writer.close()
    return filenames


def create_binary_mask_new(rgb2lab_thresh,svs_file,patch_dir,samp):			
    #img = Image.open(top_level_file_path)
    OSobj = openslide.OpenSlide(svs_file)
    #toplevel=OSobj.level_count-1
    #patch_sub_size_x=OSobj.level_dimensions[toplevel][0]
    #patch_sub_size_y=OSobj.level_dimensions[toplevel][1]
    #img = OSobj.read_region((0,0), toplevel, (patch_sub_size_x, patch_sub_size_y))
    divisor = int(OSobj.level_dimensions[0][0]/500)
    patch_sub_size_x=int(OSobj.level_dimensions[0][0]/divisor)
    patch_sub_size_y=int(OSobj.level_dimensions[0][1]/divisor)
    img = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    toplevel=[patch_sub_size_x,patch_sub_size_y,divisor]
    img = img.convert('RGB')
    np_img = np.array(img)
    #binary_img= (np_img==[254,0,0] or np_img==[255,0,0]).all(axis=2)
    #binary_img=binary_img.astype(int)
    #print(binary_img.shape)
    #np_img[binary_img == 1] = [255, 255, 255]
    #img = Image.fromarray(np_img)
    img.save(patch_dir+'/'+samp+"_original.png", "png")
    #sys.exit(0)
    # lab_img = rgb2lab(np_img)
    # l_img = lab_img[:, :, 0]
    # patch_max=round(np.amax(l_img),2)
    # patch_min=round(np.amin(l_img),2)
    # print(patch_min,patch_max)
    # print(l_img)
    # #
    # lab_img = rgb2hed(np_img)
    # l_img = lab_img[:, :, 0]
    # patch_max = round(np.amax(l_img), 2)
    # patch_min = round(np.amin(l_img), 2)
    # print(patch_min, patch_max)
    # print(l_img)
    # #
    # lab_img = rgb2hed(np_img)
    # l_img = lab_img[:, :, 2]
    # patch_max = round(np.amax(l_img), 2)
    # patch_min = round(np.amin(l_img), 2)
    # print(patch_min, patch_max)
    # print(l_img)
    #
    lab_img = rgb2hed(np_img)
    l_img = lab_img[:, :, 1]
    patch_max = round(np.amax(l_img), 2)
    patch_min = round(np.amin(l_img), 2)
    #print(patch_min, patch_max)
    #print(l_img)
    #rgb2lab_thresh=0.18
    binary_img = l_img >float(rgb2lab_thresh)
    binary_img=binary_img.astype(int)
    np_img[binary_img == 0] = [0, 0, 0]
    np_img[binary_img == 1] = [255, 255, 255]
    im_sub = Image.fromarray(np_img)
    im_sub.save(patch_dir+'/'+samp+"_mask.png", "png")
    #sys.exit(0)
    idx = np.sum(binary_img)
    mask_area = idx / (binary_img.size)
    print(mask_area)
    idx = np.where(binary_img == 1)
    list_binary_img = []
    for i in range(0,len(idx[0]),1):
        x=idx[1][i]
        y=idx[0][i]
        list_binary_img.append(str(x)+' '+str(y))

    #return np.array(binary_img)
    return list_binary_img,toplevel


def calc_patches_cord(list_binary_img,patch_level,svs_file,patch_dir,samp,patch_size,threshold_area_percent,toplevel):
    patch_start_x_list = []
    patch_stop_x_list = []
    patch_start_y_list = []
    patch_stop_y_list = []
    #bin_mask_level=toplevel
    #print(bin_mask_level)
    #print(dict_properties['increment'])
    #sys.exit(0)
    OSobj = openslide.OpenSlide(svs_file)
    minx = 0
    miny = 0
    if patch_level > len(OSobj.level_dimensions)-1:
        print("not enough levels "+str(patch_level)+" "+str(len(OSobj.level_dimensions)-1))
        sys.exit(0)
    maxx = OSobj.level_dimensions[patch_level][0]
    maxy = OSobj.level_dimensions[patch_level][1]
    start_x = minx
    total_num_patches=0
    selected_num_patches=0

    '''creating sub patches'''
    '''Iterating through x coordinate'''
    while start_x + patch_size < maxx:
        '''Iterating through y coordinate'''
        start_y = miny
        while start_y + patch_size < maxy:
            current_x=int((start_x*OSobj.level_downsamples[patch_level])/toplevel[2])
            current_y=int((start_y*OSobj.level_downsamples[patch_level])/toplevel[2])
            tmp_x = start_x  + int(patch_size)
            tmp_y = start_y  + int(patch_size)
            current_x_stop=int((tmp_x*OSobj.level_downsamples[patch_level])/toplevel[2])
            current_y_stop=int((tmp_y*OSobj.level_downsamples[patch_level])/toplevel[2])
            total_num_patches=total_num_patches+1
            #flag=0
            #for m in range(current_x,current_x_stop+1):
            #    for n in range(current_y,current_y_stop+1):
            #        if str(m)+' '+str(n) in list_binary_img:
            #            flag=1
            #poly=Polygon([(current_x, current_y), (current_x_stop, current_y), (current_x_stop, current_y_stop), (current_x, current_y_stop), (current_x, current_y)])
            #if tmp_x <= dict_properties['x_stop'][0] and tmp_y <= dict_properties['y_stop'][0] and ((str(current_x)+' '+str(current_y) in list_binary_img) or (str(current_x_stop)+' '+str(current_y) in list_binary_img) or (str(current_x)+' '+str(current_y_stop) in list_binary_img) or (str(current_x_stop)+' '+str(current_y_stop) in list_binary_img)):
            #print(current_x,current_x_stop,current_y,current_y_stop)
            #print(start_x,tmp_x,start_y,tmp_y)
            flag_list = [1 for i in range(current_x,current_x_stop+1) for j in range(current_y,current_y_stop+1) if str(i)+' '+str(j) in list_binary_img]
            #print(flag_list)
            #sys.exit(0)
            if tmp_x <= maxx and tmp_y <= maxy and (len(flag_list)/((current_y_stop+1-current_y)*(current_x_stop+1-current_x)))>threshold_area_percent:
                patch_start_x_list.append(start_x)
                patch_start_y_list.append(start_y)
                patch_stop_x_list.append(tmp_x)
                patch_stop_y_list.append(tmp_y)
                selected_num_patches=selected_num_patches+1
            #print(start_x,start_y,current_x,current_y)
            start_y = tmp_y
        start_x = tmp_x
    print(selected_num_patches,total_num_patches)
    # print(patch_start_x_list)
    # print(patch_stop_x_list)
    # print(patch_start_y_list)
    # print(patch_stop_y_list)
    # sys.exit(0)
    return patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list

def create_summary_img(patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list,samp,patch_dir,toplevel,patch_level,svs_file):
    #bin_mask_level = toplevel
    OSobj = openslide.OpenSlide(svs_file)
    poly_included = []
    poly_excluded = []
    name = ""
    for i in range(0,len(patch_stop_x_list),1):
        x1=int((patch_start_x_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        x2=int((patch_stop_x_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        y1=int((patch_start_y_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        y2=int((patch_stop_y_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        poly_included.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))

    patch_sub_size_x=toplevel[0]
    patch_sub_size_y=toplevel[1]
    #img_patch = OSobj.read_region((0,0), toplevel, (patch_sub_size_x, patch_sub_size_y))
    img_patch = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    np_img = np.array(img_patch)
    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img_patch)

    for j in range(0, len(poly_included)):
        patch1 = PolygonPatch(poly_included[j], facecolor=[0, 0, 0], edgecolor="green", alpha=0.3, zorder=2)
        ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir , samp + "_mask_patchoverlay.png"), pad_inches='tight')
    return True

def create_tfrecord(patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list,samp,patch_dir,patch_level,svs_file,toplevel,tf_output,patch_size,mut_type,threshold_mean,threshold_std,patch_byte_cutoff,xml_ann):
    '''Reading xml annotations'''
    divisor=toplevel[2]
    xml = minidom.parse(xml_ann)
    # The first region marked
    regions_ = xml.getElementsByTagName("Region")
    regions, region_labels = [], []
    region_type_label = []
    #finalcoords = np.array([])
    x_cor=1
    for region in regions_:
        vertices = region.getElementsByTagName("Vertex")
        r_label = region.getAttribute('Id')
        type_label = region.getAttribute('GeoShape')
        #print(input_label_file+' '+" Region "+r_label+" "+type_label)
        #sys.exit(0)
        #continue
        region_labels.append(r_label)
        region_type_label.append(type_label)
        # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
        #coords = np.zeros((len(vertices), 2))
        coords=[]
        for i, vertex in enumerate(vertices):
            #coords[i][0] = vertex.attributes['X'].value
            #coords[i][1] = vertex.attributes['Y'].value
            x = int(float(vertex.attributes['X'].value)/divisor)
            y = int(float(vertex.attributes['Y'].value)/divisor)
            coords.append((x,y))
            print(i,x,y)
        coords.append(coords[0])    
        #if x_cor==1:
        #	finalcoords=coords	
        #else:
        #	finalcoords= np.concatenate((finalcoords, coords), axis=0)
        #regions.append(coords)
        p1=Polygon(coords)
        p1 = p1.buffer(0)
        #print(p1)
        #print(p1.is_empty)
        #print("poly",p1.is_valid)
        if not p1.is_empty:
            regions.append(p1)
    
    tf_writer = tf.python_io.TFRecordWriter(os.path.join(tf_output, samp + '.tfrecords'))
    #file_patches = os.listdir(os.path.join(patch_dir,samp+'_patches'))
    #for i in file_patches:
    OSobj = openslide.OpenSlide(svs_file)
    poly_included = []
    for i in range(0,len(patch_start_x_list),1):
        x1 = int(patch_start_x_list[i]*OSobj.level_downsamples[patch_level])
        x2 = int(patch_stop_x_list[i]*OSobj.level_downsamples[patch_level])
        y1 = int(patch_start_y_list[i]*OSobj.level_downsamples[patch_level])
        y2 = int(patch_stop_y_list[i]*OSobj.level_downsamples[patch_level])
        img = OSobj.read_region((x1,y1), patch_level, (patch_size, patch_size))
        print(x1,y1,patch_level,patch_size)
        '''Change to grey scale'''
        grey_img = img.convert('L')
        '''Convert the image into numpy array'''
        np_grey = np.array(grey_img)
        patch_mean=round(np.mean(np_grey),2)
        patch_std=round(np.std(np_grey),2)
        
        image_format="png"
        height = patch_size
        width = patch_size
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='PNG')
        size_bytes = imgByteArr.tell()
        #if patch_mean<threshold_mean and patch_std>threshold_std and size_bytes>patch_byte_cutoff:
        image_name=samp+"_x_"+str(x1)+"_"+str(x2)+"_y_"+str(y1)+"_"+str(y2)+'_'+str(patch_mean)+'_'+str(patch_std)+'_'+str(size_bytes)+".png"
        #img.save(image_name, format='PNG')
        #if  size_bytes>0:    
        
        
        x1=int((patch_start_x_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        x2=int((patch_stop_x_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        y1=int((patch_start_y_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        y2=int((patch_stop_y_list[i]*OSobj.level_downsamples[patch_level])/toplevel[2])
        poly=Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
        
        
        res = [ poly.intersects(i) for i in regions ] 
        if True in res:
            poly_included.append(poly)
            imgByteArr = imgByteArr.getvalue()
            record = image_to_tfexample_chek2(imgByteArr, image_format, int(height), int(width), image_name, mut_type)
            tf_writer.write(record.SerializeToString())
       
    tf_writer.close()
    
    patch_sub_size_x=toplevel[0]
    patch_sub_size_y=toplevel[1]
    img_patch = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    
    np_img = np.array(img_patch)
    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img_patch)

    for j in range(0, len(poly_included)):
        patch1 = PolygonPatch(poly_included[j], facecolor=[0, 0, 0], edgecolor="green", alpha=0.3, zorder=2)
        ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir , samp + "_mask_patchoverlay_final.png"), pad_inches='tight')
    

def main():
    abspath = os.path.abspath(__file__)
    words = abspath.split("/")
    '''reading the config filename'''
    parser = argument_parse()
    arg = parser.parse_args()
    '''printing the config param'''
    print("Entered INPUT Filename " + arg.input_file)
    print("Entered Output Patch Directory " + arg.patch_dir)
    print("Entered Output TF Directory " + arg.tf_output)
    print("Entered Patch size " + arg.patch_size)
    print("Entered Level " + arg.level)
    print("Entered background Threshold pixel cutoff " + arg.threshold)
    print("Entered background Threshold pixel cutoff area percent " + arg.threshold_area_percent)
    print("Entered background Threshold mean cutoff " + arg.threshold_mean)
    print("Entered background Threshold std cutoff " + arg.threshold_std)
    print("Entered patch_byte_cutoff " + arg.patch_byte_cutoff)  
    print("Entered RGB2labthreshold Threshold std cutoff " + arg.rgb2lab_thresh)
    print("Entered mut type "+ arg.mut_type)
    print("Entered xml ann file "+ arg.ann)
    patch_sub_size = int(arg.patch_size)
    rgb2lab_thresh = arg.rgb2lab_thresh
    patch_dir = arg.patch_dir
    tf_output = arg.tf_output
    xml_ann = arg.ann
    # patch_level=2
    patch_level = int(arg.level)
    patch_size = int(arg.patch_size)
    threshold = float(arg.threshold)
    mut_type = int(arg.mut_type)
    threshold_area_percent = float(arg.threshold_area_percent)
    svs_file = arg.input_file
    threshold_mean = float(arg.threshold_mean)
    threshold_std = float(arg.threshold_std)
    patch_byte_cutoff = float(arg.patch_byte_cutoff)
    
    '''Reading TCGA file'''
    samp = os.path.basename(svs_file)
    #samp = samp.replace('.tiff', '')
    #samp = samp.replace('.svs', '')
    #samp = samp.replace('.isyntax', '')
    #samp = samp.replace('_BIG', '')

    '''creating binary mask to inspect areas with tissue and performance of threshold''' 
    list_binary_img,toplevel=create_binary_mask_new(rgb2lab_thresh,svs_file,patch_dir,samp)
    '''extracting patch coordinates for requested level based on threshold''' 
    patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list =calc_patches_cord(list_binary_img,patch_level,svs_file,patch_dir,samp,patch_size,threshold_area_percent,toplevel)
    '''creating summary image of toplevel with over lay of selected patches'''
    create_summary_img(patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list,samp,patch_dir,toplevel,patch_level,svs_file)
    '''extracting patches and creating tfrecords & passing xml annotations'''
    create_tfrecord(patch_start_x_list,patch_stop_x_list,patch_start_y_list,patch_stop_y_list,samp,patch_dir,patch_level,svs_file,toplevel,tf_output,patch_size,mut_type,threshold_mean,threshold_std,patch_byte_cutoff,xml_ann)



if __name__ == "__main__":
    main()
