#!/usr/local/biotools/python/3.4.3/bin/python3
__author__ = "Naresh Prodduturi"
__email__ = "prodduturi.naresh@mayo.edu"
__status__ = "Dev"

import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
import glob	
import openslide
import numpy as np
from PIL import Image, ImageDraw


import math
import tensorflow as tf
import io
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import geo
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch

import tensorflow as tf
import io
from dataset_utils import * 

'''function to check if input files exists and valid''' 	
def input_file_validity(file):
	'''Validates the input files'''
	if os.path.exists(file)==False:
		raise argparse.ArgumentTypeError( '\nERROR:Path:\n'+file+':Does not exist')
	if os.path.isfile(file)==False:
		raise argparse.ArgumentTypeError( '\nERROR:File expected:\n'+file+':is not a file')
	if os.access(file,os.R_OK)==False:
		raise argparse.ArgumentTypeError( '\nERROR:File:\n'+file+':no read access ')
	return file


def argument_parse():
	'''Parses the command line arguments'''
	parser=argparse.ArgumentParser(description='')
	parser.add_argument("-p","--patch_dir",help="Patch dir",required="True")
	parser.add_argument("-i","--input_file",help="input file",required="True")
	parser.add_argument("-o","--tf_output",help="output tf dir",required="True")
	parser.add_argument("-s","--patch_size",help="Patch_size",required="True")
	parser.add_argument("-t","--threshold",help="background Threshold pixel cutoff",required="True")
	parser.add_argument("-a","--threshold_area_percent",help="background Threshold pixel cutoff area percent",required="True")
	parser.add_argument("-m","--threshold_mean",help="background Threshold mean cutoff",required="True")
	parser.add_argument("-d","--threshold_std",help="background Threshold std cutoff",required="True")
	parser.add_argument("-l","--level",help="level",required="True")
	return parser

def create_summary_image(svs,filenames,patch_dir,patch_level):
	
	OSobj = openslide.OpenSlide(svs)
	inlevel=OSobj.level_count-1
	if 'tiff' in svs:
		inlevel = 7
	else:
		inlevel = patch_level+1
	level=inlevel
	input_level=patch_level
	patch_sub_size_x=OSobj.level_dimensions[inlevel][0]
	patch_sub_size_y=OSobj.level_dimensions[inlevel][1]
	img_patch = OSobj.read_region((0,0), level, (patch_sub_size_x, patch_sub_size_y))
	multi_factor=OSobj.level_downsamples
	
	poly_included=[]
	poly_excluded=[]
	name=""
	for i in filenames:
		p = i.split("X_")
		p0 = p[1].split("_")
		p1 = p[1].split("Y_")
		p2 = p1[1].split("_")	
		name=p[0]
		name = name[:-1]
		x1=p0[0]
		x2=p0[1]
		y1=p2[0]
		y2=p2[1]
		# print(p)
		# print(p[0])
		# print(p1)
		# print(p2)
		# print(name+' '+x1+' '+x2+' '+y1+' '+y2)
		# sys.exit(0)
		if x1 == "0":
			x1 = int(x1)
		else:        
			x1=(int(x1)*multi_factor[input_level])/multi_factor[level]
			x1=int(x1)
		if y1 == "0":
			y1 = int(y1)
		else:    
			y1=(int(y1)*multi_factor[input_level])/multi_factor[level]
			y1=int(y1)

		x2=(int(x2)*multi_factor[input_level])/multi_factor[level]
		x2=int(x2)
		y2=(int(y2)*multi_factor[input_level])/multi_factor[level]
		y2=int(y2)
		if 'included' in i:
			poly_included.append(Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]))
		else:
			poly_excluded.append(Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]))
	
	f, ax = plt.subplots(frameon=False)
	f.tight_layout(pad=0, h_pad=0, w_pad=0)
	ax.set_xlim(0, patch_sub_size_x)
	ax.set_ylim(patch_sub_size_y, 0)
	ax.imshow(img_patch)
	for j in range(0,len(poly_excluded)):
		patch1 = PolygonPatch(poly_excluded[j], facecolor=[0,0,0], edgecolor="red", alpha=0.4, zorder=2)
		ax.add_patch(patch1)
	for j in range(0,len(poly_included)):
		patch1 = PolygonPatch(poly_included[j], facecolor=[0,0,0], edgecolor="green", alpha=0.3, zorder=2)
		ax.add_patch(patch1)        
	ax.set_axis_off()
	DPI = f.get_dpi()
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1,wspace=0, hspace=0)
	f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
	f.savefig(patch_dir+"/"+name+".png", pad_inches='tight')
	
	return True
  
def create_patch(svs,patch_sub_size,patch_dir,samp,tf_output,patch_level,threshold,threshold_area_percent, threshold_mean, threshold_std):
	#print(svs+' '+str(patch_sub_size)+' '+patch_dir+' '+samp+' '+str(patch_level)+' '+tf_output)
	tf_writer=tf.python_io.TFRecordWriter(tf_output+'/'+samp+'.tfrecords')
	#threshold=240
	#level=2
	#threshold=242
	level=patch_level
	OSobj = openslide.OpenSlide(svs)
	minx = 0
	miny = 0
	tmp=OSobj.level_dimensions[level]
	maxx = tmp[0]
	maxy = tmp[1]
	#this factor if required to convert level0 start coordinatess to level2 start coordinates (this is required for OSobj.read_region function)
	multi_factor=OSobj.level_downsamples[level]
	#print(svs+' '+str(patch_sub_size)+' '+patch_dir+' '+str(maxx))
	start_x=minx	
	'''creating sub patches'''	
	'''Iterating through x coordinate'''	
	current_x=0
	filenames=[]
	#num=0
	#while start_x < maxx:
	while start_x+patch_sub_size < maxx:
		'''Iterating through y coordinate'''
		current_y=0
		start_y=miny
		#while start_y < maxy:
		while start_y+patch_sub_size < maxy:
			tmp_start_x=int(round(start_x*multi_factor,0))
			tmp_start_y=int(round(start_y*multi_factor,0))
			try:
				img_patch = OSobj.read_region((tmp_start_x,tmp_start_y), level, (patch_sub_size, patch_sub_size))
			except:
				sys.exit(0)
			#img_patch = OSobj.read_region((start_x,start_y), level, (maxx, maxy))
			#num=num+1
			#img_patch.save(patch_dir+'/'+str(num)+'.png', "png")
			#sys.exit(1)
			np_img = np.array(img_patch)
			#max_min_channels = [np.amax(np_img[:,:,0]), np.amax(np_img[:,:,1]), np.amax(np_img[:,:,2]), np.amin(np_img[:,:,0]), np.amin(np_img[:,:,1]), np.amin(np_img[:,:,2])]
			im_sub = Image.fromarray(np_img).convert('RGB')
			np_img = np.array(im_sub)
			patch_mean=np.mean(np_img,axis=(0, 1))
			patch_std=np.std(np_img,axis=(0, 1))
			list_patch_mean = [str(i) for i in patch_mean]
			list_patch_std = [str(i) for i in patch_std]			
			str_patch_mean = "_".join(list_patch_mean)
			str_patch_std = "_".join(list_patch_std)
			#print(str_patch_mean)
			#print(str_patch_std)
			#sys.exit(0)
			#creating IO bytes objects and 
			imgByteArr = io.BytesIO()
			im_sub.save(imgByteArr, format='PNG')
			size_bytes = imgByteArr.tell()
			
			width, height = im_sub.size
			'''Change to grey scale'''
			grey_img = im_sub.convert('L')
			'''Convert the image into numpy array'''
			np_grey = np.array(grey_img)
			patch_mean=round(np.mean(np_grey),2)
			patch_std=round(np.std(np_grey),2)
			#patch_max=round(np.amax(np_grey),2)
			#patch_min=round(np.amin(np_grey),2)
			'''Identify patched where there are tissues'''
			'''tuple where first element is rows, second element is columns'''
			idx = np.where(np_grey < threshold)
			#print(len(idx[0]))
			#print(np_grey.shape[0]*np_grey.shape[1])
			patch_area=len(idx[0])/(np_grey.shape[0]*np_grey.shape[1])

			'''proceed further only if patch has non empty values'''
			#if len(idx[0])>0 and len(idx[1])>0 and width==patch_sub_size and height==patch_sub_size:
			num_patch1=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)+"_mean_"+str(str_patch_mean)+"_patch_std_"+str(str_patch_std)+"_patch_area_"+str(patch_area)+"_size_"+str(size_bytes)
			num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)
			if patch_area> threshold_area_percent and patch_mean<threshold_mean and patch_std>threshold_std and width==patch_sub_size and height==patch_sub_size and size_bytes>19000:
				#if	threshold_area> threshold_area_percent and patch_std>5: 
				#if width==patch_sub_size and height==patch_sub_size:
				#print(num_patch1)
				'''creating patch name'''
				#num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)
				num_patch=num_patch+"_included"
				#filenames.append(num_patch)
				#tmp_png=patch_dir+'/'+num_patch+'.png'
				# tmp_png=patch_dir+'/'+"included_mean_"+str(patch_mean)+"patch_std_"+str(patch_std)+"_"+num_patch+'.png'
				# else:
					# '''creating patch name'''
					# num_patch=samp+"_X_"+str(start_x)+"_"+str(start_x+patch_sub_size)+"_Y_"+str(start_y)+"_"+str(start_y+patch_sub_size)
					# filenames.append(num_patch)
					# tmp_png=patch_dir+'/'+"excluded_mean_"+str(patch_mean)+"patch_std_"+str(patch_std)+"_"+num_patch+'.png'
			
				# '''saving image'''
				#im_sub.save(patch_dir+'/'+num_patch+".png", "png")
				#sys.exit(1)
				image_format="png"    
				height=patch_sub_size
				width=patch_sub_size 
				image_name=num_patch 
	
				sub_type=2
				#if p[1] == "TCGA-LUAD":
					#sub_type=0
				#if	p[1] == "TCGA-LUSC":
					#sub_type=1
				#sub_type=p[1]
				if 'BRAF' in samp:
					sub_type=1
				else:
					sub_type=0
				mut_type=""
				if 'WT' in samp:
					mut_type='WT'
				elif 'BRAF_V600E' in samp:
					mut_type='BRAF_V600E'
				elif 'BRAF_V600K' in samp:
					mut_type='BRAF_V600K'
				elif 'BRAF_V600NENK' in samp:
					mut_type='BRAF_V600NENK'
				elif 'BRAF_V600X' in samp:
					mut_type='BRAF_V600X'
				else:
					mut_type='Normal'	
				#imgByteArr = io.BytesIO()
				#im_sub.save(imgByteArr, format='PNG')
				imgByteArr = imgByteArr.getvalue()
				record=image_to_tfexample_braf(imgByteArr,image_format,int(height),int(width),image_name, sub_type, mut_type)
				tf_writer.write(record.SerializeToString())
			filenames.append(num_patch)
			start_y	= start_y+patch_sub_size
			current_y = current_y+patch_sub_size
		start_x = start_x+patch_sub_size	
		current_x = current_x+patch_sub_size	
	#sys.exit(1)
	tf_writer.close()
	return filenames
	
def main():	
	abspath=os.path.abspath(__file__)
	words = abspath.split("/")
	'''reading the config filename'''
	parser=argument_parse()
	arg=parser.parse_args()
	'''printing the config param'''
	print("Entered INPUT Filename "+arg.input_file)
	print("Entered Output Patch Directory "+arg.patch_dir)
	print("Entered Output TF Directory "+arg.tf_output)
	print("Entered Patch size "+arg.patch_size)
	print("Entered Level "+arg.level)
	print("Entered background Threshold pixel cutoff "+arg.threshold)
	print("Entered background Threshold pixel cutoff area percent "+arg.threshold_area_percent)
	print("Entered background Threshold mean cutoff "+arg.threshold_mean)
	print("Entered background Threshold std cutoff "+arg.threshold_std)	
	patch_sub_size=int(arg.patch_size)
	patch_dir=arg.patch_dir
	tf_output=arg.tf_output
	#patch_level=2
	patch_level=int(arg.level)
	threshold=float(arg.threshold)
	threshold_area_percent=float(arg.threshold_area_percent)
	svs_file=arg.input_file
	threshold_mean=float(arg.threshold_mean)
	threshold_std=float(arg.threshold_std)
	'''Reading TCGA file'''
	samp=os.path.basename(svs_file)
	samp=samp.replace('.tiff','')
	samp=samp.replace('.svs','')
	'''Creating Patches'''
	filenames=create_patch(svs_file,patch_sub_size,patch_dir,samp,tf_output,patch_level,threshold,threshold_area_percent, threshold_mean, threshold_std)
	#sys.exit(0)
	#print("\n".join(filenames))
	'''Creating Masked images'''
	output=create_summary_image(svs_file,filenames,patch_dir,patch_level)
	

	
if __name__ == "__main__":
	main()
