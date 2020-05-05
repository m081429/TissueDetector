import openslide
import os
import sys
lst = os.listdir("./inputfiles")
for i in lst:
	file='./inputfiles/'+i
	try:
		OSobj = openslide.OpenSlide(file)
		print(file,OSobj.level_count,OSobj.level_dimensions[0])
	except:
		print(file,"NA","NA")
	#sys.exit(0)
		
