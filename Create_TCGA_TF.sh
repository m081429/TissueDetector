#! /bin/bash
#$ -q 1-day
#$ -l h_vmem=30G
#$ -M prodduturi.naresh@mayo.edu
#$ -t 1-572:1
# #$ -t 1-1:1
#$ -m abe
#$ -V
#$ -cwd
# #$ -pe threaded 4
#$ -j y
#$ -o /research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/BRAF_TIFF/LOG

#set -x
dir=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/BRAF_TIFF
cd $dir
FINAL_COMBINED_FILE=$dir/Final_input.txt
temp_dir=$dir/LOG
PATCH_DIR=$dir/Final_Patch
TF_DIR=$dir/Final_TF
Patch_size=224
source /research/bsi/tools/biotools/tensorflow/1.12.0/PKG_PROFILE
source /research/bsi/tools/biotools/openslide/3.4.1/PKG_PROFILE
source /research/bsi/tools/biotools/tensorflow/1.12.0/miniconda/bin/activate tf-gpu-cuda8
#python k.py
#python confirm_tf_record1.py > k.txt 2>&1
#python confirm_tf_record1.py > k.txt 2>&1
#python get_num_levels_resol.py
#exit
#for ((SGE_TASK_ID=1;SGE_TASK_ID<=572;SGE_TASK_ID=SGE_TASK_ID+50));
#do
	#SGE_TASK_ID=235
	samp=`head -$SGE_TASK_ID  $FINAL_COMBINED_FILE|tail -1`
	nu=`echo $samp|grep "tiff"|wc -l|cut -f1 -d ' '`
	if [ "$nu" -gt "0" ];
	then
		python3 Create_ImagePatches_level2.py -i $samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l 2 -t 240 -a 0.5 -m 220 -d 5
		exit
	else
		#python3 Create_ImagePatches_level2.py -i $samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l 2 -t 242 -a 0.4 -m 239 -d 5
		k=1
	fi
	
#done

#python confirm_tf_record.py
#python confirm_tf_record.py > confirm_tf_record.txt

#deactivate
#conda deactivate
