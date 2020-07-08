#! /bin/bash
#$ -q day-rhel7
#$ -l h_vmem=50G
#$ -M prodduturi.naresh@mayo.edu
# #$ -t 1-472:1
#$ -m abe
#$ -V
#$ -cwd
#$ -j y
#$ -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/Quincy_tfrecords/log
set -x
dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/Quincy_tfrecords
cd $dir
lev=0
hed=0.17
PATCH_DIR=$dir/patch_level_quincy
TF_DIR=$dir/tfrecord_level_quincy
mkdir -p $PATCH_DIR $TF_DIR 
Patch_size=512
#SGE_TASK_ID=1
threshold_area_percent_patch=0.5
samp="/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/163370.svs"
mut="1"
/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/conda_env/tf2/bin/python sampler_openslide_quincy.py -i $samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l $lev -a $threshold_area_percent_patch  -x $hed -c $mut
