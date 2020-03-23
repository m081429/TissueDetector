#samp=/data/Naresh/PhilipsSDK/CodeSample_N/isyntax/00264c6d5a6540739b54e891a1d744a3.isyntax

set -x
dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/data/CHEK2/CHEK2_tiff/
PATCH_DIR=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/data/CHEK2/CHEK2_patch
TF_DIR=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/data/CHEK2/CHEK2_tfrecord
Patch_size=256
IFS=$'\n'
for i in `cat sampler_isyntax.txt|grep "0002f30a017048e38fdf6eef9cd79988"`
do
samp=`echo $i|cut -f1`
mut=`echo $i|cut -f2`
samp=$dir"$samp"
samp=`echo $samp|sed -e 's/.tiff$/_BIG.tiff/g'`
mut=`echo $mut|sed -e 's/CHEK2/1/'|sed -e 's/WT/0/'`
python sampler_openslide.py -i $samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l 2 -t 240 -a 0.4 -m 220 -d 5 -x 0.17 -c $mut
#samp1=`echo $i|cut -f1|sed -e 's/.isyntax//g'`
#echo $samp1
#cp $PATCH_DIR/$samp1*/*.png $PATCH_DIR/$samp1.ori_img.png
done
