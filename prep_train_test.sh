indir=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/TCGA_MAYO/tfrecord_brca2
outdir=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/TCGA_MAYO/FINAL_TF/BRCA2
file=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/TCGA_MAYO/confirm_tf_record.brca2.final.xls
IFS=$'\n'
for i in `cat $file`
do
	tfr=`echo $i|cut -f1|sed -e 's/.tfrecords//g'`
	label=`echo $i|cut -f2`
	num=`echo $i|cut -f3`
	cate=`echo $i|cut -f4`
	mv $indir/$tfr.tfrecords $outdir/$tfr.$label.$cate.tfrecords
	#exit
done