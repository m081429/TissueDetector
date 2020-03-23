export LD_LIBRARY_PATH=/research/bsi/tools/biotools/philips-pathology-sdk/Python-3.6.3/lib:/research/bsi/tools/biotools/philips-pathology-sdk/usr/lib64
export PATH=/research/bsi/tools/biotools/philips-pathology-sdk/Python-3.6.3/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/scripts/philips_sdk/philips-pathologysdk-1.2.1
input="/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/data/CHEK2/CHEK2"
output="/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Naresh/data/CHEK2/CHEK2_tiff"
for i in `ls $input/*.isyntax`
do
	fn=`basename $i|sed -e 's/.isyntax/.tiff/'`
	echo $i $fn
	python isyntax_to_tiff.py $i 0 0 0
	mv $fn $output
done
