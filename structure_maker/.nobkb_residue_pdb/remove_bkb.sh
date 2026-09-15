for filename in $(ls *.pdb);
do
	python nobackbone.py $filename
done

