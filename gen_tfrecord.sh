for((i=0; i<20; i++))
do
    python gen_tfrecord.py $1 $i &
done
