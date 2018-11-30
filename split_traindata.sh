shuf $1 > $1.shuf
mv $1.shuf $1
for((i=0; i < 12; i++))
do
    awk 'NR%12==i' i="$i" $1 > $1_${i}
done
