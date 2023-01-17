for i in 30 40 50 60 70
do
    for j in 0 5 10 15 20
    do
        /home/jxt/PPPM-TDBEM/build/experiments/acoustic $1 $i $j
    done
done
python test.py $1