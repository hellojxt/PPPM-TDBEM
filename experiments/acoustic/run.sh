for obj in bowl glasscup tyra
do
    rm -rf ../../dataset/acoustic/$obj/output/*
    rm -rf ./img/$obj/*
    for i in 30 40 50 60 70
    do
        for j in 0 1 2 3 4
        do
            /home/jxt/PPPM-TDBEM/build/experiments/acoustic $obj $i $j
        done
    done
    python test.py $obj
done
