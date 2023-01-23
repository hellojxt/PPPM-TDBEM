rm -r /home/jxt/PPPM-TDBEM/demo/$1/output/img
/home/jxt/PPPM-TDBEM/build/experiments/demo /home/jxt/PPPM-TDBEM/demo/$1/scene.cfg
python audio_convert.py /home/jxt/PPPM-TDBEM/demo/$1/output/