#!/usr/bin/env sh

#/home/tj/ydj_caffe/build/tools/caffe train -gpu=0 --solver=/home/tj/tpj/mltask/cal256_n60/cal256_solver.prototxt --weights=/home/tj/tpj/mltask/models-1.0.2/CNN_S/model

/home/tj/ydj_caffe/build/tools/caffe train -gpu=1 --solver=/home/tj/tpj/mltask/cal256_n60/cal256_n60_double_alex_nopre6_2/quick_solver.prototxt 
###--weights=/home/tj/tpj/mltask/cal256_n60/cal256_model_n60_double_alex_nopre/cal2562_iter_60000_fine_tuning.caffemodel

#/home/tj/ydj_caffe/build/tools/caffe train -gpu=1 --solver=/home/tj/tpj/mltask/cal256_n60/cal256_solver.prototxt --weights=/home/tj/tpj/mltask/cal101_n30/cal101_model_n30/cal101_iter_45000.caffemodel

###/home/tj/ydj_caffe/build/tools/caffe test -gpu=0 --weights=/home/tj/tpj/mltask/cal256_n60/cal256_n60_double_alex_nopre6/cal256_model_n60_double_alex_nopre6/cal256_iter_200000.caffemodel --model=cal256_train_double_alex_nopre6.prototxt --class_num=256 --iterations=721 --outfile_name=_iter_200000

#/home/tj/ydj_caffe/build/tools/caffe test -gpu=0 --weights=/home/tj/tpj/mltask/cal256_n60/cal256_model_n60/cal256_iter_21000.caffemodel --model=cal256_train.prototxt --class_num=256 --iterations=300 --outfile_name=_iter_21000

#!/usr/bin/env sh

#TOOLS=../../build/tools

#GLOG_logtostderr=1 $TOOLS/train_net.bin lenet_solver.prototxt
