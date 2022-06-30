#!/usr/bin/env sh

#/home/tj/ydj_caffe/build/tools/caffe train -gpu=0 --solver=/home/tj/tpj/mltask/cal256_n60/cal256_solver.prototxt --weights=/home/tj/tpj/mltask/models-1.0.2/CNN_S/model

/home/tj/caffe2/build/tools/caffe train -gpu=0 --solver=/home/tj/tpj/cifar10/cifar10_double_alex_nopre6_80/quick_solver.prototxt 
###--weights=/media/tmp/cifar10/cifar10_double_alex_nopre6_64/cifar10_model_double_alex_nopre6_64/cifar10_fine_tuning1_iter_10000.caffemodel

#/home/tj/ydj_caffe/build/tools/caffe train -gpu=1 --solver=/home/tj/tpj/mltask/cal256_n60/cal256_solver.prototxt --weights=/home/tj/tpj/mltask/cal101_n30/cal101_model_n30/cal101_iter_45000.caffemodel

###/home/tj/caffe2/build/tools/caffe test -gpu=0 --weights=/media/tmp/cifar10/cifar10_double_alex_nopre6_80/cifar10_model_double_alex_nopre6_80/cifar10_iter_180000.caffemodel --model=cifar10_train_double_alex_nopre6_80.prototxt --class_num=10 --iterations=500 --outfile_name=_iter_200000

#/home/tj/ydj_caffe/build/tools/caffe test -gpu=0 --weights=/home/tj/tpj/mltask/cal256_n60/cal256_model_n60/cal256_iter_21000.caffemodel --model=cal256_train.prototxt --class_num=256 --iterations=300 --outfile_name=_iter_21000

#!/usr/bin/env sh

#TOOLS=../../build/tools

#GLOG_logtostderr=1 $TOOLS/train_net.bin lenet_solver.prototxt
