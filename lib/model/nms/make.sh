#!/usr/bin/env bash
#export CUDA_PATH=/usr/local/cuda/
#export CXXFLAGS="-std=c++11"
#export CFLAGS="-std=c99"
#
#export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
#export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
