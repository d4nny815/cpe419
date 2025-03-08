#!/bin/bash
dpcpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend "--cuda-gpu-arch=sm_61" src/main.cpp -o sycl_test