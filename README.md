# CudaDigitalAnnealing

Used OS: Ubuntu 20.04.3 LTS

kernel: Linux 5.13.0-51-generic

Image version: nvidia/cuda:10.0-devel-ubuntu18.04

command to create container: docker run -it --gpus all nvidia/cuda:10.0-devel-ubuntu18.04 bash

[Code Source](https://github.com/Shutoparu/CudaDigitalAnnealing.git)

To compile and run the code, run in terminal:

> nvcc cudaDigitalAnnealing.cu -o cudaDA
> 
> ./cudaDA