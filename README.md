TensorRT 6.0.1 preformance regression against TensorFlow with 3D convolutions and pooling
=========================================================================================

This repository is meant to contain a dataset and scripts to reproduce
the performance regression we have observed across TF and TF-TRT. It
appears that running a network that processes 3D convolutions and
pooling layers is executed faster through TF than through TF-TRT. We
have observed this behavious on two GPUs: GTX 1080 8GB, and V100 16GB.

TensorFlow needs to built against TensorRT 6 and needs to contain at least
[4297539768bfb6d45d3248fc6471e84e260efc6c](https://github.com/tensorflow/tensorflow/commit/4297539768bfb6d45d3248fc6471e84e260efc6c).