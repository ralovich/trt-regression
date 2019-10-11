
export PATH=$PATH:~/git/tensorflow_r1.15/bazel-bin/tensorflow/tools/benchmark/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/extras/CUPTI/lib64/

# requires the following TF bazel target to be built and in PATH: bazel build tensorflow/tools/benchmark:benchmark_model
benchmark_model --graph=model_100.pb --show_flops --input_layer=input_1_100_100_100_1 --input_layer_type=float --input_layer_shape=1,100,100,100,1 --output_layer=inception_v3_3d_1/output/sub_1
