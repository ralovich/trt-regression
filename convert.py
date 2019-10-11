#!/usr/bin/env python3

input_graph='model_100.pb'
output_graph='model_100-tf-trt.pb'
output_nodes=['inception_v3_3d_1/output/sub_1']

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
with tf.Session() as sess:
    with tf.gfile.GFile(input_graph, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    
    converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=output_nodes) #output nodes
    trt_graph = converter.convert()
    
    for n in trt_graph.node:
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
            with tf.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
                f.write(n.attr["serialized_segment"].s)
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))
    
    trt_graph_def = trt_graph

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(trt_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(trt_graph_def.node))    
