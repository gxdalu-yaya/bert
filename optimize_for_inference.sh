python -m tensorflow.python.tools.optimize_for_inference --input merge1_graph.pb --output graph_optimized.pb --input_names=input_ids,input_mask,segment_ids --output_names=Softmax
