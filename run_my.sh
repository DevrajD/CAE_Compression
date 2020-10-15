python keras_2_tf.py --keras_hdf5 ./mymethods/encoder_weights.h5 \
                     --tf_ckpt=./mymethods/tf_chkpt.ckpt

freeze_graph \
    --input_meta_graph  ./mymethods/tf_chkpt.ckpt.meta \
    --input_checkpoint  ./mymethods/tf_chkpt.ckpt \
    --output_graph      ./mymethods/frozen_graph.pb \
    --output_node_names conv2d_5/Relu \ #activation_3/Softmax \
    --input_binary      true

vai_q_tensorflow quantize \
        --input_frozen_graph=./mymethods/frozen_graph.pb \
        --input_nodes=input_1 \
        --input_shapes=?,512,768,3 \
        --output_nodes=conv2d_5/Relu \
        --input_fn=image_input_fnmy.calib_input1 \
        --output_dir=mymethods \
        --calib_iter=3

vai_c_tensorflow \
       --frozen_pb=./mymethods/deploy_model.pb \
       --arch=./custom.json \
       --output_dir=mymethods \
       --net_name=encoder_net \
       --options "{'mode':'debug'}"
