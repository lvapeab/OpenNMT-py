

> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/example/data -src_vocab_size 700 -tgt_vocab_size 700

> python train.py -data data/example/data -save_model data/example/model/tmp_ -world_size 1 -gpu_ranks 0 -rnn_size 64 -word_vec_size 64 -layers 1 -train_steps 5000 -optim adam -learning_rate 0.001 -save_checkpoint_steps 1000 -valid_steps 1000 -dropout 0 -encoder_type brnn -report_every 100

> python trainslate.py -train_from data/example/model/tmp__step_3000.pt -data data/example/data -gpu_ranks 1 -output out_file -verbose