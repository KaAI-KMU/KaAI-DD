CUDA_VISIBLE_DEVICES=0 python model_in,out.py \
               --root_path ~/Driver-Intention-Prediction \
               --video_path_inside  /home/jihwan/Driver-Intention-Prediction/datasets/annotation/face_camera \
               --video_path_outside  /home/jihwan/Driver-Intention-Prediction/datasets/annotation/road_camera \
               --annotation_path /home/jihwan/Driver-Intention-Prediction/datasets/annotation \
			   --result_path_inside inresults \
			   --result_path_outside outresults \
			   --dataset_inside Brain4cars_Inside \
			   --dataset_outside Brain4cars_Outside \
			   --n_classes 400 \
			   --n_finetune_classes 5 \
			   --pretrain_path_inside /home/jihwan/Driver-Intention-Prediction/inpt/resnet-50-kinetics.pth \
			   --pretrain_path_outside /home/jihwan/Driver-Intention-Prediction/outpt/convlstm.pth \
			   --ft_begin_index 4 \
			   --model resnet \
			   --model_depth 50 \
			   --resnet_shortcut B \
			   --batch_size 8 \
			   --n_threads 4 \
			   --checkpoint 5  \
			   --n_epochs 40 \
			   --resnext_cardinality 32 \
			   --begin_epoch 1 \
			   --sample_duration 16 \
			   --end_second 5 \
			   --train_crop 'driver focus'\
			   --n_scales_inside 3 \
			   --n_scales_outside 1 \
			   --norm_value_inside 1 \
			   --norm_value_outside 255 \
			   --learning_rate 0.1 \
			   --n_fold 0 \
			   --sample_duration_inside 16 \
			   --sample_duration_outside 5 \

