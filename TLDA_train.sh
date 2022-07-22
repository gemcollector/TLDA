CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
	--algorithm tlda \
	--seed 55 \
	--num_shared_layers 4 \
	--projection_dim  50 \
	--domain_name  ball_in_cup \
	--task_name catch \
	--action_repeat 4
