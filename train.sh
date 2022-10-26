### viton_resize dataset
python run.py --data_root ../datasets/viton_resize  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage ILM --gpu_id 0 --num_iters 800000
### MPV dataset  (use data_mpv)
python run.py --data_root ../datasets/MPV --data_mpv  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage MWM --load_iter 2000 --gpu_id 0 --num_iters 800000