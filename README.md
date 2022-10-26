# GT-MUST
[ACM MM 2022] [GT-MUST] GT-MUST: Gated Try-on by Learning the Mannequin-Specific Transformation

## Running the program
There are several arguments that can be used, which are
```
--data_root +str #where to get the images
--data_mpv #whether to load MPV dataset (employ dataset_mpv.py for DataLoader)
--data_mode +str #determine to get the training data or testing data
--stage +str #determine the traning stage, ILM, MWM, or GTM
--model_save_path +str #where to save the model during training
--result_save_path +str #where to save the inpainting results during testing
--num_iters +int #the max training iterations
--model_path +str #the pretrained generator to use during training/testing
--batch_size +int #the size of mini-batch for training
--n_threads +int
--test #test the model
--gpu_id +int #which gpu to use
--load_iter +int #for loading pretrained modules
```

To fully exploit the performance of the network, we suggest to use the following training procedure, in specific

1. Train the first module ILM,
```
python run.py --data_root ../datasets/viton_resize  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage ILM --gpu_id 0 --num_iters 200000
```
Use the --data_mpv for MPV dataset, i.e.,
```
python run.py --data_root ../datasets/MPV --data_mpv  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage ILM --gpu_id 0 --num_iters 200000
```

2. Train the second module MWM,
```
python run.py --data_root ../datasets/viton_resize  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage MWM --gpu_id 1 --load_iter 200000 --num_iters 200000
```

3. Train the last module GTM,
```
python run.py --data_root ../datasets/viton_resize  --data_mode train  --model_save_path checkpoint  --model_path checkpoint --stage GTM --gpu_id 2 --load_iter 200000 --num_iters 800000
```

4. Test the model
```
python run.py  --test  --data_root ../datasets/viton_resize  --data_mode test  --gpu_id 1 --model_path checkpoint  --result_save_path results --load_iter 200000
```

## How long to train the model for
All the descriptions below are under the assumption that the size of mini-batch is 16,

| Module | ILM    | MWM    | GTM   |
| :----  | :----: | :----: | :----:|
| Iters  |  200K  |  200K  | 400K  |


## Citation
If you find the article or code useful for your project, please refer to
```
@inproceedings{wang2022gt,
  title={GT-MUST: Gated Try-on by Learning the Mannequin-Specific Transformation},
  author={Wang, Ning and Zhang, Jing and Zhang, Lefei and Tao, Dacheng},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2182--2190},
  year={2022},
  doi = {10.1145/3503161.3547775},
  url = {https://doi.org/10.1145/3503161.3547775}
}
```
## Paper
See the [Paper](/Paper) folder
