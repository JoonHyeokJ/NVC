## download link of office31_w2a_sdat.pth: https://drive.google.com/file/d/1O00X21Mo1UfpfuwczGfl11hKQcumXVKC/view?usp=sharing
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Office-31 --dataset_name Office31 -t A -a vit_base_patch16_224 --no-pool --ckpt_path /your/path/to/office31_w2a_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;

## download link of office31_w2a_nvc.pth: https://drive.google.com/file/d/1AuJR7IbR6NKE_IDWgcZuUSdFs24U4lO1/view?usp=sharing
CUDA_VISIBLE_DEVICES=2 python attention_visualization_for_a_dataset.py --root /your/path/to/Office-31 --dataset_name Office31 -t A -a vit_base_patch16_224 --no-pool --ckpt_path /your/path/to/office31_w2a_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
