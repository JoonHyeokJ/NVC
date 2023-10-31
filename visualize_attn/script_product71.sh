#### download link of Retail-71: https://drive.google.com/file/d/1ySCLGlJ9KEo2dOTIpFs_kpFfI_pf1E8v/view?usp=sharing

#### vit_tiny
## download link of retail71_vit_tiny_rs0_sdat.pth: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=sharing
## download link of retail71_vit_tiny_rs0_nvc.pth: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=sharing
# validation set, i.e. the target-domain training set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/retail71/0 --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/retail71/0 --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# easy-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/1     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/1     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# medium-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/2     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/2     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# hard-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/3     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/3     --dataset_name Retail -t validation -a vit_tiny_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_tiny_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;

#### vit_small
## download link of retail71_vit_small_rs0_sdat.pth: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=sharing
## download link of retail71_vit_small_rs0_nvc.pth: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=sharing
# validation set, i.e. the target-domain training set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/retail71/0 --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/retail71/0 --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# easy-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/1     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/1     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# medium-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/2     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/2     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
# hard-difficulty test set
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/3     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Retail-71/test/3     --dataset_name Retail -t validation -a vit_small_patch16_224 --no-pool --ckpt_path /your/path/to/retail71_vit_small_rs0_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;