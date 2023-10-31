########## all of models here is trained with "Office31"
########## tSNE with respect to domain

## [W2A]
# office31 + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1O00X21Mo1UfpfuwczGfl11hKQcumXVKC/view?usp=sharing
CUDA_VISIBLE_DEVICES=1 python tSNE_plot_domain_w_negaug.py --ckpt_path /your/path/to/office31_w2a_sdat.pth --arch vit_base_patch16_224 --dataset_name Office31 -s W -t A -b 30 --root /your/path/to/Office-31 --method SDAT --no-pool       --result_path 'near_model' ;
# office31 + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1AuJR7IbR6NKE_IDWgcZuUSdFs24U4lO1/view?usp=sharing
CUDA_VISIBLE_DEVICES=1 python tSNE_plot_domain_w_negaug.py --ckpt_path /your/path/to/office31_w2a_nvc.pth  --arch vit_base_patch16_224 --dataset_name Office31 -s W -t A -b 30 --root /your/path/to/Office-31 --method negaug --no-pool       --result_path 'near_model' ;
