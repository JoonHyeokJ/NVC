#### VisDA2017

# SDAT, pretrained model: https://drive.google.com/file/d/1vZVfOo0UWl4PcHSZG7evSrE1Qz1roc1P/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/VisDA2017 -d VisDA2017 -s Synthetic -t Real --epochs 20  -b 96 -i 250 -p 50 --per-class-eval --train-resizing cen.crop --no-pool --lr 0.002 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/sdat ;

# SDAT+NVC, pretrained model: https://drive.google.com/file/d/1eE5t5Rwb7oJ-6C48sM_3DqpIQGgoeCNF/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/VisDA2017 -d VisDA2017 -s Synthetic -t Real --epochs 20  -b 96 -i 250 -p 50 --per-class-eval --train-resizing cen.crop --no-pool --lr 0.002 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.1     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/nvc ;

