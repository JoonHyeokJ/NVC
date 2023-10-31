#### Office31

# SDAT
# pretrained models
# |_ office31_a2d_sdat.pth: https://drive.google.com/file/d/1vEk_HKxDCaV4W4klaLzNBY8XHcKTP--A/view?usp=drive_link
# |_ office31_a2w_sdat.pth: https://drive.google.com/file/d/1Us__O0qS0e2T6OMbrRfG79tDbCENrin6/view?usp=drive_link
# |_ office31_d2a_sdat.pth: https://drive.google.com/file/d/1QZa-OFztaadCOPcUV9U8MCwfOfX3d683/view?usp=drive_link
# |_ office31_d2w_sdat.pth: https://drive.google.com/file/d/15f82tjgfvPY-D_yc4DzrND2V2XgioUKj/view?usp=drive_link
# |_ office31_w2a_sdat.pth: https://drive.google.com/file/d/1O00X21Mo1UfpfuwczGfl11hKQcumXVKC/view?usp=drive_link
# |_ office31_w2d_sdat.pth: https://drive.google.com/file/d/1nPiyody49o3vedSXlG1h4Oixk5NtkJBz/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s A -t D --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/A2D/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s A -t W --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/A2W/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s D -t A --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/D2A/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s D -t W --epochs 60 -i 9  -p 4  -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/D2W/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s W -t A --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/W2A/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s W -t D --epochs 60 -i 9  -p 4  -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/W2D/sdat ;

# SDAT+NVC
# pretrained models
# |_ office31_a2d_nvc.pth: https://drive.google.com/file/d/1QMdOVy6YC7Dm9Jma6olzaMM5YgSW2em8/view?usp=drive_link
# |_ office31_a2w_nvc.pth: https://drive.google.com/file/d/1lLgLHLAQa1U2Y7_A9eEIkJJsOOETUafB/view?usp=drive_link
# |_ office31_d2a_nvc.pth: https://drive.google.com/file/d/14ma_UyMgSXg9s7-oadkiT8r0A4o4NNtO/view?usp=drive_link
# |_ office31_d2w_nvc.pth: https://drive.google.com/file/d/1YUpKPo1KJRQHXnPGOQQLTe8jAPl4xt-Z/view?usp=drive_link
# |_ office31_w2a_nvc.pth: https://drive.google.com/file/d/1AuJR7IbR6NKE_IDWgcZuUSdFs24U4lO1/view?usp=drive_link
# |_ office31_w2d_nvc.pth: https://drive.google.com/file/d/1-LSe95l9rSQd1ONe_pIO3deH1snMczoj/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s A -t D --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/A2D/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s A -t W --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/A2W/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s D -t A --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/D2A/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s D -t W --epochs 60 -i 9  -p 4  -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/D2W/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s W -t A --epochs 60 -i 30 -p 15 -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/W2A/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-31 -d Office31 -s W -t D --epochs 60 -i 9  -p 4  -b 96 --no-pool --lr 0.007 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.9 --triplet_temp 0.7     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/W2D/nvc ;