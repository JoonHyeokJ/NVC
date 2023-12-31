#### download link of Retail-71: https://drive.google.com/file/d/1ySCLGlJ9KEo2dOTIpFs_kpFfI_pf1E8v/view?usp=sharing
### RS: Rule-based Synthesis applied on source domain to generate intermediate-domain datasets. Note that target-domain dataset is NOT augmented by RS.
#   |_ 0 means O: original source-domain dataset (i.e., no RS is applied)
#   |_ 1 means E: intermediate-domain dataset generated by RS with easy difficulty
#   |_ 2 means M: intermediate-domain dataset generated by RS with medium difficulty
#   |_ 3 means MP: intermediate-domain dataset generated by RS with medium difficulty and zero padding
#   |_ 4 means H: intermediate-domain dataset generated by RS with hard difficulty
#   |_ The multi-digit indicates the multiple intermediate datasets corresponding to the digits are mixed. Some examples are as follows.
#       |_ 013 means the mixture of O, E, and MP.
#       |_ 02 means the mixture of O and M 


### vit_tiny
# retail71(no RS) + SDAT (trade-off coefficient is 0, i.e, --triplet_coef 0.0), pretrained model: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/0 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_tiny_patch16_224  --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/sdat_rs0 ;
# retail71(RS) + SDAT (trade-off coefficient is 0, i.e, --triplet_coef 0.0), pretrained model: https://drive.google.com/file/d/1D1bz03KrKNPdQqdRzkR0AK_gTRA-Vl9j/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/02 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_tiny_patch16_224  --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/sdat_rs02 ;
# retail71(no RS) + SDAT+NVC (Ours), pretrained model: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/0 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_tiny_patch16_224  --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/nvc_rs0 ;
# retail71(RS) + SDAT+NVC (Ours), pretrained model: https://drive.google.com/file/d/14SwCWr0cQ_RCH9W0qmCrfDrszB4n68zC/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/02 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_tiny_patch16_224  --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/nvc_rs02 ;

### vit_small
# retail71(no RS) + SDAT (trade-off coefficient is 0, i.e, --triplet_coef 0.0), pretrained model: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/0 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_small_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/sdat_rs0 ;
# retail71(RS) + SDAT (trade-off coefficient is 0, i.e, --triplet_coef 0.0), pretrained model: https://drive.google.com/file/d/13LO1Tjm9_XWFXSb7axaq_A5iWSDaIfuI/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/013 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_small_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/sdat_rs013 ;
# retail71(no RS) + SDAT+NVC (Ours), pretrained model: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/0 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_small_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/nvc_rs0 ;
# retail71(RS) + SDAT+NVC (Ours), pretrained model: https://drive.google.com/file/d/1xpuzzrJnardRJVG42iPQiGQJoNtFw1T0/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/013 -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185 --per-class-eval --no-pool --lr 0.015 --seed 0 -a vit_small_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/nvc_rs013 ;

