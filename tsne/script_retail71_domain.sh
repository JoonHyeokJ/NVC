# download link of Retail-71: https://drive.google.com/file/d/1ySCLGlJ9KEo2dOTIpFs_kpFfI_pf1E8v/view?usp=sharing
########## all of models here is trained with "retail71"
########## tSNE with respect to domain, vit-tiny
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_tiny_rs0_sdat.pth   --arch vit_tiny_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method SDAT --no-pool       --result_path 'near_model' ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1D1bz03KrKNPdQqdRzkR0AK_gTRA-Vl9j/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_tiny_rs02_sdat.pth  --arch vit_tiny_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method SDAT --no-pool       --result_path 'near_model' ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_tiny_rs0_nvc.pth    --arch vit_tiny_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method negaug --no-pool       --result_path 'near_model' ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/14SwCWr0cQ_RCH9W0qmCrfDrszB4n68zC/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_tiny_rs02_nvc.pth   --arch vit_tiny_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method negaug --no-pool       --result_path 'near_model' ;

########## tSNE with respect to domain, vit-small
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_small_rs0_sdat.pth   --arch vit_small_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method SDAT --no-pool       --result_path 'near_model' ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/13LO1Tjm9_XWFXSb7axaq_A5iWSDaIfuI/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_small_rs013_sdat.pth --arch vit_small_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method SDAT --no-pool       --result_path 'near_model' ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_small_rs0_nvc.pth    --arch vit_small_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method negaug --no-pool       --result_path 'near_model' ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1xpuzzrJnardRJVG42iPQiGQJoNtFw1T0/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0 python tSNE_plot_domain.py --ckpt_path /your/path/to/retail71_vit_small_rs013_nvc.pth  --arch vit_small_patch16_224 --dataset_name Retail -s train -t validation -b 30 --root /your/path/to/Retail-71/retail71/0 --method negaug --no-pool       --result_path 'near_model' ;
