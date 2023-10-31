#### download link of Retail-71: https://drive.google.com/file/d/1ySCLGlJ9KEo2dOTIpFs_kpFfI_pf1E8v/view?usp=sharing
# Caution: If your testee model is stored at /your/log/dir/checkpoints/best.pth, you must specify the flag '--log' as /your/log/dir
#   |_ That is, the flag '--log' has to be speficied with the path of grandparent directory of your model(.pth).

### vit_tiny, easy test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1D1bz03KrKNPdQqdRzkR0AK_gTRA-Vl9j/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/14SwCWr0cQ_RCH9W0qmCrfDrszB4n68zC/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;

### vit_tiny, medium test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1D1bz03KrKNPdQqdRzkR0AK_gTRA-Vl9j/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/14SwCWr0cQ_RCH9W0qmCrfDrszB4n68zC/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;

### vit_tiny, hard test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/16sFyR5hR8DIZsBO-ioZecYIE9YLLH_eV/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1D1bz03KrKNPdQqdRzkR0AK_gTRA-Vl9j/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1HK84R7AO3a4-uULZRBY6tEXocU1SRIJJ/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/14SwCWr0cQ_RCH9W0qmCrfDrszB4n68zC/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;



### vit_small, easy test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/13LO1Tjm9_XWFXSb7axaq_A5iWSDaIfuI/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1xpuzzrJnardRJVG42iPQiGQJoNtFw1T0/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;

### vit_small, medium test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/13LO1Tjm9_XWFXSb7axaq_A5iWSDaIfuI/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1xpuzzrJnardRJVG42iPQiGQJoNtFw1T0/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;

### vit_small, hard test
# retail71(no RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1NxtQXlZ1rKkaQUcNZjhKDjaVhJGqUL7V/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
# retail71(RS) + SDAT(baseline), pretrained model: https://drive.google.com/file/d/13LO1Tjm9_XWFXSb7axaq_A5iWSDaIfuI/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
# retail71(no RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/12f8WdYnlDzzb8HUNwMwF9ufF6EhWxdhG/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
# retail71(RS) + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/1xpuzzrJnardRJVG42iPQiGQJoNtFw1T0/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;
