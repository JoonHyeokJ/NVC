# Caution: If your testee model is stored at /your/log/dir/checkpoints/best.pth, you must specify the flag '--log' as /your/log/dir
#   |_ That is, the flag '--log' has to be speficied with the path of grandparent directory of your model(.pth).


# Retail71, vit_tiny: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs0_nvc.pth ;
# Retail71+RS, vit_tiny: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_tiny_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_tiny_rs02_nvc.pth ;

# Retail71, vit_small: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs0_nvc.pth ;
# Retail71+RS, vit_small: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/1 -d Retail -s train -t validation --phase test_negaug --test_difficulty 1 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/2 -d Retail -s train -t validation --phase test_negaug --test_difficulty 2 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Retail-71/test/3 -d Retail -s train -t validation --phase test_negaug --test_difficulty 3 -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_small_patch16_224 --log /your/path/to/grandparent_directory_of_retail71_vit_small_rs013_nvc.pth ;

# VisDA2017, vit_base: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/VisDA2017 -d VisDA2017 -s Synthetic -t Real --phase test_negaug -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_visda2017_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=0 python eval_w_negaug.py /your/path/to/VisDA2017 -d VisDA2017 -s Synthetic -t Real --phase test_negaug -b 120 -p 5 --per-class-eval --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_visda2017_nvc.pth ;

# OfficeHome, vit_base: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2cl_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2pr_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2rw_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2ar_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2pr_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2rw_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2ar_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2cl_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2rw_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2ar_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2cl_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2pr_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2cl_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2pr_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_ar2rw_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2ar_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2pr_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_cl2rw_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2ar_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2cl_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Rw --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_pr2rw_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Ar --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2ar_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Cl --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2cl_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Pr --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_officehome_rw2pr_nvc.pth ;

# Office31, vit_base: SDAT vs ours
# SDAT
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s A -t D --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_a2d_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s A -t W --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_a2w_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s D -t A --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_d2a_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s D -t W --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_d2w_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s W -t A --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_w2a_sdat.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s W -t D --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_w2d_sdat.pth ;
# SDAT+NVC
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s A -t D --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_a2d_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s A -t W --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_a2w_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s D -t A --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_d2a_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s D -t W --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_d2w_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s W -t A --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_w2a_nvc.pth ;
CUDA_VISIBLE_DEVICES=2 python eval_w_negaug.py /your/path/to/Office-31 -d Office31 -s W -t D --phase test_negaug -b 120 -p 5 --no-pool --seed 0 -a vit_base_patch16_224 --log /your/path/to/grandparent_directory_of_office31_w2d_nvc.pth ;
