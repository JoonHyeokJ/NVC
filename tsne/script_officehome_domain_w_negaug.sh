########## all of models here is trained with "OfficeHome"
########## tSNE with respect to domain

## [Ar2Cl]
# officehome + SDAT(baseline), pretrained model: https://drive.google.com/file/d/1AB-DqoWRZiTOrm087Y7x_loU0Mbcvof1/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python tSNE_plot_domain_w_negaug.py --ckpt_path /your/path/to/officehome_ar2cl_sdat.pth --arch vit_base_patch16_224 --dataset_name OfficeHome -s Ar -t Cl -b 20 --root /your/path/to/Office-Home --method SDAT --no-pool       --result_path 'near_model' ;
# officehome + SDAT+NVC(ours), pretrained model: https://drive.google.com/file/d/13DuKIeVKxUwyKI-Xo1JoOnH7LRVzdWfe/view?usp=drive_link
CUDA_VISIBLE_DEVICES=2 python tSNE_plot_domain_w_negaug.py --ckpt_path /your/path/to/officehome_ar2cl_nvc.pth  --arch vit_base_patch16_224 --dataset_name OfficeHome -s Ar -t Cl -b 20 --root /your/path/to/Office-Home --method negaug --no-pool       --result_path 'near_model' ;
