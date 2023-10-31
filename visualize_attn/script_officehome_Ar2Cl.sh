## download link of officehome_ar2cl_sdat.pth: https://drive.google.com/file/d/1AB-DqoWRZiTOrm087Y7x_loU0Mbcvof1/view?usp=sharing
CUDA_VISIBLE_DEVICES=0 python attention_visualization_for_a_dataset.py --root /your/path/to/Office-Home --dataset_name OfficeHome -t Cl -a vit_base_patch16_224 --no-pool --ckpt_path /your/path/to/officehome_ar2cl_sdat.pth --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/sdat --rollout_head_fusion max ;

## download link of officehome_ar2cl_nvc.pth: https://drive.google.com/file/d/13DuKIeVKxUwyKI-Xo1JoOnH7LRVzdWfe/view?usp=sharing
CUDA_VISIBLE_DEVICES=2 python attention_visualization_for_a_dataset.py --root /your/path/to/Office-Home --dataset_name OfficeHome -t Cl -a vit_base_patch16_224 --no-pool --ckpt_path /your/path/to/officehome_ar2cl_nvc.pth  --clarify_pred --result_path /the/target_directory/where/you/want/to/save/result/images/nvc  --rollout_head_fusion max ;
