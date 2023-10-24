corruption=("background" "cutout" "density" "density_inc" "distortion" \
            "distortion_rbf" "distortion_rbf_inv" "gaussian" "impulse" \
            "original" "rotation" "scale" "shear" "uniform" "upsampling"          
             "ufsampling")

for model in   pointnet_cls
do
    for c in ${corruption[*]};
    do 
        for ((s=1;s<=5;s++))
        do
            for dataset in  ModelNet40
            do   
                python attack_evaluation.py --target_model $model --dataset $dataset --corruption $c --severity $s --attack_dir attack --output_dir model_attacked   &   
                
            done
        done
    done
done