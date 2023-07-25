function process_args {
    gpu_i=$1
    exec_num=$2
    dset_num=$3
        shift 3  # 無名引数数

    task=(
        "original"
        # "true_domains"
    )

    parent="home"  # choices([office, home])
    
    COMMAND="conda deactivate && conda deactivate"
    COMMAND+=" && conda activate da"

    if [ $parent = 'office' ]; then
        dsetlist=("Amazon_Dslr" "Dslr_Webcam" "Webcam_Amazon")
    elif [ $parent = 'home' ]; then
        dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
    elif [ $parent = 'DomainNet' ]; then
        dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
    else
        echo "不明なデータセット: $parent" >&2
        return 1
    fi


    for tsk in "${task[@]}"; do
        if [ $dset_num -eq -1 ]; then
            for dset in "${dsetlist[@]}"; do
                COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  python train.py \
                            --dset $dset \
                            --task $tsk \
                            --dataset $parent \
                            --data_dir /nas/data/syamagami/GDA/data/
                        "
            done
        else
            dset=${dsetlist[$dset_num]}
            COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  python train.py \
                            --dset $dset \
                            --task $tsk \
                            --dataset $parent \
                            --data_dir /nas/data/syamagami/GDA/data/
                        "
        fi
    done

    echo $COMMAND
    echo ''

    eval $COMMAND

}

# 最初の3つの引数をチェック
if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"
