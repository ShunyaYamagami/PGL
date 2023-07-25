if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

gpu_i=$1
exec_num=$2
dset_num=$3

parent="OfficeHome"

# COMMAND="trap 'echo "エラーが発生しました。"; conserve $gpu_i' ERR"
# COMMAND+=" && conda deactivate && conda deactivate"
COMMAND="conda deactivate && conda deactivate"
COMMAND+=" && conda activate pgl"

if [ $parent = 'Office31' ]; then
    dsetlist=("Amazon_Dslr" "Dslr_Webcam" "Webcam_Amazon")
elif [ $parent = 'OfficeHome' ]; then
    dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
elif [ $parent = 'DomainNet' ]; then
    dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
else
    echo "不明なデータセット: $parent" >&2
    return 1
fi


if [ $dset_num -eq -1 ]; then
    for dset in "${dsetlist[@]}"; do
        COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python train.py \
                    --dset $dset \
                    --data_dir /nas/data/syamagami/GDA/data/
                "
    done
else
    dset=${dsetlist[$dset_num]}
    COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python train.py \
                    --dset $dset \
                    --data_dir /nas/data/syamagami/GDA/data/
                "
fi

echo $COMMAND
echo ''

eval $COMMAND
