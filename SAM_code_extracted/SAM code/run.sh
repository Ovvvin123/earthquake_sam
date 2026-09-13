
(All) (base) jyr@jyr-4090:/NAS/jyr/mount/bak/data1_Jiang/MF$ 

/NAS/jyr/mount/bak/bakdata/data/all_eqA_296_predict_ds.h5


python trainV8.py --job extract_value --value feature --valueFile /H5/all_eqA_296_predict_feature.h5 --faultFile data/extention_shapefiles/tectonics_Extention_0.95_faults.shp


python trainV8.py --job train --dropout 0  --dw 0.1 --d0 0.02 --alpha 0.2 --mn 256 --resDir train_sam_v12_0/ --lr0 0.001 --dataset key.lst --gpuIndex 0 --batchsize 8 --stepPerMiniBatch 8 --modelType mars_sam

python trainV10.py  --dropout 0.  --dw 0.08 --d0 0.02 --alpha 0.05 --mn 256 --resDir train_sam_v5000_0/ --model mars_sam_vit --lr0 0.001 --dataset key.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train 


python trainV10.py  --dropout 0.  --dw 0.08 --d0 0.02 --alpha 0.05 --mn 256 --resDir train_sam_otherV8/ --model mars_sam_vit --lr0 0.001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther


python trainV10.py  --dropout 0.  --dw 0.08 --d0 0.02 --alpha 0.025 --mn 256 --resDir train_sam_otherV10/ --model mars_sam_vit --lr0 0.001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther


python trainV10.py  --dropout 0.  --dw 0.08 --d0 0.02 --alpha 0.025 --mn 256 --resDir train_sam_otherV10_10otherV2-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther --isalign  



python trainV10.py  --dropout 0.  --dw 0.15 --d0 0.015 --alpha 0.02 --mn 256 --resDir train_sam_otherV10_10otherV2-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther --isalign  

python trainV10.py  --dropout 0.  --dw 0.15 --d0 0.025 --alpha 0.02 --mn 256 --resDir train_sam_otherV10_10otherV3-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther --isalign  



python trainV10.py  --dropout 0.  --dw 0.08 --d0 0.02 --alpha 0.02 --mn 256 --resDir train_sam_otherV10_10otherV5-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1 --job train --isOther --isalign  


python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.02 --mn 256 --resDir train_sam_otherV10_10otherV6-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 

python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.05 --mn 256 --resDir train_sam_otherV10_10otherV8-align/ --model mars_sam_vit --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 

python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.05 --mn 256 --resDir train_sam_otherV10_10otherV8-align_only/ --model mars_sam --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 1 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 


python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.025 --mn 256 --resDir train_sam_otherV10_10otherV10-align_only/ --model mars_sam --lr0 0.0001 --dataset key_all_new.lst --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 


python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.025 --mn 256 --resDir train_sam_otherV10_10otherV10-align_only_add/ --model mars_sam --lr0 0.0001 --dataset key_all.lst --datasetAdd key_all_new.lst  --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 


python trainV10.py  --dropout 0.  --dw 0.12 --d0 0.02 --alpha 0.1 --mn 256 --resDir train_sam_otherV10_12otherV10-align_only_add/ --model mars_sam --lr0 0.0001 --dataset key_all.lst --datasetAdd key_all_new.lst  --gpuIndex 0 --batchsize 24 --stepPerMiniBatch 1  --isOther --isalign  --job train 