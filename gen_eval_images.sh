# remove all eval data
echo "============= Remove all previous data ===================="
rm -rf val/
echo $?
# Set 5
echo "============ EDSR ====================="
echo "============= SET 5 ===================="
python test.py EDSR --pretrained_model result1000/EDSR/model/model_best.pt --HR_valDataroot data/benchmark/Set5/HR/ --LR_valDataroot 'data/benchmark/Set5/LR_bicubic/X2' --datasetName "Set5" --gpu 1
# Set 14
echo "============= SET 14 ===================="
python test.py  EDSR --pretrained_model result1000/EDSR/model/model_best.pt --HR_valDataroot data/benchmark/Set14/HR/ --LR_valDataroot 'data/benchmark/Set14/LR_bicubic/X2' --datasetName "Set14" --gpu 1
# Urban100
echo "============= Urban 100  ===================="
python test.py EDSR --pretrained_model result1000/EDSR/model/model_best.pt --HR_valDataroot data/benchmark/Urban100/HR/ --LR_valDataroot 'data/benchmark/Urban100/LR_bicubic/X2' --datasetName "Urban100" --gpu 1
# B100
echo "============= B 100 ===================="
python test.py EDSR --pretrained_model result1000/EDSR/model/model_best.pt --HR_valDataroot data/benchmark/B100/HR/ --LR_valDataroot 'data/benchmark/B100/LR_bicubic/X2' --datasetName "B100" --gpu 1
echo "++++++++++++++RCAN +++++++++++++++++++++"
echo "============= SET 5 ===================="
python test.py RCAN --pretrained_model result1000/RCAN/model/model_best.pt --HR_valDataroot data/benchmark/Set5/HR/ --LR_valDataroot 'data/benchmark/Set5/LR_bicubic/X2' --datasetName "Set5" --gpu 1
# Set 14
echo "============= SET 14 ===================="
python test.py  RCAN --pretrained_model result1000/RCAN/model/model_best.pt --HR_valDataroot data/benchmark/Set14/HR/ --LR_valDataroot 'data/benchmark/Set14/LR_bicubic/X2' --datasetName "Set14" --gpu 1
# Urban100
echo "============= Urban 100  ===================="
python test.py RCAN --pretrained_model result1000/RCAN/model/model_best.pt --HR_valDataroot data/benchmark/Urban100/HR/ --LR_valDataroot 'data/benchmark/Urban100/LR_bicubic/X2' --datasetName "Urban100" --gpu 1
echo "============= B 100 ===================="
python test.py RCAN --pretrained_model result1000/RCAN/model/model_best.pt --HR_valDataroot data/benchmark/B100/HR/ --LR_valDataroot 'data/benchmark/B100/LR_bicubic/X2' --datasetName "B100" --gpu 1
