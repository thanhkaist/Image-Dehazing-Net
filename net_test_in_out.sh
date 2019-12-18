#remove all eval data
echo "============= Remove all previous data ===================="
rm -rf val/
echo $?
echo "============ Indoor Dataset ====================="
python test.py Normal --pretrained_model resultOut4/Net1/model/Gen_10000.pt --dataset "Indoor" --resblock 6 --gpu 0
echo "============= Outdoor Dataset ===================="
python test.py Normal --pretrained_model resultOut4/Net1/model/Gen_10000.pt --dataset "Outdoor" --resblock 6 --gpu 0
