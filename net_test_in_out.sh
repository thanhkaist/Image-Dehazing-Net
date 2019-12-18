# remove all eval data
echo "============= Remove all previous data ===================="
rm -rf val/
echo $?

echo "============ Indoor Dataset ====================="
python test.py Normal --pretrained_model resultIn6/Net1/model/model_best.pt --dataset "Indoor" --resblock 6 --gpu 1

echo "============= Outdoor Dataset ===================="
python test.py Normal --pretrained_model resultIn6/Net1/model/model_best.pt --dataset "Indoor" --resblock 6 --gpu 1

