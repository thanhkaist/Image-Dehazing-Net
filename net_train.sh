#python main.py Normal --saveDir ./resultIn1 --load Net1 --trainset Indoor --testset Indoor --gpu 0 --epochs 4000 --period 30 --lamda 0.5 --alpha 0.5 #--finetuning
python main.py Normal --saveDir ./resultOut1 --load Net1 --trainset Outdoor --testset Outdoor --gpu 0 --epochs 4000 --period 30 --lamda 0.5 --alpha 0.5 #--finetuning
