### for fmnist

nohup python -u FCI_fmnist.py --n_rep 1 > FCI_resnet18_fmnist_OOD0.log &
nohup python -u FCI_fmnist.py > FCI_resnet18_fmnist_OOD0.log &
nohup python -u FCI_fmnist.py --net resnet34 > FCI_resnet34_fmnist_OOD0.log &
nohup python -u FCI_fmnist.py --net vgg16 > FCI_vgg16_fmnist_OOD0.log &

nohup python -u FCI_fmnist.py --miss 5 > FCI_resnet18_fmnist_OOD5.log &
nohup python -u FCI_fmnist.py --net resnet34 --miss 5 > FCI_resnet34_fmnist_OOD5.log &
nohup python -u FCI_fmnist.py --net vgg16 --miss 5 > FCI_vgg16_fmnist_OOD5.log &

nohup python -u FCI_fmnist.py --miss 10 > FCI_resnet18_fmnist_OOD10.log &
nohup python -u FCI_fmnist.py --net resnet34 --miss 10 > FCI_resnet34_fmnist_OOD10.log &
nohup python -u FCI_fmnist.py --net vgg16 --miss 10 > FCI_vgg16_fmnist_OOD10.log &