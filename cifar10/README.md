### for cifar10

nohup python -u FCI_cifar10.py --n_rep 1 > FCI_resnet18_cifar10_OOD0.log &
nohup python -u FCI_cifar10.py > FCI_resnet18_cifar10_OOD0.log &
nohup python -u FCI_cifar10.py --net resnet34 > FCI_resnet34_cifar10_OOD0.log &
nohup python -u FCI_cifar10.py --net vgg16 > FCI_vgg16_cifar10_OOD0.log &

nohup python -u FCI_cifar10.py --miss 5 > FCI_resnet18_cifar10_OOD5.log &
nohup python -u FCI_cifar10.py --net resnet34 --miss 5 > FCI_resnet34_cifar10_OOD5.log &
nohup python -u FCI_cifar10.py --net vgg16 --miss 5 > FCI_vgg16_cifar10_OOD5.log &

nohup python -u FCI_cifar10.py --miss 10 > FCI_resnet18_cifar10_OOD10.log &
nohup python -u FCI_cifar10.py --net resnet34 --miss 10 > FCI_resnet34_cifar10_OOD10.log &
nohup python -u FCI_cifar10.py --net vgg16 --miss 10 > FCI_vgg16_cifar10_OOD10.log &