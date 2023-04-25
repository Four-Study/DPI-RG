### for cifar10

nohup python -u FPI_cifar10.py --n_rep 1 > FPI_resnet18_cifar10_OOD0.log &
nohup python -u FPI_cifar10.py > FPI_resnet18_cifar10_OOD0.log &
nohup python -u FPI_cifar10.py --net resnet34 > FPI_resnet34_cifar10_OOD0.log &
nohup python -u FPI_cifar10.py --net vgg16 > FPI_vgg16_cifar10_OOD0.log &

nohup python -u FPI_cifar10.py --miss 5 > FPI_resnet18_cifar10_OOD5.log &
nohup python -u FPI_cifar10.py --net resnet34 --miss 5 > FPI_resnet34_cifar10_OOD5.log &
nohup python -u FPI_cifar10.py --net vgg16 --miss 5 > FPI_vgg16_cifar10_OOD5.log &

nohup python -u FPI_cifar10.py --miss 10 > FPI_resnet18_cifar10_OOD10.log &
nohup python -u FPI_cifar10.py --net resnet34 --miss 10 > FPI_resnet34_cifar10_OOD10.log &
nohup python -u FPI_cifar10.py --net vgg16 --miss 10 > FPI_vgg16_cifar10_OOD10.log &