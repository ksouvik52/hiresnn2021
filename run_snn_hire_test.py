import os
import sys

########################################################################
#### ~~~~~~               White box testing       ~~~~~~
## Hereere, pretrained_snn is the model on which wb test will be done.
########################################################################

cmd1 = "python snn_free_nonorm_for_test_purpose.py --dataset CIFAR100 --batch_size 32 --architecture VGG11 \
        --epochs 1 --timesteps 8 --leak 1.0 --devices 1 \
        --pretrained_snn='HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_model.pt'"
os.system(cmd1)


########################################################################
#### ~~~~~~               Black box testing       ~~~~~~
## Here, pretrained_snn is the model on which bb test will be done.
## and pretrained_snn_bb is the model that generates the adversarial images.
## We take models of same variants trained with different seed. 
########################################################################
cmd2 = "python snn_free_nonorm_for_bbtest_purpose.py --dataset CIFAR100 --batch_size 32 --architecture VGG11 \
        --epochs 1 --timesteps 8 --leak 1.0 --devices 1 \
        --pretrained_snn='HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_model.pt'\
        --pretrained_snn_bb='HIRE_SNN_models/vgg11_cifar100_hiresnn_tstep8_bb_test_model.pt'"
os.system(cmd2)
