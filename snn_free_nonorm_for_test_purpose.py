#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from vgg_spiking_nodewise import *
import sys
import os
import shutil
import argparse
from attack_model_spike_cnt import Attack


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(epoch, test_loader, best_Acc, args):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    avg_spike_cnt = []

    if args.dataset == 'CIFAR10':
        mean = torch.Tensor(np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    if args.dataset == 'CIFAR100':
        mean = torch.Tensor(np.array([0.5071,0.4867,0.4408])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2675,0.2565,0.2761])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    
    if args.test_only:
        temp1 = []  
        temp2 = []  
        for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):    
            temp1 = temp1+[round(value.item(),2)]   
        for key, value in sorted(model.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))): 
            temp2 = temp2+[round(value.item(),2)]   
        f.write('\n Thresholds: {}, leak: {}'.format(temp1, temp2))

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            data = data - mean
            data.div_(std)
            output, spike_count  = model(data) 
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                f.write('\n Images {}/{} Accuracy: {}/{}({:.4f})'
                    .format(
                    test_loader.batch_size*(batch_idx+1),
                    len(test_loader.dataset),
                    correct.item(),
                    data.size(0),
                    top1.avg*100
                    )
                )
        
        temp1 = []
        temp2 = []
        for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[value.item()]
        for key, value in sorted(model.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[value.item()]
        
        if epoch>5 and top1.avg<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, time: {}'
            .format(
            losses.avg, 
            top1.avg*100,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        best_Acc.append(top1.avg*100)
        return top1.avg, best_Acc


def validate_fgsm(val_loader, model, args, eps=0.031):
    if args.dataset == 'CIFAR10':
        mean = torch.Tensor(np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    if args.dataset == 'CIFAR100':
        mean = torch.Tensor(np.array([0.5071,0.4867,0.4408])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2675,0.2565,0.2761])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    losses = AverageMeter('Loss')
    prec1_fgsm = 0.0 
    prec5_fgsm = 0.0
    n = 0
    model.eval()
    attacker = Attack(dataloader=val_loader,
                          attack_method='fgsm', epsilon=0.031) 
    for i, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        n += target.size(0)
        data.requires_grad = False
        perturbed_data = attacker.attack_method(model, data, target, args)
        perturbed_data.sub_(mean).div_(std)
        output_fgsm,_ = model(perturbed_data)
        loss_fgsm = F.cross_entropy(output_fgsm, target)
        _, pred_fgsm = output_fgsm.topk(5, 1, largest=True, sorted=True)
        target_fgsm = target.view(target.size(0),-1).expand_as(pred_fgsm)
        correct_fgsm = pred_fgsm.eq(target_fgsm).float()
        prec1_fgsm += correct_fgsm[:,:1].sum()
        prec5_fgsm += correct_fgsm[:,:5].sum()
        losses.update(loss_fgsm.item(), data.size(0))

    top1_fgsm = 100.*(prec1_fgsm/float(n))    
    top5_fgsm = 100.*(prec5_fgsm/float(n))
    print('\n Top1 FGSM:{}'.format(top1_fgsm))
    return top1_fgsm


def validate_pgd(val_loader, model, eps=0.031, K=7, a=0.01):
    if args.dataset == 'CIFAR10':
        mean = torch.Tensor(np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    if args.dataset == 'CIFAR100':
        mean = torch.Tensor(np.array([0.5071,0.4867,0.4408])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2675,0.2565,0.2761])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()
    losses = AverageMeter('Loss')
    prec1_pgd = 0.0 
    prec5_pgd = 0.0
    n = 0
    model.eval()
    print('Value of K:{}, eps:{}'.format(K, eps))
    for i, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        n += target.size(0)
        orig_input = data.clone()
        randn = torch.FloatTensor(data.size()).uniform_(-eps, eps).cuda()
        data += randn
        data.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(data, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output,_ = model(in1)
            ascend_loss = F.cross_entropy(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = torch.sign(ascend_grad)*a
            data += pert.data
            data = torch.max(orig_input-eps, data)
            data = torch.min(orig_input+eps, data)
            data.clamp_(0, 1.0)
        data.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output,_ = model(data)
            loss_pgd = F.cross_entropy(output, target)

            # measure accuracy and record loss
            _, pred_pgd = output.topk(5, 1, largest=True, sorted=True)
            target_pgd = target.view(target.size(0),-1).expand_as(pred_pgd)
            correct_pgd = pred_pgd.eq(target_pgd).float()
            prec1_pgd += correct_pgd[:,:1].sum()
            prec5_pgd += correct_pgd[:,:5].sum()
            losses.update(loss_pgd.item(), data.size(0))
            
    top1_pgd = 100.*(prec1_pgd/float(n))    
    top5_pgd = 100.*(prec5_pgd/float(n))
    print('\n Top1 PGD:{}'.format(top1_pgd))
    return top1_pgd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','IMAGENET', 'TINY_IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG4','VGG5','VGG6','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--pgd_iter',               default=7,                  type=int,       help='number of pgd iterations')
    parser.add_argument('--pgd_step',               default=0.01,               type=float,       help='pgd attack step size')
    parser.add_argument('--epochs',                 default=30,                 type=int,       help='number of training epochs')
    parser.add_argument('--timesteps',              default=20,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear'])
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--resume',                 default='',                 type=str,       help='resume training from this state')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    timesteps           = args.timesteps
    leak                = args.leak
    default_threshold   = args.default_threshold
    activation          = args.activation
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    resume              = args.resume
    start_epoch         = 1
    max_accuracy        = 0.0
    
    log_file = './logs/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+str(datetime.datetime.now())
    log_file+=identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout
    
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))   

    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                            ])
        transform_test  = transforms.Compose([transforms.ToTensor()])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = '../SNN_adversary/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='../SNN_adversary/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
        train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = './cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
        train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, dropout=0.2, kernel_size=kernel_size, dataset=dataset)
  
    if pretrained_snn:
                
        state = torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    else:
        print("Please provide an snn file name")
    
    #model = nn.DataParallel(model) 

    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    
    if resume:
        f.write('\n Resuming from checkpoint {}'.format(resume))
        state = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        #f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
              
        #epoch           = state['epoch']
        start_epoch     = epoch + 1
        #max_accuracy    = state['accuracy']
        #optimizer.load_state_dict(state['optimizer'])
        for param_group in optimizer.param_groups:
            learning_rate =  param_group['lr']

        f.write('\n Loaded from resume epoch: {}, accuracy: {:.4f} lr: {:.1e}'.format(epoch, max_accuracy, learning_rate))

    test_Acc = [0]
    pgd_test_acc = [0]
    fgsmtest_acc = [0]
    for epoch in range(start_epoch, epochs+1):
        start_time = datetime.datetime.now()
        top1, test_Acc = test(epoch, test_loader, test_Acc, args)
        top1_fgsm = validate_fgsm(test_loader, model, args, eps=0.031)
        top1_pgd = validate_pgd(test_loader, model, eps=0.031, K=7, a=0.01)
        fgsmtest_acc.append(top1_fgsm)
        pgd_test_acc.append(top1_pgd)
        #print('Epoch:{}, TestAcc:{}, PGD acc:{}'.format(epoch, top1, top1_pgd))
        print('Epoch:{}, TestAcc:{}, FGSM acc:{}, PGD acc:{}'.format(epoch, top1*100, top1_fgsm, top1_pgd))
    
    #f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))




