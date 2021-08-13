import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import copy
import numpy as np

class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd 
        
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
            elif attack_method == 'pgd':
                self.attack_method = self.pgd

    ## For SNN pgd takes two more args: mean and std to manually perform normalization for 
    ## each of the k iterated perturbed data generated intermediately.                               
    def fgsm(self, model, data, target, args, data_min=0, data_max=1):
        
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

        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        #As we take the raw un-normalized data, we convert to a normalized data
        # and then feed to model
        perturbed_data_norm = perturbed_data -mean
        perturbed_data_norm.div_(std)
        output,_ = model(perturbed_data_norm)
        #print('perturbed_data.requires_grad:', perturbed_data.requires_grad) 
        loss = F.cross_entropy(output, target)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward()
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data += self.epsilon*sign_data_grad
            # Adding clipping to maintain [min,max] range, default 0,1 for image
            perturbed_data.clamp_(data_min, data_max)
    
        return perturbed_data
        
    ## For SNN pgd takes two more args: mean and std to manually perform normalization for 
    ## each of the k iterated perturbed data generated intermediately.
    def pgd(self, model, data, target, k=7, a=0.01, random_start=True,
               d_min=0, d_max=1): #to reduce time for SNN kept k = 3, or else for ANN we use k=7 
        
        mean = torch.Tensor(np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 32, 32).cuda()
        std = torch.Tensor(np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 32, 32).cuda()

        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()                                     
        perturbed_data.requires_grad = True
        
        data_max = data + self.epsilon
        data_min = data - self.epsilon
        data_max.clamp_(d_min, d_max)
        data_min.clamp_(d_min, d_max)

        if random_start:
            with torch.no_grad():
                perturbed_data.data = data + perturbed_data.uniform_(-1*self.epsilon, self.epsilon)
                perturbed_data.data.clamp_(d_min, d_max)
        
        for _ in range(k):
            ##for SNNs we don't have a mean, std layer separately, so we manually do mean
            ## subtraction here with every perturbed data generated

            in1 = perturbed_data - mean
            in1.div_(std)
            output,_ = model( in1 )
            #print('output shape:{}, target shape:{}', output.shape, target.shape) 
            loss = F.cross_entropy(output, target)
            
            if perturbed_data.grad is not None:
                perturbed_data.grad.data.zero_()
            
            loss.backward()
            data_grad = perturbed_data.grad.data
            
            with torch.no_grad():
                perturbed_data.data += a * torch.sign(data_grad)
                perturbed_data.data = torch.max(torch.min(perturbed_data, data_max),
                                                data_min)
        perturbed_data.requires_grad = False
        
        return perturbed_data
