#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

cfg = {
	'VGG4' : [64, 'A', 128, 'A'],
	'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class VGG_SNN(nn.Module):

	def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, dropout=0.2, kernel_size=3, dataset='CIFAR10'):
		super().__init__()
		
		self.vgg_name 		= vgg_name
		self.act_func 		= LinearSpike.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		self.dropout 		= dropout
		self.kernel_size 	= kernel_size
		self.dataset 		= dataset
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		
		self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
		
		self._initialize_weights2()

		threshold 	= {}
		lk 	  		= {}
		for l in range(len(self.features)):
			if isinstance(self.features[l], nn.Conv2d):
				threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(l)]			= nn.Parameter(torch.tensor(leak))
								
				
		prev = len(self.features)
		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				threshold['t'+str(prev+l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(prev+l)] 		= nn.Parameter(torch.tensor(leak))
				
		self.threshold 	= nn.ParameterDict(threshold)
		self.leak 		= nn.ParameterDict(lk)
		
	def _initialize_weights2(self):
		for m in self.modules():
            
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):

		# Initialize thresholds
		self.scaling_factor = scaling_factor
		
		for pos in range(len(self.features)):
			if isinstance(self.features[pos], nn.Conv2d):
				if thresholds:
					self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})
				#print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))

		prev = len(self.features)

		for pos in range(len(self.classifier)-1):
			if isinstance(self.classifier[pos], nn.Linear):
				if thresholds:
					self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})
				#print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))


	def _make_layers(self, cfg):
		layers 		= []
		if self.dataset =='MNIST':
			in_channels = 1
		else:
			in_channels = 3

		for x in (cfg):
			stride = 1
						
			if x == 'A':
				layers.pop()
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
							nn.ReLU(inplace=True)
							]
				layers += [nn.Dropout(self.dropout)]
				in_channels = x

		features = nn.Sequential(*layers)
		
		layers = []
		if self.dataset == 'IMAGENET':
			layers += [nn.Linear(512*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG6' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*4*4, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG4' and self.dataset== 'MNIST':
			layers += [nn.Linear(128*7*7, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			#layers += [nn.Linear(4096, 4096, bias=False)]
			#layers += [nn.ReLU(inplace=True)]
			#layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(1024, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG16' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif (self.vgg_name == 'VGG5' or self.vgg_name == 'VGG11') and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2*4, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG6' and self.dataset == 'MNIST':
			layers += [nn.Linear(128*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name != 'VGG6' and self.dataset == 'MNIST':
			layers += [nn.Linear(512*1*1, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]


		classifer = nn.Sequential(*layers)
		return (features, classifer)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps

	def timestep_update(self, timesteps):
		self.timesteps 	= timesteps
			
	def neuron_init(self, x):
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)			
		
		self.mem 	= {}
		self.spike 	= {}
		self.mask 	= {}
		self.spike_count= {}

		for l in range(len(self.features)):
								
			if isinstance(self.features[l], nn.Conv2d):
				self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
				#this is to just keep track what happens to spike after conv
				
                

			elif isinstance(self.features[l], nn.ReLU):
				if isinstance(self.features[l-1], nn.Conv2d):
					self.spike[l] 	= torch.ones(self.mem[l-1].shape)*(-1000)
					self.spike_count[l] = torch.zeros(self.mem[l-1].size())

				elif isinstance(self.features[l-1], nn.AvgPool2d):
					self.spike[l] 	= torch.ones(self.batch_size, self.features[l-2].out_channels, self.width, self.height)*(-1000)
                    

			elif isinstance(self.features[l], nn.Dropout):
				self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape).cuda())

			elif isinstance(self.features[l], nn.AvgPool2d):
				self.width = self.width//self.features[l].kernel_size
				self.height = self.height//self.features[l].kernel_size
				self.spike_count[l] = torch.zeros(self.batch_size, self.features[l-2].out_channels,self.width,self.height)
                
		
		prev = len(self.features)

		for l in range(len(self.classifier)):
			
			if isinstance(self.classifier[l], nn.Linear):
				self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features)
                
			
			elif isinstance(self.classifier[l], nn.ReLU):
				self.spike[prev+l] 		= torch.ones(self.mem[prev+l-1].shape)*(-1000)
				self.spike_count[prev+l] 	= torch.zeros(self.mem[prev+l-1].size())

			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape).cuda())
				
		
	def percentile(self, t, q):

		k = 1 + round(.01 * float(q) * (t.numel() - 1))
		result = t.view(-1).kthvalue(k).values.item()
		return result
		
	def forward(self, x, find_max_mem=False, spike_count=[], max_mem_layer=0):
		
		self.neuron_init(x)
		max_mem=0.0
				
		for t in range(self.timesteps):
			out_prev = x
			
			for l in range(len(self.features)):
				
				if isinstance(self.features[l], (nn.Conv2d)):
					
					if find_max_mem and l==max_mem_layer:
						cur = self.percentile(self.features[l](out_prev).view(-1), 99.7)
						if (cur>max_mem):
							max_mem = torch.tensor([cur])
						break

					delta_mem 		= self.features[l](out_prev)
					self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + delta_mem
					mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					self.mem[l] 	= self.mem[l]-rst

				elif isinstance(self.features[l], nn.ReLU):
					
					out 			= self.act_func(mem_thr, (t-1-self.spike[l]))
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
					self.spike_count[l][out.bool()] 	= self.spike_count[l][out.bool()] + 1
					out_prev  		= out.clone()

				elif isinstance(self.features[l], nn.AvgPool2d):
					out_prev 		= self.features[l](out_prev)
					self.spike_count[l][out_prev.bool()] 	= self.spike_count[l][out_prev.bool()] + 1
                    
				
				elif isinstance(self.features[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]
			
			if find_max_mem and max_mem_layer<len(self.features):
				continue

			out_prev       	= out_prev.reshape(self.batch_size, -1)
			prev = len(self.features)
			
			for l in range(len(self.classifier)-1):
													
				if isinstance(self.classifier[l], (nn.Linear)):
					
					if find_max_mem and (prev+l)==max_mem_layer:
						cur = self.percentile(self.classifier[l](out_prev).view(-1),99.7)
						if cur>max_mem:
							max_mem = torch.tensor([cur])
						break
					
					delta_mem 			= self.classifier[l](out_prev)
					self.mem[prev+l]	= getattr(self.leak, 'l'+str(prev+l)) * self.mem[prev+l] + delta_mem
					mem_thr 			= (self.mem[prev+l]/getattr(self.threshold, 't'+str(prev+l))) - 1.0
					rst 				= getattr(self.threshold,'t'+str(prev+l)) * (mem_thr>0).float()	
					self.mem[prev+l]	= self.mem[prev+l]-rst


				elif isinstance(self.classifier[l], nn.ReLU):
					out 				= self.act_func(mem_thr, (t-1-self.spike[prev+l]))
					self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
					self.spike_count[prev+l][out.bool()] 	= self.spike_count[prev+l][out.bool()] + 1
					out_prev  			= out.clone()

				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[prev+l]
			
			# Compute the classification layer outputs
			if not find_max_mem:
				self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
		if find_max_mem:
			return max_mem

		return self.mem[prev+l+1], self.spike_count



