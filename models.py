
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from pylab import *

class ReprogramMask(nn.Module):
    """Reprograms as 
    x = tanh(weight*mask)+pad(input)
    normalize(x)
    output = model(x)
    """
    def __init__(self, model, input_size, model_input_size, output_number):
        super().__init__()
        
        self.model = [model]
        self.output_number = output_number
        
        self.model[0].eval()
        for p in self.model[0].parameters():
            p.requires_grad=False
        
        
        inh, inw = input_size
        minh, minw = model_input_size
        
        self.padding = (minh-inh)//2, (minh-inh)//2 , (minw-inw)//2, (minw-inw)//2
        print(self.padding)
        
        self.mask = nn.Parameter(F.pad(torch.zeros(*input_size), self.padding, value=1.0), requires_grad=False)
        imshow(self.mask)
        # added weight
        self.weight = nn.Parameter(torch.randn(3,*model_input_size, requires_grad=True))
        
        
    def forward(self, input, plot_inp=False):
        padded_input = F.pad(input, self.padding)
        model_input = F.tanh(self.weight*self.mask)+padded_input
        normed_input = self.normalize_batch(model_input)
        if plot_inp:
            axis('off')
            imshow((model_input[0].detach().cpu().numpy().transpose([1,2,0])))
        out = self.model[0](normed_input)
        return out[:,:self.output_number]
    
    def normalize_batch(self, input, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        ms = torch.tensor(means, device=str(input.device)).view(1,3,1,1)
        ss = torch.tensor(stds, device=str(input.device)).view(1,3,1,1)
#         print(input.device)
#         print(ms.device)
#         out = input[:,:,:,:]
#         out[:,0,:,:] = (input[:,0,:,:] - means[0])/stds[0]
#         out[:,1,:,:] = (input[:,1,:,:] - means[1])/stds[1]
#         out[:,2,:,:] = (input[:,2,:,:] - means[2])/stds[2]
        return (input-ms)/ss

class Reprogram(nn.Module):
    """Reprograms as 
    x = sigmoid(2*(weight+input))
    normalize(x)
    output = model(x)
    """
    def __init__(self, model, model_input_size, output_number):
        super().__init__()
        
        self.model = [model]
        self.output_number = output_number
        
        self.model[0].eval()
        for p in self.model[0].parameters():
            p.requires_grad=False
        
        
        # added weight
        self.weight = nn.Parameter(torch.randn(3,*model_input_size, requires_grad=True))
        print(self.weight.requires_grad)
                
    def forward(self, input, plot_inp=False):
        model_input = F.sigmoid(2*(input+self.weight))
        normed_input = self.normalize_batch(model_input)
        if plot_inp:
            axis('off')
            imshow((model_input[0].detach().cpu().numpy().transpose([1,2,0])))
        out = self.model[0](normed_input)
        return out[:,:self.output_number]
    
    def normalize_batch(self, input, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        ms = torch.tensor(means, device=str(input.device)).view(1,3,1,1)
        ss = torch.tensor(stds, device=str(input.device)).view(1,3,1,1)
#         print(input.device)
#         print(ms.device)
#         out = input[:,:,:,:]
#         out[:,0,:,:] = (input[:,0,:,:] - means[0])/stds[0]
#         out[:,1,:,:] = (input[:,1,:,:] - means[1])/stds[1]
#         out[:,2,:,:] = (input[:,2,:,:] - means[2])/stds[2]
        return (input-ms)/ss

class ReprogramTanh(nn.Module):
    """Reprograms as 
    x = tanh(normalize(input)+weight)
    output = model(x)
    """
    def __init__(self, model, model_input_size, output_number):
        super().__init__()
        
        self.model = [model]
        self.output_number = output_number
        
        self.model[0].eval()
        for p in self.model[0].parameters():
            p.requires_grad=False
        
        
        # added weight
        self.weight = nn.Parameter(torch.randn(3,*model_input_size, requires_grad=True))
        print(self.weight.requires_grad)
                
    def forward(self, input, plot_inp=False):
        model_input = F.tanh(self.normalize_batch(input)+self.weight)
        if plot_inp:
            axis('off')
            imshow(0.5*(1+model_input[0].detach().cpu().numpy().transpose([1,2,0])))
        out = self.model[0](model_input)
        return out[:,:self.output_number]
    
    def normalize_batch(self, input, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        ms = torch.tensor(means, device=str(input.device)).view(1,3,1,1)
        ss = torch.tensor(stds, device=str(input.device)).view(1,3,1,1)
#         print(input.device)
#         print(ms.device)
#         out = input[:,:,:,:]
#         out[:,0,:,:] = (input[:,0,:,:] - means[0])/stds[0]
#         out[:,1,:,:] = (input[:,1,:,:] - means[1])/stds[1]
#         out[:,2,:,:] = (input[:,2,:,:] - means[2])/stds[2]
        return (input-ms)/ss

class ReprogramMult(nn.Module):
    """Reprograms as 
    x = sigmoid(weight)*input
    normalize(x)
    output = model(x)
    """
    def __init__(self, model, model_input_size, output_number):
        super().__init__()
        
        self.model = [model]
        self.output_number = output_number
        
        self.model[0].eval()
        for p in self.model[0].parameters():
            p.requires_grad=False
        
        
        # added weight
        self.weight = nn.Parameter(torch.randn(3,*model_input_size, requires_grad=True))
        print(self.weight.requires_grad)
                
    def forward(self, input, plot_inp=False):
        model_input = F.sigmoid(self.weight)*input
        normed_input = self.normalize_batch(model_input)
        if plot_inp:
            axis('off')
            imshow((model_input[0].detach().cpu().numpy().transpose([1,2,0])))
        out = self.model[0](normed_input)
        return out[:,:self.output_number]
    
    def normalize_batch(self, input, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        ms = torch.tensor(means, device=str(input.device)).view(1,3,1,1)
        ss = torch.tensor(stds, device=str(input.device)).view(1,3,1,1)
#         print(input.device)
#         print(ms.device)
#         out = input[:,:,:,:]
#         out[:,0,:,:] = (input[:,0,:,:] - means[0])/stds[0]
#         out[:,1,:,:] = (input[:,1,:,:] - means[1])/stds[1]
#         out[:,2,:,:] = (input[:,2,:,:] - means[2])/stds[2]
        return (input-ms)/ss