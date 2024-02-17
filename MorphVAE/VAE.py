# This file defines the VAE model architecture. 

import torch
from torch import nn
from custom_mlp import MLP,Exp
import pyro
import pyro.distributions as dist
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam
import numpy as np

class VAE(nn.Module):
    
    def __init__(self, y_size = 5, x_size = 5, h_size = 5, 
                hidden_layers_g = [500,], hidden_layers_p = [500,], y_emb=128, h_emb=128, 
                 config_enum = None, use_cuda = False):
        
        # y_size: number of tasks，x_size: shape of the robot design
        # h_size: unused，hidden_layers: #hidden units and #hidden layers
        
        super().__init__()
        
        self.y_size = y_size
        self.x_size = x_size
        self.h_size = h_size
        self.hidden_sizes_g = hidden_layers_g
        self.hidden_sizes_p = hidden_layers_p
        self.allow_broadcast = config_enum=="parallel"
        self.use_cuda = use_cuda
        self.h_emb = h_emb
        self.y_emb = y_emb
        
        self.operator_x = torch.zeros(5*self.x_size**2, 5*self.x_size**2)
        for i in range(self.x_size**2):
            self.operator_x[i*5:(i+1)*5,i*5:(i+1)*5] = 1
        
        self.setup_networks()
    
    # construct encoders and decoders
    def setup_networks(self):
        hidden_sizes_g = self.hidden_sizes_g
        hidden_sizes_p = self.hidden_sizes_p
        
        # posterior: voxels & latents -> task (not used, since task type is given during inference)
        self.encoder_y = MLP([5*self.x_size**2+self.h_emb]+hidden_sizes_p+[self.y_size],
                            activation=nn.ReLU,
                            output_activation=None,
                            allow_broadcast=self.allow_broadcast,
                            use_cuda=self.use_cuda)
        
        # posterior: voxels -> latents
        self.encoder_h = MLP([5*self.x_size**2]+hidden_sizes_p+[[self.h_emb, self.h_emb]],
                            activation=nn.ReLU,
                            output_activation=[None, Exp],
                            allow_broadcast=self.allow_broadcast,
                            use_cuda=self.use_cuda)
        
        # generator: task embedding -> latents
        self.decoder_h = MLP([self.y_emb]+hidden_sizes_g+[[self.h_emb, self.h_emb]],
                            activation=nn.ReLU,
                            output_activation=[None,Exp],
                            allow_broadcast=self.allow_broadcast,
                           use_cuda=self.use_cuda)
        
        #self.emb_h = torch.rand(self.h_size, self.h_emb,requires_grad=True)
        
        # embedding layer for tasks
        self.emb_y = MLP([3,128], activation=nn.Identity, output_activation=None, 
                        allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        
        # generator: latents -> voxels
        self.decoder_x = MLP([self.h_emb]+hidden_sizes_g+[5*self.x_size**2],
                           activation=nn.ReLU,
                           output_activation=None,
                           allow_broadcast=self.allow_broadcast,
                           use_cuda=self.use_cuda)
        
        # enable GPU for faster training
        if self.use_cuda:
            self.cuda()
            
    # define the generative process
    @config_enumerate(default="parallel")
    def model(self, xs, ys=None, hs=None):
        pyro.module("ss_vae", self)
        if xs==None:
            batch_size = 1
        else:
            batch_size = xs.size(0)
            
        with pyro.plate("data"):
            
            # task
            alpha_prior = torch.ones(batch_size, self.y_size)/(1.0*self.y_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior).to_event(1), obs=ys)
            ys_emb = self.emb_y.forward(ys)
            
            # latents
            loc,scale = self.decoder_h.forward(ys_emb)
            hs_emb = pyro.sample("h", dist.Normal(loc, scale).to_event(1))
           
            # voxels
            xs_prob = self.decoder_x.forward([hs_emb])
            if xs!=None:
                xs = pyro.sample("xs", dist.OneHotCategorical(logits=xs_prob.reshape(-1,25,5)).to_event(2), obs=xs.reshape(-1,25,5))
            else:
                xs = pyro.sample("xs", dist.OneHotCategorical(logits=xs_prob.reshape(-1,25,5)).to_event(2), obs=None)
        
        return (xs.reshape(-1,self.x_size**2*5), ys, hs_emb, xs_prob)  
        
    # define the approximate posteriors
    @config_enumerate(default="parallel")
    def guide(self, xs, ys=None, hs=None):
        with pyro.plate("data"):
            
            # inference of h (x->h)
            if hs is None:
                loc,scale = self.encoder_h.forward(xs)
                hs_emb = pyro.sample("h", dist.Normal(loc, scale).to_event(1))
            
            # inference of y (x+h->y)
            if ys is None:
                hs_emb = self.emb_h.forward(hs)
                alpha = self.encoder_y.forward([xs,hs_emb])
                ys = pyro.sample("y", dist.OneHotCategorical(logits=alpha).to_event(1)) 
                
        return(xs.reshape(-1,self.x_size**2*5), ys, hs_emb)