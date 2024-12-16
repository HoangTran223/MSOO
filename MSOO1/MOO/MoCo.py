import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MoCo:
    def __init__(self, beta=0.5, beta_sigma=0.5, gamma=0.1, gamma_sigma=0.5, rho=0):
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.gamma = gamma
        self.gamma_sigma = gamma_sigma
        self.rho = rho
        self.initialized = False

    def init_param(self, components):
        self.step = 0
        self.task_num = len(components)
        self.grad_dim = components[0].numel()
        self.device = components[0].device
        self.y = torch.zeros(self.task_num, self.grad_dim, device=self.device)
        self.lambd = torch.ones(self.task_num, device=self.device) / self.task_num
        self.initialized = True

    def apply(self, components, losses):
        if not self.initialized:
            self.init_param(components)
        
        self.step += 1
        
        grads = [comp.clone().view(-1) for comp in components]  # Ensure gradients are 1D tensors
        losses = torch.tensor(losses, device=self.device)

        with torch.no_grad():
            for tn in range(self.task_num):
                norm = grads[tn].norm() + 1e-8
                grads[tn] = grads[tn] / norm * losses[tn]

        grads_tensor = torch.stack(grads) 
        lr_y = self.beta / self.step ** self.beta_sigma
        self.y = self.y - lr_y * (self.y - grads_tensor)

        y_yy = self.y @ self.y.t()
        rho_eye = self.rho * torch.eye(self.task_num, device=self.device)
        lr_lambd = self.gamma / self.step ** self.gamma_sigma
        lambd_update = self.lambd - lr_lambd * (y_yy + rho_eye) @ self.lambd
        self.lambd = F.softmax(lambd_update, dim=-1)

        adjusted_grad = self.y.t() @ self.lambd  

        batch_weight = self.lambd.detach()

        return adjusted_grad, batch_weight
