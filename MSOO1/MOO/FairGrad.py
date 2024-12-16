import torch
import numpy as np
from scipy.optimize import least_squares

class FairGrad:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def apply(self, components):
        device = components[0].device
        task_num = len(components)
        grad_dim = components[0].numel()

        grads = torch.stack([g.view(-1) for g in components]) 

        GTG = grads @ grads.t()  
        A = GTG.cpu().numpy()

        x_start = np.ones(task_num) / task_num

        def objfn(x):
            return A @ x - np.power(1 / x, 1 / self.alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        weights = torch.tensor(w_cpu, device=device, dtype=grads.dtype)

        adjusted_grad = torch.zeros_like(grads[0], device=device)
        for i in range(task_num):
            adjusted_grad += weights[i] * grads[i]

        return adjusted_grad, weights
