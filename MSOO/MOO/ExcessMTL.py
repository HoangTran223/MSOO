import torch

class ExcessMTL:
    def __init__(self, task_num, robust_step_size=0.1):
        self.task_num = task_num
        self.robust_step_size = robust_step_size
        self.loss_weight = torch.ones(task_num, requires_grad=False)
        self.grad_sum = None
        self.first_epoch = True
        self.initial_w = None

    def apply(self, components):
        device = components[0].device
        dtype = components[0].dtype
        task_num = len(components)
        grad_dim = components[0].numel()

        grads = torch.stack([g.view(-1) for g in components])  # Shape: (task_num, grad_dim)

        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(grads)

        w = torch.zeros(task_num, device=device, dtype=dtype)

        for i in range(task_num):
            self.grad_sum[i] += grads[i] ** 2
            h_i = torch.sqrt(self.grad_sum[i] + 1e-7)
            w[i] = grads[i].dot(grads[i] / h_i)

        if self.first_epoch:
            self.initial_w = w.clone()
            self.first_epoch = False
            self.loss_weight = torch.ones(task_num, device=device, dtype=dtype)
        else:
            w_adjusted = w / self.initial_w
            self.loss_weight *= torch.exp(w_adjusted * self.robust_step_size)
            self.loss_weight /= self.loss_weight.sum() / task_num
            self.loss_weight = self.loss_weight.detach()

        adjusted_grad = torch.zeros(grad_dim, device=device, dtype=dtype)
        for i in range(task_num):
            adjusted_grad += self.loss_weight[i] * grads[i]

        return adjusted_grad, self.loss_weight
