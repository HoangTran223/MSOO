import torch

class MGDA:
    def __init__(self):
        pass

    def apply(self, components):
        device = components[0].device
        task_num = len(components)
        grad_vec = torch.stack([g.clone().view(-1) for g in components]) 
        
        grad_mat = grad_vec @ grad_vec.t()  
        
        if task_num == 1:
            weights = torch.tensor([1.0], device=device)
        elif task_num == 2:
            g1, g2 = grad_vec[0], grad_vec[1]
            g1g1 = torch.dot(g1, g1)
            g1g2 = torch.dot(g1, g2)
            g2g2 = torch.dot(g2, g2)
            gamma, _ = self._min_norm_element_from2(g1g1, g1g2, g2g2)
            weights = torch.tensor([gamma, 1 - gamma], device=device)
        else:
            weights = self._min_norm_solver(grad_vec)
        adjusted_grad = grad_vec.t() @ weights 
        
        return adjusted_grad, weights

    def _min_norm_element_from2(self, v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)
        cost = v1v1 * gamma * gamma + v2v2 * (1 - gamma) * (1 - gamma) + 2 * v1v2 * gamma * (1 - gamma)
        return gamma, cost

    def _min_norm_solver(self, grad_vec, max_iter=250, stop_tol=1e-5):
        device = grad_vec.device
        task_num = grad_vec.size(0)
        sol_vec = torch.ones(task_num, device=device) / task_num  
        grad_mat = grad_vec @ grad_vec.t()  

        k = 0
        while k < max_iter:
            grad_dir = grad_mat @ sol_vec  
            idx_min = torch.argmin(grad_dir)
            v = torch.zeros(task_num, device=device)
            v[idx_min] = 1.0

            gap = (sol_vec - v) @ grad_dir
            if gap.item() < stop_tol:
                break

            t = gap / ((sol_vec - v) @ (grad_mat @ (sol_vec - v)))
            t = t.item()
            t = max(0.0, min(1.0, t))

            sol_vec = sol_vec + t * (v - sol_vec)
            k += 1

        return sol_vec