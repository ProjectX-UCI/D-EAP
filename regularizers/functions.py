import torch

class Regularizers():
    def none(var):
        return 0

    def l1(var):
        return var.abs().sum()
    
    def l2(var):
        return var.pow(2).sum()

    # formalized in 3.1 of the paper
    # f_sigma(s_i) = exp( -1/2 * (s/sigma)^2 ))
    # MUST INCORPORATE SIGMA VALUE
    def static_l0(var):

        # fixed sigma value (abstract later)
        sigma = .5

        f_s = torch.exp(-1.0/2*torch.pow(var/sigma,2))
        return f_s.sum()
    
    # formalized in 3.2 of the paper
    def dynamic_l0(var):
        raise NotImplementedError