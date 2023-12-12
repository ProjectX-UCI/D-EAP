import torch

# fixed sigma value (abstract later)
sigma = .5

class Regularizers():

    @staticmethod
    def none(var):
        return 0

    @staticmethod
    def l1(var):
        return var.abs().sum()
    
    @staticmethod
    def l2(var):
        return var.pow(2).sum()

    # formalized in 3.1 of the paper
    # f_sigma(s) = 1 - exp( -1/2 * (s/sigma)^2 ))
    @staticmethod
    def static_l0(var):

        f_s = 1 - torch.exp((-1.0/2)*torch.pow(var/sigma,2))
        return f_s.sum()
    
    # formalized in 3.2 of the paper
    @staticmethod
    def dynamic_l0(var):
        raise NotImplementedError