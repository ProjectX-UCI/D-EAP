import torch

# fixed sigma value (abstract later)
sigma = .5

class Regularizers():
    """Class containing all regularization terms to be compared and measured in this study

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.Tensor: regularization value for given parameter vector
    """

    @staticmethod
    def none(var):
        return 0

    @staticmethod
    def l1(var):
        return var.abs().sum()
    
    @staticmethod
    def l2(var):
        return var.pow(2).sum()

    @staticmethod
    def static_l0(var):
        """Static l0 norm approximator, formalized in section 3.1 of our paper.
        Formula: f_sigma(s) = 1 - exp( -1/2 * (s/sigma)^2 ))

        Args:
            var torch.Tensor: parameter vector

        Returns:
            torch.Tensor: regularization value for given parameter vector
        """

        f_s = 1 - torch.exp((-1.0/2)*torch.pow(var/sigma,2))
        return f_s.sum()
    
    # formalized in 3.2 of the paper
    @staticmethod
    def dynamic_l0(var):
        raise NotImplementedError