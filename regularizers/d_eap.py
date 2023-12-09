import torch

class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError

class D_EAP(_Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, model, lambda_reg=0.01):
        super(D_EAP, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * D_EAP.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * D_EAP.__add_l0(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l0(var):

        # fixed sigma value (abstract later)
        sigma = .5

        # 1. get all node parameters
        node_params = var
        print(node_params)

        # 2. apply f_sigma(s_i) = exp( -1/2 * (s/sigma)^2 ))

        sigma = torch.tensor(sigma,dtype=torch.float32)
        
        # just declare as -1/2, no tensor
        coefficient = torch.tensor(-1/2,dtype=torch.float32)

        b = torch.exp(coefficient*torch.pow(node_params/sigma,2))
        print(b)

        # 3. sum together transformed parameters
        c = b.sum()
        print(c)

        # 4. length - sum
        # d = node_params.size() - c
        # print(d)

        return c




