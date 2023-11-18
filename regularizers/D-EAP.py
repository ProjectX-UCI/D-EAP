from regularizer import _Regularizer

class D_EAP(_Regularizer):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(D_EAP, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError

