# RESOLVES CONFUSING ERROR
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from regularizers.d_eap import D_EAP
# from utils.data_utils import load_data
from utils.training_utils import training_loop,package_model_components
from models.model import Test
# from utils.evaluation_utils import plotDF

import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

if __name__ == '__main__':
    list_of_regularizers = [D_EAP]
    model_components = package_model_components(Test,list_of_regularizers)
    model_package = model_components[0]
    
    inputs = torch.tensor([1,2],dtype=torch.float32)
    labels = torch.tensor([1],dtype=torch.float32)

    loss = training_loop(model_package,inputs,labels)
    print(loss)