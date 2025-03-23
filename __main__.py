from train import conv_model
import torch
from torch import nn,save,load





if __name__ == '__main__':
    with open('model_state.pt', 'rb') as f: 
        conv_model.load_state_dict(load(f))  
