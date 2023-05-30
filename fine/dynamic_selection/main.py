import torch
t = torch.tensor([1.0]) # create tensor with just a 1 in it
t = t.cuda() # Move t to the gpu
print(t) # Should print something like tensor([1], device='cuda:0')
print(t.mean()) # Test an operation just to be sure

from traintools import *
from utils.args import *
import warnings
warnings.filterwarnings("ignore")

def main(parse, config: ConfigParser):
    
    if parse.traintools == 'robustloss':
        robustlosstrain(parse, config)
    elif parse.traintools == 'robustlossgt':
        gtrobustlosstrain(parse, config)
    elif parse.traintools == 'trainingclothing1m':
        trainClothing1m(parse, config)
    elif parse.traintools == 'coteaching':
        coteachingtrain(parse, config)
    elif parse.traintools == 'instance':
        traininstance(parse, config)
    else:
        raise NotImplemented
        
if __name__ == '__main__':
    config, parse = parse_args()
    
    ### TRAINING ###
    main(parse, config)
