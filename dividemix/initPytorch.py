import torch 
s = 32
dev = torch.device('cuda')
torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
