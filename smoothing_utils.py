import torch

def hierarchical_smoothing(latent, cw=75, mw=45, fw=7):
    if cw > 1:
        latent[:,0:6,:] = smoothing(latent, cw, idx=0)
        # coarse_weight = torch.eye(latent.shape[2]).unsqueeze(-1).repeat(1,1,cw).to('cuda') / cw
        # coarse_input = latent[:,0:6,:].permute(1,2,0)
        # coarse_input = torch.nn.functional.pad(coarse_input, (cw//2, cw//2), mode='reflect')
        # coarse = torch.nn.functional.conv1d(coarse_input, coarse_weight, bias=None, stride=1)
        # latent[:,0:6,:] = coarse.permute(2,0,1)
    
    if mw > 1:
        latent[:,6:12,:] = smoothing(latent, mw, idx=6)
        # middle_weight = torch.eye(latent.shape[2]).unsqueeze(-1).repeat(1,1,mw).to('cuda') / mw
        # middle_input = latent[:,6:12,:].permute(1,2,0)
        # middle_input = torch.nn.functional.pad(middle_input, (mw//2, mw//2), mode='reflect')
        # middle = torch.nn.functional.conv1d(middle_input, middle_weight, bias=None, stride=1)
        # latent[:,6:12,:] = middle.permute(2,0,1)

    if fw > 1:
        latent[:,12:,:] = smoothing(latent, fw, idx=12)
        # fine_weight = torch.eye(latent.shape[2]).unsqueeze(-1).repeat(1,1,fw).to('cuda') / fw
        # fine_input = latent[:,12:,:].permute(1,2,0)
        # fine_input = torch.nn.functional.pad(fine_input, (fw//2, fw//2), mode='reflect')
        # fine = torch.nn.functional.conv1d(fine_input, fine_weight, bias=None, stride=1)
        # latent[:,12:,:] = fine.permute(2,0,1)
    return latent

def smoothing(input, window_size, idx=0):
    weight = torch.eye(input.shape[2]).unsqueeze(-1).repeat(1,1,window_size).to('cuda') / window_size
    triangular = torch.zeros(weight.shape[-1])
    triangular[:window_size//2+1] = torch.linspace(0.1, 2, window_size//2+1)
    triangular[window_size//2: ] = torch.linspace(2, 0.1, window_size//2+1)
    triangular /= torch.sum(triangular) / window_size
    weight = weight * triangular.cuda()
    selected_input = input[:,idx:idx+6,:].permute(1,2,0)
    selected_input = torch.nn.functional.pad(selected_input, (window_size//2, window_size//2), mode='reflect')
    coarse = torch.nn.functional.conv1d(selected_input, weight, bias=None, stride=1)
    return coarse.permute(2,0,1)

class EMA():
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        
    def forward(self,x, last_average):
        new_average = (1- self.mu)*x + self.mu*last_average
        return new_average
    
def moving_average(x, alpha=0.99):
    ema = EMA(alpha)
    x = x.T
    interpol = [x[:,0]]
    last_ave = x[:,0]
    for step in x.T[1:]:
        last_ave = ema.forward(step, last_ave)
        interpol.append(last_ave)

    interpoled = torch.Tensor(interpol)
    return interpoled