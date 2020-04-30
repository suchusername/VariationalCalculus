import numpy as np
import torch

class CVproblem:
    def __init__(self, f, t_0, t_f, x_0, x_f, known=None, x_exact=None):
        self.f = f
        self.t_0 = t_0
        self.t_f = t_f
        self.x_0 = x_0
        self.x_f = x_f
        if known is None:
            self.t_known = None
            self.x_known = None
        else:
            self.t_known, self.x_known = tuple(map(list, zip(*known)))
            self.t_known = torch.tensor(self.t_known).float().view(-1,1)
            self.x_known = torch.tensor(self.x_known).float().view(-1,1)
        self.x_exact = x_exact
        
    def compute_error(self, x):
        # x = (x_1, ..., x_N-1) - numpy array
        
        if self.x_exact is None:
            return None
        
        x_full = np.concatenate(([self.x_0], x))
        N_ = x.shape[0] + 1
        ts_ = np.linspace(self.t_0, self.t_f, N_+1)
        return np.mean(np.abs(x_full - self.x_exact(ts_[:-1])))

# constrained problem format:
# [f, t_0, t_f. x_0, x_f, known, x_exact]

constrained_problems = [
    
    CVproblem(lambda t,x,xp: x*x + t*xp + xp*xp, 0, 1, 0, 0.25, None,
             lambda t: 0.5 + (2 - np.e)/4/(np.e*np.e - 1) * np.exp(t) + (np.e - 2*np.e*np.e)/4/(np.e*np.e - 1) * np.exp(-t)),
    
    CVproblem(lambda x,y,yp: (y + yp) * (y + yp), 0, 1, 0, 1, None,
             lambda x: np.sinh(x) / np.sinh(1) ),
    
    CVproblem(lambda x,y,yp: 2*y/x + y*yp + x*x*yp*yp , 1, np.e, 1, 0, None,
             lambda x: (1 - np.log(x)) / x ),
    
    CVproblem(lambda x,y,yp: 2*y - y*yp + x*yp*yp , 1, 3, 1, 4, None,
             lambda x: x + np.log(x) / np.log(3) ),
    
    CVproblem(lambda x,y,yp: 4*y*y + yp*yp + 8*y, 0, np.pi / 4, -1, 0, None,
             lambda x: np.sinh(2*x) / np.sinh(np.pi / 2) - 1 ),
    
    CVproblem(lambda x,y,yp: yp*yp + y*y + 2*torch.exp(2*x)*y, 0, 1, 1/3, np.e*np.e/3, None,
             lambda x: np.exp(2*x) / 3 ),
    
    CVproblem(lambda x,y,yp: yp*yp + 4*y*y + 2*y*torch.cos(x), 0, np.pi/2, 0.8, np.exp(np.pi), None,
             lambda x: np.exp(2*x) - 0.2*np.cos(x) ),
    
    CVproblem(lambda x,y,yp: x*x*yp*yp + 12*y*y, -2, -1, 1/16, 1, None, 
             lambda x: 1/x/x/x/x ),
    
    CVproblem(lambda x,y,yp: 2*y + y*yp + x*x*yp*yp, 1, 2, 0, 1+np.log(2), None,
             lambda x: np.log(x) - 2/x + 2 ),
    
    CVproblem(lambda x,y,yp: (x*yp + y)*(x*yp + y), 1, 2, 1, 0.5, None,
             lambda x: 1/x ),
    
    CVproblem(lambda x,y,yp: (yp + y)*(yp + y) + 2*y*torch.sin(x), 0, np.pi, 0, 1, None,
             lambda x: np.sinh(x)/np.sinh(np.pi) - 0.5*np.sin(x) ),
    
    CVproblem(lambda x,y,yp: x*x*x + 0.5*y*y + 2*yp*yp, 0, 1, 0, 2, None,
             lambda x: 2 * np.sinh(0.5 * x) / np.sinh(0.5) ),
    
    CVproblem(lambda x,y,yp: x*yp*yp + y*y/x + 2*y*torch.log(x) / x, 1, 2, 0, 1 - np.log(2), None,
             lambda x: 2 / 3 * (x - 1/x) - np.log(x) ),
    
    CVproblem(lambda x,y,yp: 3*y*y/x/x/x + yp*yp/x + 8*y, 1, 2, 0, 8*np.log(2), None, 
             lambda x: x*x*x*np.log(x) ),
    
    CVproblem(lambda x,y,yp: x*yp*yp + y*y/x + 4*y, 1, 2, 0, 2*np.log(2), None,
             lambda x: x*np.log(x) ),
    
    CVproblem(lambda x,y,yp: yp*yp + 6*y*y/x/x - 32*y*torch.log(x), 1, 2, 3, 4*(4*np.log(2) + 3), None, 
             lambda x: x*x*(4*np.log(x) + 3) ),
    
    CVproblem(lambda x,y,yp: x*x*yp*yp + 2*y*y + 32*x*x*y*torch.log(x), 1, 2, -5, 4*(4*np.log(2) - 5), None, 
             lambda x: x*x*(4*np.log(x) - 5) ),
    
    CVproblem(lambda x,y,yp: x*yp*yp + 4*y*y/x - 18*y*torch.log(x), 1, 2, 2, 2*(3*np.log(2) + 2), None,
             lambda x: (14/15 + np.log(2))*x*x + (16/15 - np.log(2))/x/x + 9/4*np.log(x) ),
    
    CVproblem(lambda x,y,yp: x*yp*yp + 2*y*yp, 1, 2, 0, np.log(2), None,
             lambda x: np.log(x) ),
    
    CVproblem(lambda x,y,yp: x*yp*yp + y*yp + x*y, 1, 2, 1/8, 0.5 - np.log(2), None,
             lambda x: x*x/8 - np.log(x) ),
    
    CVproblem(lambda x,y,yp: (-y + 0.5*yp*yp) * torch.sin(x), np.pi/4, np.pi/2, -np.log(np.sqrt(2)), 0, None,
             lambda x: np.log(np.sin(x)) ),
    
    None
    
]













