import numpy as np
from functools import partial

# Исходная постановка задачи

def f(t, x, xp):
    return x*x + t*xp + xp*xp

# boundary conditions
t_0 = 0
t_f = 1
x_0 = 0
x_f = 0.25

def f_prime_x(t, x, xp):
    return 2*x

def f_prime_xp(t, x, xp):
    return t + 2*xp

def f_prime_x_x(t, x, xp):
    return 2 + 0*t

def f_prime_x_xp(t, x, xp):
    return 0 + 0*t

def f_prime_xp_xp(t, x, xp):
    return 2 + 0*t

def x_exact(t):
    return 0.5 + (2 - np.e)/4/(np.e*np.e - 1) * np.exp(t) + (np.e - 2*np.e*np.e)/4/(np.e*np.e - 1) * np.exp(-t)

# Дискретизация задачи

def xp_approx_jac_h(N):
    # returns jacobian of def_approx of shape N *(N-1) - contains partial derivatives of xp(x_1, ..., x_N-1)
    
    # approximation of derivatives can be calculated as: xp = 1/h * J * x, xp_0 += -1/h * x_0, xp_N-1 += 1/h * x_N
    # where J is a return of this function 
    
    J = np.zeros((N, N-1))
    idx = np.arange(N-1)
    J[idx, idx] = 1
    J[idx+1, idx] = -1
    
    return J

def J(x, t, x_0, x_f, xp_jac, f):
    # t = (t_0, ..., t_N), x = (x_1, ..., x_N-1)
    # x_0 and x_f - given boundary conditions
    # xp_jac - matrix-jacobian (multiplied by h!) of xp(x_1, ..., x_N-1) of shape N * (N-1)
    # f - given function
    
    N = t.shape[0] - 1 # number of points
    h = t[1] - t[0] # time step
    x_full = np.concatenate(([x_0], x)) # vector (x_0, ..., x_N-1)
    
    xp = xp_jac @ x / h # calculating derivative approximations
    xp[0] += (-1) * x_0 / h
    xp[N-1] += x_f / h    
    
    return np.sum(f(t[:-1], x_full, xp)) * h


def J_grad(x, t, x_0, x_f, xp_jac, f_prime_x, f_prime_xp):
    # t = (t_0, ..., t_N), x = (x_1, ..., x_N-1)
    # x_0 and x_f - given boundary conditions
    # xp_jac - matrix-jacobian (multiplied by h!) of xp(x_1, ..., x_N-1) of shape N * (N-1)
    # f_prime_x, f_prime_xp - partial derivatives of f
    
    N = t.shape[0] - 1 # number of points
    h = t[1] - t[0] # time step
    x_full = np.concatenate(([x_0], x)) # vector (x_0, ..., x_N-1)
    
    xp = xp_jac @ x / h # calculating derivative approximations
    xp[0] += (-1) * x_0 / h
    xp[N-1] += x_f / h    
    
    return h * f_prime_x(t[1:-1], x_full[1:], xp[1:]) + xp_jac.T @ f_prime_xp(t[:-1], x_full, xp)


def J_hessian(x, t, x_0, x_f, xp_jac, f_prime_x_x, f_prime_x_xp, f_prime_xp_xp):
    # t = (t_0, ..., t_N), x = (x_1, ..., x_N-1)
    # x_0 and x_f - given boundary conditions
    # xp_jac - matrix-jacobian (multiplied by h!) of xp(x_1, ..., x_N-1) of shape N * (N-1)
    # f_prime_x_x, f_prime_x_xp, f_prime_xp_xp - second partial derivatives of f 
    
    N = t.shape[0] - 1 # number of points
    h = t[1] - t[0] # time step
    x_full = np.concatenate(([x_0], x)) # vector (x_0, ..., x_N-1)
    
    xp = xp_jac @ x / h # calculating derivative approximations
    xp[0] += (-1) * x_0 / h
    xp[N-1] += x_f / h 
    
    R = np.diag(f_prime_x_xp(t[1:-1], x_full[1:], xp[1:])) @ xp_jac[1:] # auxilliary matrix
    
    return (h * np.diag(f_prime_x_x(t[1:-1], x_full[1:], xp[1:])) + R + R.T +
            xp_jac.T @ np.diag(f_prime_xp_xp(t[:-1], x_full, xp)) @ xp_jac / h)
            
# Фиксирование числа точек
    
N = 50
ts = np.linspace(t_0,t_f,N+1)

J_target = partial(J, t = ts, x_0 = x_0, x_f = x_f, xp_jac = xp_approx_jac_h(N), f = f)
J_target_grad = partial(J_grad, t = ts, x_0 = x_0, x_f = x_f, xp_jac = xp_approx_jac_h(N), 
                        f_prime_x = f_prime_x, f_prime_xp = f_prime_xp)
J_target_hessian = partial(J_hessian, t = ts, x_0 = x_0, x_f = x_f, xp_jac = xp_approx_jac_h(N),
                          f_prime_x_x = f_prime_x_x, f_prime_x_xp = f_prime_x_xp, f_prime_xp_xp = f_prime_xp_xp)