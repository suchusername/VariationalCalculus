import numpy as np
from problem import *

# Useful functions

def sigmoid(x):
    """works with both numbers and vectors"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """works with both numbers and vectors"""
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, sizes=None, act_func=None, fix_first_layer=False, load_from=None, output=False):
        """
        'sizes' - list of layers. (Ex: [2,3,1] - 2 inputs, a hidden layer with 3 neurons, 1 output)
        'act_func' - a tuple of 2 lists (activation funcs, derivatives) for each layer (if None, every neuron has a sigmoid)
        'fix_first_layer' - fixes layer between first two layers as 1 and biases as 0
        'load_from' - .npy file that is a list [sizes, biases, weights]
        'output' - if True, prints out some progress of a neural network
        
        
        Список ``sizes`` содержит количество нейронов в соответствующих слоях
        нейронной сети. К примеру, если бы этот лист выглядел как [2, 3, 1],
        то мы бы получили трёхслойную нейросеть, с двумя нейронами в первом
        (входном), тремя нейронами во втором (промежуточном) и одним нейроном
        в третьем (выходном, внешнем) слое. Смещения и веса для нейронных сетей
        инициализируются случайными значениями, подчиняющимися стандартному нормальному
        распределению. Обратите внимание, что первый слой подразумевается слоем, 
        принимающим входные данные, поэтому мы не будем добавлять к нему смещение 
        (делать это не принято, поскольку смещения используются только при 
        вычислении выходных значений нейронов последующих слоёв)
        """
        
        if load_from is None and sizes is not None:
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.normal(size=(y, 1)) for y in sizes[1:]]
            # randn - samples from standard normal distrubution, returns ndarray of shape (y,1)
            # biases[0] - biases of the first hidden layer (l=2)

            self.weights = [np.random.normal(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
            if fix_first_layer:
                self.weights[0] = np.ones((sizes[1], sizes[0]))
                self.biases[0] = np.zeros((sizes[1],1))
            # zip has a list (length = #of layers - 1) of tuples (# of neurons on layer l-1 and layer l)
            # returns list of weight matrices with random weights
            # weights[0] - weights between input layer (l=1) and first hidden layer (l=2)
        elif load_from is not None:
            loaded = np.load(load_from)
            self.num_layers = len(loaded[0])
            self.sizes = loaded[0]
            self.biases = loaded[1]
            self.weights = loaded[2]
        else:
            return None
                
        if (act_func is not None) and (len(self.sizes) - len(act_func) != 1):
            print("Number of elements in act_func list does not match the number of layers.")
            return None
        
        # setting activation functions and their derivatives (both are None if sigmoids are used)
        self.set_activation_functions(act_func)
        
        self.fix_first_layer = fix_first_layer
        
        # whether we want to print debugging information
        self.output = output
        
    def set_activation_functions(self, act_func):
        """
        Sets activation functions of the network and their derivatives layer by layer.
        act_func[0] - a list with activation functions
        act_func[1] - a list with their derivatives
        
        act_func[0][0] - a list that contains activations functions of first hidden layer.
            if act_func[0][0] is 1 element and not a list, then each neuron in the layer has the same activation function.
            if act_func[0][0] is None, then the whole layer has sigmoids.
            
        Same applies to derivatives. 
            
        If act_func is None, then the whole network has sigmoids (default).
        """
        
        if act_func is None:
            self.act_func, self.act_func_prime = (None, None)
            return
                    
        ret = []
        ret_prime = []
        
        if (len(act_func[0]) != len(act_func[1])):
            print("Length of lists of activation functions and their derivatives do not match.")
            self.act_func, self.act_func_prime = (None, None)
            return
        
        for i in range(len(act_func[0])):
            
            if act_func[0][i] is None:
                ret.append([sigmoid] * self.sizes[i+1])
            elif isinstance(act_func[0][i], list):
                if len(act_func[0][i]) != self.sizes[i+1]:
                    print("Lengths of activation function list for layer", i+2, "doesn't match the size of a layer.")
                    self.act_func, self.act_func_prime = (None, None)
                    return
                ret.append(act_func[0][i])
            else:
                ret.append([act_func[0][i]] * self.sizes[i+1])
                
            if act_func[1][i] is None:
                ret_prime.append([sigmoid_prime] * self.sizes[i+1]) 
            elif isinstance(act_func[1][i], list):
                if len(act_func[1][i]) != self.sizes[i+1]:
                    print("Lengths of activation function derivatives list for layer", i+2, "doesn't match the size of a layer.")
                    self.act_func, self.act_func_prime = (None, None)
                    return
                ret_prime.append(act_func[1][i])
            else:
                ret_prime.append([act_func[1][i]] * self.sizes[i+1])
                
        self.act_func, self.act_func_prime = (ret, ret_prime)
        return
        
    def feedforward(self, x):
        """
        Calculate network's answer when input is 'x'.
        """
        if self.act_func is None:
            a = np.copy(x)
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a)+b)
            return a
        
        a = np.copy(x)
        l = 0
        for b, w in zip(self.biases, self.weights):
            a2 = np.dot(w, a)+b
            a = np.zeros_like(b)
            for j in range(a2.shape[0]):
                a[j] = self.act_func[l][j](a2[j])
            l += 1
        return a
    
    def backprop(self, x, cost_func_grad):
        """
        x - input of the network
        
        Returns a tuple (y, nabla_b, nabla_w) - value at x + gradient of cost function by all network's parameters
        y - output of the network on input x
        nabla_b - list of nd.array's of same shape as self.biases
        nabla_w - list of nd.array's of same shape as self.weights
        """
        """
        Возвращает кортеж ``(nabla_b, nabla_w)`` -- градиент целевой функции по всем параметрам сети.
        ``nabla_b`` и ``nabla_w`` -- послойные списки массивов ndarray,
        такие же, как self.biases и self.weights соответственно.
        """
        
        if self.act_func is None:
            # assuming x is a vertical vector
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            a = [x]
            #list 'a' will contain all activations

            # filling the array 'a' with network's activations
            for b, w in zip(self.biases, self.weights):
                a.append(sigmoid(w.dot(a[-1]) + b))

            # calculating the derivatives of cost function w.r.t. biases of last layer
            delta = cost_func_grad(a[-1]) * a[-1] * (1 - a[-1])

            # error of the last layer
            nabla_b[-1] = delta
            # производная J по смещениям выходного слоя
            nabla_w[-1] = delta.dot(a[-2].T)
            # производная J по весам выходного слоя

            for l in range(2, self.num_layers):
                delta = (self.weights[1-l].T.dot(delta)) * a[-l] * (1-a[-l])
                # ошибка на слое L-l
                nabla_b[-l] = delta
                # производная J по смещениям L-l-го слоя
                nabla_w[-l] = delta.dot(a[-l-1].T)
                # производная J по весам L-l-го слоя
                
            return a[-1], nabla_b, nabla_w
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = [x]
        z = []
        
        l = 0
        for b, w in zip(self.biases, self.weights):
            #print(l, b.shape, w.shape, a[-1].shape)
            z.append(w.dot(a[-1]) + b)
            a1 = np.zeros_like(b)
            for j in range(a1.shape[0]):
                a1[j] = self.act_func[l][j](z[-1][j])
            a.append(a1)
            l += 1
            
        a_grad = np.zeros_like(a[-1])
        for j in range(a_grad.shape[0]):
            a_grad[j] = self.act_func_prime[-1][j](z[-1][j])
        delta = cost_func_grad(a[-1].flatten()) * a_grad
        
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(a[-2].T)
        
        for l in range(2, self.num_layers):
            a_grad = np.zeros_like(a[-l])
            for j in range(a_grad.shape[0]):
                a_grad[j] = self.act_func_prime[-l][j](z[-l][j])
            delta = (self.weights[1-l].T.dot(delta)) * a_grad
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(a[-l-1].T)
        
        return a[-1], nabla_b, nabla_w
        
        
        
    
    def update_params(self, ts, alpha, lambda_1, lambda_2, J_grad_x):
        """
        Update weights and biases after one epoch of stochastic gradient descent.
        
        This version of update function is specifically modified for a calculus of variations problem.
        
        'ts' - selected training data (list of inputs [t_1, ..., t_N-1])
        'alpha' - learning rate
        'lambda_1' - L1-regularization constant
        'lambda_2' - L2-regularization constant
        'J_grad_x' - gradient of loss function with respect to network outputs
        
        Returns the network outputs before this update
        """
        
        xs = []
        nabla_bs = []
        nabla_ws = []
        for t in ts:
            x, nabla_b, nabla_w = self.backprop(np.array(t).reshape(1,1), lambda x: 1)
            xs.append(x[0][0])
            nabla_bs.append(nabla_b)
            nabla_ws.append(nabla_w)
        
        J_grad = J_grad_x(np.array(xs))
        
        # transposing lists so that they are first sorted by layer and then by points
        nabla_bs = list(map(list, zip(*nabla_bs)))
        nabla_ws = list(map(list, zip(*nabla_ws)))
        
        # creating lists of partial derivatives
        nabla_b = []
        nabla_w = []
        #print(self.biases)
        #print(self.weights)
        for i in range(len(nabla_bs)):
            nb = (np.array(nabla_bs[i]).squeeze().T @ J_grad).reshape((-1,1))
            nw = np.array(nabla_ws[i]).transpose(1,2,0) @ J_grad
            # adding regularization
            nb = alpha * nb + lambda_2 * self.biases[i] + lambda_1 * self.biases[i]
            nw = alpha * nw + lambda_2 * self.weights[i] + lambda_1 * self.weights[i]
            nabla_b.append(nb)
            nabla_w.append(nw)
            
       
        
        if self.fix_first_layer:
            self.biases  = [self.biases[0]] + [b - nb for b, nb in zip(self.biases[1:],  nabla_b[1:])]
            self.weights = [self.weights[0]] + [w - nw for w, nw in zip(self.weights[1:], nabla_w[1:])]
        else:
            self.biases  = [b - nb for b, nb in zip(self.biases,  nabla_b)]
            self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]

        return xs
    
    def gradient_descent(self, ts, alpha, lambda_1, lambda_2, J_grad_x, J_x, n_iter, retall=False):
        # Stochastic gradient descent
        # eta === learning rate
        
        """
        'ts' - selected training data, (list of inputs [t_1, ..., t_N-1])
        'alpha' - learning rate
        'lambda_1' - L1-regularization constant
        'lambda_2' - L2-regularization constant
        'J_grad_x' - gradient of loss function with respect to network outputs
        'J_x' - loss function with respect to network outputs
        'n_iter' - number of iterations (epochs)
        """
        
        iterations = []
        
        for j in range(n_iter):
            xs = self.update_params(ts, alpha, lambda_1, lambda_2, J_grad_x)
            iterations.append((xs, J_x(xs)))
            
            if self.output:
                print("Iteration {0}: J = {1}".format(j, iterations[-1][1]))
            
        best = np.array([self.feedforward(np.array(t).reshape(1,1)) for t in ts]).squeeze()
        iterations.append((best, J_x(best)))
        if self.output:
                print("Iteration {0}: J = {1}".format(n_iter, iterations[-1][1]))
                
        if not retall:
            return best
        return (best, iterations)       
    
    
    def save_to(file_path):
        NN_params = [self.sizes, self.biases, self.weights]
        np.save(file_path, NN_params)
        return
    
    