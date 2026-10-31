from .engine import Tensor
import math
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=None
    def parameters(self):
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list,tuple)):
                for item in value:
                    if isinstance(item , Module):
                        yield  from item.parameters()
                    elif isinstance(item, Tensor):
                        yield item

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features=in_features
        self.out_features=out_features

        bound=math.sqrt(6.0/(in_features+out_features)) #Xavier (Glorot) Initialization
        weight_data=[random.uniform(-bound, bound) for _ in range(in_features*out_features)]
        self.weight=Tensor(weight_data,shape=(in_features,out_features))

        if bias:
            bias_data=[0.0 for _ in range (out_features)]
            self.bias=Tensor(bias_data, shape=(out_features,))

        else: # if using normalization do not need bias
            self.bias=None

    def forward(self, X):
        out=X@ self.weight
        if self.bias is not None:
            out=out+self.bias
        return out
class MLP(Module):

    def __init__(self, in_features,hidden_dims, out_features):
        dims=[in_features]+ hidden_dims + [out_features]
        self.layers=[]
        for i in range(len(dims)-1):
            self.layers.append(Linear(dims[i], dims[i+1]))

    def forward(self,X):
        for i, layer in enumerate(self.layers):
            X=layer(X)

            if i < len(self.layers)-1:
                X=X.relu()
        return X
    

class SGD:
    """ Stochastic  Gradient Descent optimizer."""
    def __init__(self,parameters, lr=0.01):
        self.parameters=list(parameters)
        self.lr=lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                for i in range(len(p.data)):
                    p.data[i]-=self.lr*p.grad.data[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad=None

