class Tensor:
    def __init__(self, data, shape=None, _children=()):
        if shape is None:
            self.data,self.shape=self._flatten_list_calulate_shape(data)
        else:
            self.shape=shape
            self.data=data
        self.strides=self.calculate_strides(self.shape)
        self._prev=set(_children)
        self._backward= lambda: None
        self.grad=None

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def _flatten_list_calulate_shape(self, data):
       if not isinstance(data, (list,tuple) ):
           return [data], ()
           
       if len(data)==0:
           return [], (0,)
           
       flatten_list=[]
       shape=[]
       for item in data:
           flatten_data,shape_inter=self._flatten_list_calulate_shape(item)
           flatten_list.extend(flatten_data)
           shape.append(shape_inter)
           
       assert all(s==shape[0] for s in shape)
       current_shape=(len(data),) + shape[0]
       return flatten_list,current_shape

    def calculate_strides(self,shape):
       strides=[]
       stride=1
       for dimension in reversed(shape):
           strides.insert(0, stride)
           stride*=dimension
       return strides

    def __getitem__(self,idx):
        if not isinstance(idx, tuple):
            idx=(idx,)
        assert len(idx)==len(self.shape)
        flat_index= sum(i*s for i, s in zip(idx,self.strides))
        return self.data[flat_index]
        
    def _ndindex(self, shape):
        """Generates all multi-dimensional index combinations for a given shape."""
        if not shape:
            yield ()
            return
            
        # Recursive generator for ndindex
        head = shape[0]
        tail = shape[1:]
        for i in range(head):
            for rest in self._ndindex(tail):
                yield (i,) + rest

    def axes_to_sum(self, original_shape):
        if not isinstance (original_shape, tuple):
            original_shape=tuple(original_shape)
        number_padded=len(self.shape)-len(original_shape)
        padd=(1,)*number_padded + original_shape
        sum_axis=[]
        for idx, (c,s) in enumerate(zip(self.shape, padd)):
            if idx < number_padded or c != s:
              sum_axis.append(idx)  
        return sum_axis,number_padded

    def unbroadcast(self, original_shape):
        sum_axis, number_padded=self.axes_to_sum(original_shape)
        padd=(1,)*number_padded + tuple(original_shape)
        out_strides=self.calculate_strides(padd)

        size=1
        for d in padd:
            size*=d
        out_data=[0.0]*size

        for coord in self._ndindex(self.shape):
            out_coord=tuple(0 if idx in sum_axis else coordinate for idx,coordinate in enumerate(coord))
            slot=sum(c*s for c,s in zip(out_coord, out_strides))
            out_data[slot] += self[coord]
        return Tensor(out_data, tuple(original_shape))
        
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        shape_a = self.shape
        shape_b = other.shape
        
        max_dims = max(len(shape_a), len(shape_b))
        padded_a = (1,) * (max_dims - len(shape_a)) + shape_a
        padded_b = (1,) * (max_dims - len(shape_b)) + shape_b
        
        result_shape = []
        for dim_a, dim_b in zip(padded_a, padded_b):
            if dim_a == dim_b:
                result_shape.append(dim_a)
            elif dim_a == 1:
                result_shape.append(dim_b)
            elif dim_b == 1:
                result_shape.append(dim_a)
            else:
                raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")
                
        result_shape = tuple(result_shape)
        
        
        result_data = []
        for coords in self._ndindex(result_shape):
            trailing_coords_a = coords[-len(shape_a):] if len(shape_a) > 0 else ()
            coord_a = tuple(c % s for c, s in zip(trailing_coords_a, shape_a)) 
            val_a = self[coord_a]
            
            trailing_coords_b = coords[-len(shape_b):] if len(shape_b) > 0 else () #  If shape_a is a 0 D scalar (shape_a = ()), len(shape_a) is 0.
                                                                                    #In Python, coords[-0:] is identical to coords[0:], which returns the entire tuple rather than an empty tuple (),so to prevent it we add this contition of trailing_coords_b
            coord_b = tuple(c % s for c, s in zip(trailing_coords_b, shape_b))
            val_b = other[coord_b]
           
            result_data.append(val_a + val_b)
        out=Tensor(result_data, result_shape,_children=(self,other))

        def _backward():
            self._accumulate_grads(out.grad.unbroadcast(self.shape))
            other._accumulate_grads(out.grad.unbroadcast(other.shape))

        out._backward= _backward
        return out

    def __mul__(self,other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        maximum_shape= max(len(self.shape), len(other.shape))
        padd_a= (1,)* (maximum_shape-len(self.shape))+ self.shape
        padd_b= (1,)* (maximum_shape-len(other.shape))+ other.shape

        result_shape=[]
        for dim_a, dim_b in zip(padd_a,padd_b):
            if dim_a==dim_b:
                result_shape.append(dim_a)
            elif dim_a==1:
                result_shape.append(dim_b)
            elif dim_b==1:
                result_shape.append(dim_a)
            else:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")

        result_shape=tuple(result_shape)

        result_data=[]

        for coords in self._ndindex(result_shape):
            trailing_coords= coords[-len(self.shape):] if len(self.shape) > 0 else ()
            coord_a=tuple(c%s for c,s in zip(trailing_coords,self.shape))
            value_a= self[coord_a]

            trailing_coords=coords[-len(other.shape):] if len(other.shape) >0 else ()
            coord_b=tuple(c%s for c ,s in zip(trailing_coords,other.shape))
            value_b=other[coord_b]

            result_data.append(value_a * value_b)

        out=Tensor(result_data,result_shape,_children=(self, other))

        def _backward():
           self._accumulate_grads((out.grad*other).unbroadcast(self.shape))
           other._accumulate_grads((out.grad*self).unbroadcast(other.shape))
        
        out._backward=_backward
        return out

    
    def _accumulate_grads(self,grad):
        if self.grad is None:
            self.grad=Tensor(list(grad.data),self.shape)
        else:
            for i in range(len(self.grad.data)):
                self.grad.data[i]+=grad.data[i]
    
    def backwards(self):
        topo=[]
        visited=set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)
        build_topo(self)
        if self.grad is None:
            self.grad=Tensor([1.0]*len(self.data),self.shape)
        for v in reversed(topo):
            v._backward()

    def sum(self):
        total=sum(self.data)
        out=Tensor([total], shape=(), _children=(self,))

        def _backward():
            grad_val = out.grad.data[0]
            self._accumulate_grads(Tensor([grad_val]*len(self.data),self.shape))

        out._backward=_backward
        return out

    def mean(self):
        n=len(self.data)
        avg=sum(self.data)/n
        out=Tensor([avg], shape=(), _children=(self,))

        def _backward():
            grad_val=out.grad.data[0] /n
            self._accumulate_grads(Tensor([grad_val]* len(self.data) ,self.shape))

        out._backward=_backward
        return  out

   

    def transpose(self, axes=None):
        ndim = len(self.shape)
        if ndim <= 1:
            return self

        # Default: reverse all axes (like NumPy)
        if axes is None:
            axes = tuple(reversed(range(ndim)))
        else:
            axes = tuple(axes)
            assert len(axes) == ndim and sorted(axes) == list(range(ndim)), (
                f"Invalid axes {axes} for tensor of dimension {ndim}"
            )

        new_shape = tuple(self.shape[ax] for ax in axes)

        # Compute inverse permutation for autograd backward
        inverse_axes= [0]*ndim
        for k,ax in enumerate(axes):
            inverse_axes[ax]=k
        inverse_axes=tuple(inverse_axes)

        new_data = []
        for new_coord in self._ndindex(new_shape):
            original_shape=[0]*ndim
            for k,idx in enumerate(axes):
                original_shape[idx]=new_coord[k]
            new_data.append(self[tuple(original_shape)])

        out = Tensor(new_data, new_shape, _children=(self,))

        def _backward():
            self._accumulate_grads(out.grad.transpose(inverse_axes))

        out._backward = _backward
        return out

    @property
    def T(self):
      return self.transpose()   

    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            other= Tensor(other)
        
        assert len(self.shape)==2 and len(other.shape)==2, "2D matrices required"
        assert self.shape[1]==other.shape[0], f" Matrices {self.shape} and {other.shape} are not compatible"
        
        m=self.shape[0]
        n=other.shape[1]
        k1=other.shape[0]

        result_data=[]
        for i in range(m):
            for j in range(n):
                cij=0.0
                for k in range (k1):
                    cij+=self[i,k]*other[k,j]
                result_data.append(cij)

        out = Tensor(result_data, shape=(m,n), _children=(self,other))
        def _backward():
            self._accumulate_grads(out.grad @ other.T)
            other._accumulate_grads(self.T @ out.grad)
        out._backward=_backward
        return out

    def relu(self, _NegativeSlope=0.01):
        out_data=[val if val>0 else _NegativeSlope*val for val in self.data]
        out=Tensor(out_data, shape=self.shape, _children=(self,))

        def _backward():
            grad_data=[ g*1 if val>0 else g*_NegativeSlope for val, g in zip(self.data, out.grad.data)]
            self._accumulate_grads(Tensor(grad_data,shape=self.shape))
        out._backward=_backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self,other):
        return self + (-other)

    def __radd__(self,other):
        return self+other

    def __rmul__(self,other):
        return self*other
