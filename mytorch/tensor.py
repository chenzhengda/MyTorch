import ctypes
import os

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
        ('device', ctypes.c_char_p)
    ]

class Tensor:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    _C = ctypes.CDLL(os.path.join(module_dir, "libMyTorch.so"))
    
    def __init__(self, data=None, device="cpu", requires_grad=False):

        if data != None:
            if isinstance(data, (float, int)):
                data = [data]

            data, shape = self.flatten(data)
            
            self.shape = shape.copy()
            
            self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
            self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
            self._ndim_ctype = ctypes.c_int(len(shape))
            self._device_ctype = device.encode('utf-8')

            self.ndim = len(shape)
            self.device = device

            self.numel = 1
            for s in self.shape:
                self.numel *= s

            self.requires_grad = requires_grad
            self.hooks = []
            self.grad = None
            self.grad_fn = None

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
            Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)
            
            self.tensor = Tensor._C.create_tensor(
                self._data_ctype,
                self._shape_ctype,
                self._ndim_ctype,
                self._device_ctype
            )
        
        else:
            self.tensor = None,
            self.shape = None,
            self.ndim = None,
            self.device = device
            self.requires_grad = None
            self.hooks = []
            self.grad = None
            self.grad_fn = None
            
    def flatten(self, nested_list):
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape
        
        flat_data, shape = flatten_recursively(nested_list)
        return flat_data, shape

    def __del__(self):
            
        if hasattr(self, '_data_ctype') and self._data_ctype is not None:
        
            Tensor._C.delete_strides.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_strides.restype = None
            Tensor._C.delete_strides(self.tensor)

            Tensor._C.delete_device.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_device.restype = None
            Tensor._C.delete_device(self.tensor)

            Tensor._C.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_tensor.restype = None
            Tensor._C.delete_tensor(self.tensor)

        elif self.tensor is not None:
            Tensor._C.delete_strides.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_strides.restype = None
            Tensor._C.delete_strides(self.tensor)

            Tensor._C.delete_data.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_data.restype = None
            Tensor._C.delete_data(self.tensor)

            Tensor._C.delete_shape.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_shape.restype = None
            Tensor._C.delete_shape(self.tensor)

            Tensor._C.delete_device.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_device.restype = None
            Tensor._C.delete_device(self.tensor)

            Tensor._C.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_tensor.restype = None
            Tensor._C.delete_tensor(self.tensor)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        if len(indices) != self.ndim:
            raise ValueError("Number of indices must match the number of dimensions")
        
        Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float
                                           
        indices = (ctypes.c_int * len(indices))(*indices)
        value = Tensor._C.get_item(self.tensor, indices)  

        return value 

    def __str__(self):
        def print_recursively(tensor, depth, index):
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[-1]):
                    index[-1] = i
                    result += str(tensor[tuple(index)]) + ", "
                return result.strip()
            else:
                result = ""
                if depth > 0:
                    result += "\n" + " " * ((depth - 1) * 4)
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    result += "["
                    result += print_recursively(tensor, depth + 1, index) + "],"
                    if i < tensor.shape[depth] - 1:
                        result += "\n" + " " * (depth * 4)
                return result.strip(",")

        index = [0] * self.ndim
        result = "tensor(["
        result += print_recursively(self, 0, index)
        result += f"""], device="{self.device}", requires_grad={self.requires_grad})"""
        return result

    def __repr__(self):
        return self.__str__()


    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = other * self.ones_like()

        broadcasted_shape_add = []

        # Function to determine if broadcasting is needed and get the broadcasted shape
        def broadcast_shape(shape1, shape2):
            if shape1 == shape2:
                return shape1, False
            
            max_len = max(len(shape1), len(shape2))
            shape1 = [1] * (max_len - len(shape1)) + shape1
            shape2 = [1] * (max_len - len(shape2)) + shape2

            
            for dim1, dim2 in zip(shape1, shape2):
                if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                    raise ValueError("Shapes are not compatible for broadcasting")
                broadcasted_shape_add.append(max(dim1, dim2))
            return broadcasted_shape_add, True

        broadcasted_shape_add, needs_broadcasting = broadcast_shape(self.shape, other.shape)

        if needs_broadcasting:
            # Call add_broadcasted_tensor if broadcasting is needed
            if other.ndim == self.ndim - 1:
                other = other.reshape([1] + other.shape)
            
            elif self.ndim == other.ndim - 1:
                self = self.reshape([1] + self.shape)
                
            
                
            Tensor._C.add_broadcasted_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.add_broadcasted_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.add_broadcasted_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = broadcasted_shape_add.copy()
            result_data.ndim = len(broadcasted_shape_add)

            result_data.device = self.device
            result_data.numel = 1
            for s in result_data.shape:
                result_data.numel *= s

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                raise NotImplementedError("AddBroadcastedBackward is not implemented")
        
        else:
            # Call add_tensor if shapes are identical
            Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim

            result_data.device = self.device
            result_data.numel = self.numel  # Update this to calculate the correct number of elements if broadcasting

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                raise NotImplementedError("AddBackward is not implemented")

        return result_data
