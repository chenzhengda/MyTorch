from mytorch.tensor import Tensor

if __name__ == "__main__":
    data = [[0, 1, 2], [0, 2, 2], [0, 1, 2]]
    tensor = Tensor(data)
    print("--Test create tensor--")
    print("tensor:", tensor)
    print("--Test get tensor shape--")
    print("tensor shape:", tensor.shape)
    print("--Test get tensor item--")
    print("tensor[0, 0]:", tensor[0, 0])
    print("tensor[1, 1]:", tensor[1, 1])
    print("tensor[2, 2]:", tensor[2, 2])
    print("--Test add tensor--")
    tensor1 = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    tensor2 = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = tensor1 + tensor2
    print("result:", result)
