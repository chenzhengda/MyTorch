#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim, char* device);
    void delete_tensor(Tensor* tensor);
    void delete_strides(Tensor* tensor);
    void delete_shape(Tensor* tensor);
    void delete_strides(Tensor* tensor);
    void delete_device(Tensor* tensor);
    float get_item(Tensor* tensor, int* indices);
}

#endif /* TENSOR_H */