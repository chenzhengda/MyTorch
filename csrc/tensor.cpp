#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "cpu.h"

extern "C" {

    Tensor* create_tensor(float* data, int* shape, int ndim, char* device) {
        
        Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
        if (tensor == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        tensor->data = data;
        tensor->shape = shape;
        tensor->ndim = ndim;

        tensor->device = (char*)malloc(strlen(device) + 1);
        if (device != NULL) {
            strcpy(tensor->device, device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        
        tensor->size = 1;
        for (int i = 0; i < ndim; i++) {
            tensor->size *= shape[i];
        }

        tensor->strides = (int*)malloc(ndim * sizeof(int));
        if (tensor->strides == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            tensor->strides[i] = stride;
            stride *= shape[i];
        }

        return tensor;
    }

    void delete_tensor(Tensor* tensor) {
        if (tensor != NULL) {
            free(tensor);
            tensor = NULL;
        }
    }

    void delete_shape(Tensor* tensor) {
        if (tensor->shape != NULL) {
            free(tensor->shape);
            tensor->shape = NULL;
        }
    }

    void delete_data(Tensor* tensor) {
        if (tensor->data != NULL) {
            if (strcmp(tensor->device, "cpu") == 0) {
                free(tensor->data);
            } else {
                // Unimplemented
            }
            tensor->data = NULL;
        }
    }

    void delete_strides(Tensor* tensor) {
        if (tensor->strides != NULL) {
            free(tensor->strides);
            tensor->strides = NULL;
        }
    }

    void delete_device(Tensor* tensor) {
        if (tensor->device != NULL) {
            free(tensor->device);
            tensor->device = NULL;
        }
    }

    float get_item(Tensor* tensor, int* indices) {
        int index = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            index += indices[i] * tensor->strides[i];
        }

        float result;
        if (strcmp(tensor->device, "cpu") == 0) {
            result = tensor->data[index];
        } else {
            // Unimplemented
        }

        return result;
    }

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }        
        
        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            add_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            // Unimplemented
        }     
    }

}