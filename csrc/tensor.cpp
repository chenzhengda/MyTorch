#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

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

}