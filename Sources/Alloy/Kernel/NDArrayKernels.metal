//
//  NDArrayKernels.metal
//  ScalarGrad
//
//  Created by Yuhao Chen on 11/10/24.
//

#include <metal_stdlib>
using namespace metal;

// Element-wise addition
kernel void elementwise_add(const device float* in1 [[ buffer(0) ]],
                            const device float* in2 [[ buffer(1) ]],
                            device float* out [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    out[id] = in1[id] + in2[id];
}

// Gradient for addition
kernel void elementwise_add_grad(const device float* grad_out [[ buffer(0) ]],
                                 device float* grad_in1 [[ buffer(1) ]],
                                 device float* grad_in2 [[ buffer(2) ]],
                                 uint id [[ thread_position_in_grid ]]) {
    // Accumulate gradients
    grad_in1[id] += grad_out[id];
    grad_in2[id] += grad_out[id];
}

// Element-wise multiplication
kernel void elementwise_multiply(const device float* in1 [[ buffer(0) ]],
                                 const device float* in2 [[ buffer(1) ]],
                                 device float* out [[ buffer(2) ]],
                                 uint id [[ thread_position_in_grid ]]) {
    out[id] = in1[id] * in2[id];
}

// Gradient for multiplication w.r.t. lhs
kernel void elementwise_mul_grad_lhs(const device float* rhs_data [[ buffer(0) ]],
                                     const device float* grad_out [[ buffer(1) ]],
                                     device float* grad_lhs [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    grad_lhs[id] += rhs_data[id] * grad_out[id];
}

// Gradient for multiplication w.r.t. rhs
kernel void elementwise_mul_grad_rhs(const device float* lhs_data [[ buffer(0) ]],
                                     const device float* grad_out [[ buffer(1) ]],
                                     device float* grad_rhs [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    grad_rhs[id] += lhs_data[id] * grad_out[id];
}
