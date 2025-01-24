//
//  SGD.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/15/25.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders
import Metal
import Foundation

/// Performs a single step of SGD for all `params` that affect `loss`.
///
/// - Parameters:
///   - loss: The final NDArray representing your scalar loss node.
///   - params: A mutable array of trainable NDArray parameters (weights, biases).
///   - learningRate: The scalar learning rate for SGD.
///   - device: Optional MTLDevice. Defaults to system default if `nil`.
///   - feeds: Optional placeholders if your graph has inputs (e.g., data, labels).
///
/// - Throws: NDArrayError if something goes wrong (graph build, backward pass, etc.)
public func SGD(
    loss: NDArray,
    params: inout [NDArray],
    learningRate: Float,
    device: MTLDevice? = nil,
    feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
) throws {
    // 1) Compute gradients w.r.t. each parameter
    let gradients = try backward(
        loss: loss,
        parameters: params,
        device: device,
        feeds: feeds
    )
    
    // 2) For each parameter, do param = param - lr * grad
    for param in params {
        // Retrieve the gradient Data for this parameter
        guard let gradData = gradients[param] else {
            // No gradient found (maybe it doesn't affect loss)
            // Typically you'd do nothing or treat it as zero grad
            continue
        }
        
        // Retrieve param CPU data
        guard let paramData = param.data else {
            // If no CPU data, might be a placeholder param
            // Possibly skip or throw
            continue
        }
        
        // Convert both to [Float]
        let paramFloats: [Float] = paramData.toFloatArray() ?? []
        let gradFloats: [Float]  = gradData.toFloatArray()  ?? []
        
        precondition(paramFloats.count == gradFloats.count,
                     "Parameter and gradient have mismatched sizes.")
        
        // 3) Update step on CPU
        //    param[i] -= learningRate * grad[i]
        var updatedFloats = [Float](repeating: 0, count: paramFloats.count)
        for i in 0..<paramFloats.count {
            updatedFloats[i] = paramFloats[i] - learningRate * gradFloats[i]
        }
        
        // 4) Write updated data back into param NDArray
        param.data = updatedFloats.toData(shape: param.shape)  // convert [Float] → Data
    }
}

/// Performs a single SGD update step using MPSGraph's built-in optimizer.
/// This function computes gradients and updates parameters directly on the GPU.
///
/// - Parameters:
///   - loss: The NDArray representing the final loss in the computation graph.
///   - params: An array of trainable NDArray parameters (weights, biases).
///   - learningRate: The learning rate for SGD.
///   - device: Optional MTLDevice. Defaults to the system's default device.
///   - feeds: Optional dictionary for feeding placeholder data into the graph.
///
public func SGD(
    loss: NDArray,
    params: [NDArray],
    learningRate: Float,
    device: MTLDevice? = nil,
    feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
) throws {
    // 1. Compute gradients using the backward function
    let gradients = try backward(
        loss: loss,
        parameters: params,
        device: device,
        feeds: feeds
    )
    
    // 2. Build the computation graph from the loss NDArray
    let (graph, nodeMap) = try GraphBuilder.buildGraph(from: loss)
    
    // 3. Retrieve the MPSGraphTensor corresponding to the loss
    
    
    // 4. Create a variable tensor for the learning rate
    let lrData = withUnsafeBytes(of: learningRate) { Data($0) }
    let learningRateTensor = graph.variable(
        with: lrData,
        shape: [],
        dataType: .float32,
        name: "lr tensor"
    )
    
    // 5. Iterate over each parameter and apply SGD update
    for param in params {
        
        // a. Retrieve the corresponding MPSGraphVariableOp for the parameter
        guard let values = nodeMap[param] else {
            continue
        }
        
        
        // b. Retrieve the gradient data for this parameter
        guard let gradData = gradients[param] else {
            // If no gradient is found, skip updating this parameter
            // Optionally, you can initialize gradients to zero here
            continue
        }
        
        // c. Create a variable tensor for the gradient
        // Convert `Data` to `MPSGraphTensor` by creating a variable
        let gradTensor = graph.variable(
            with: gradData,
            shape: param.shape.map { NSNumber(value: $0) },
            dataType: .float32,
            name: param.label
        )
        
        graph.stochasticGradientDescent(learningRate: learningRateTensor, values: values, gradient: gradTensor, name: "sgd_update_\(param.label ?? "param")")
    }
    
    // 6. Execute the graph to perform updates on the GPU
    let realDevice = device ?? MTLCreateSystemDefaultDevice()!
    guard let commandQueue = realDevice.makeCommandQueue() else {
        throw NDArrayError.operationError("Failed to create a Metal command queue.")
    }
    
    // 7. Run the graph without needing to fetch any output tensors
    graph.run(
        with: commandQueue,
        feeds: feeds,
        targetTensors: [],       // No need to fetch outputs
        targetOperations: nil    // No specific operations to target
    )
}
