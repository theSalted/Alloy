//
//  Runtime.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/28/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders

/// Executes the DAG for multiple NDArray “roots” simultaneously.
/// - Parameters:
///   - roots: The NDArray “roots” we want to evaluate (or materialize data for).
///   - device: Optional MTLDevice. Defaults to system default.
///   - feeds: If you have placeholders among *any* of the roots/parents,
///            provide data via [MPSGraphTensor : MPSGraphTensorData].
/// - Throws: NDArrayError if graph construction or MPSGraph runtime fails.
///
/// After this call, each “root” NDArray’s `data` will contain the result.
public func run(
    _ roots: NDArray...,
    device: MTLDevice? = nil,
    feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
) throws {
    let (graph, nodeToTensor) = try GraphBuilder.buildGraph(from: roots)
    
    // Gather the final MPSGraphTensors for each requested root
    let finalTensors = roots.compactMap { nodeToTensor[$0] }
    
    let realDevice = device ?? MTLCreateSystemDefaultDevice()!
    guard let cmdQueue = realDevice.makeCommandQueue() else {
        throw NDArrayError.operationError("Failed to create a Metal command queue.")
    }

    // Execute in one pass. If you have multiple finalTensors,
    // pass them all as targets so the graph will compute them.
    let resultMap = graph.run(
        with: cmdQueue,
        feeds: feeds,
        targetTensors: finalTensors,
        targetOperations: nil
    )

    // For each requested root, retrieve data from resultMap and store in `NDArray.data`
    for root in roots {
        guard let t = nodeToTensor[root] else {
            throw NDArrayError.operationError("Missing final tensor for root \(root.label ?? "<?>").")
        }
        guard let mpsndarray = resultMap[t]?.mpsndarray() else {
            throw NDArrayError.operationError(
                "Failed to convert MPSGraphTensorData to MPSNDArray for root \(root.label ?? "<?>")."
            )
        }
        
        let shape = root.shape
        let elementCount = shape.reduce(1, *)
        let totalBytes = elementCount * MemoryLayout<Float>.size
        var bufferData = Data(count: totalBytes)
        
        // Read bytes directly into `bufferData`
        bufferData.withUnsafeMutableBytes { rawBufferPtr in
            guard let ptr = rawBufferPtr.baseAddress else { return }
            mpsndarray.readBytes(ptr, strideBytes: nil)
        }
        
        // Write the updated data back to the root NDArray
        root.data = bufferData
    }
}


/// Executes the DAG by building an MPSGraph and running it.
/// - Parameters:
///   - root: The root NDArray. All dependencies (parents, etc.) are captured in the DAG.
///   - device: If unspecified, defaults to the system default device.
///   - feeds: If you have placeholders, provide data via `[MPSGraphTensor : MPSGraphTensorData]`.
///   - targetOperations: Optional. If you have extra ops to schedule, pass them here.
/// - Returns: The final `MPSNDArray` corresponding to `root`.
/// - Throws: `NDArrayError` if graph construction fails, or MPSGraph runtime errors.
public func run(
    _ root: NDArray,
    device: MTLDevice? = nil,
    feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
) throws {
    let (graph, nodeMap) = try GraphBuilder.buildGraph(from: root)
    
    // The final node’s MPSGraphTensor
    guard let finalTensor = nodeMap[root] else {
        throw NDArrayError.operationError("No final tensor for root NDArray.")
    }
    
    let realDevice = device ?? MTLCreateSystemDefaultDevice()!
    guard let cmdQueue = realDevice.makeCommandQueue() else {
        throw NDArrayError.operationError("Failed to create a Metal command queue.")
    }
    
    let resultMap = graph.run(
        with: cmdQueue,
        feeds: feeds,
        targetTensors: [finalTensor],
        targetOperations: nil
    )
    
    guard let mpsndarray = resultMap[finalTensor]?.mpsndarray() else {
        throw NDArrayError.operationError("Failed to convert MPSGraphTensorData to MPSNDArray")
    }
    
    // Prepare a Data buffer of the correct size
    let shape = root.shape
    let elementCount = shape.reduce(1, *)
    let totalBytes = elementCount * MemoryLayout<Float>.size
    var bufferData = Data(count: totalBytes)
    
    // Read bytes directly into `bufferData`
    bufferData.withUnsafeMutableBytes { rawBufferPtr in
        guard let ptr = rawBufferPtr.baseAddress else { return }
        mpsndarray.readBytes(ptr, strideBytes: nil)
    }
    
    // Store the data in the NDArray
    root.data = bufferData
}


/// Compute gradients of `loss` w.r.t. `parameters`.
///
/// - Parameters:
///   - loss: The final NDArray representing your scalar loss (or it could be a mean over batch).
///   - parameters: An array of NDArrays (e.g. your trainable weights, biases).
///   - device: Optional MTLDevice. Defaults to system default.
///   - feeds: A dictionary for placeholder data if needed, just like forward pass.
/// - Throws: NDArrayError if something goes wrong.
/// - Returns: A dictionary mapping `[NDArray: Data]` = the gradients for each parameter.
public func backward(
    loss: NDArray,
    parameters: [NDArray],
    device: MTLDevice? = nil,
    feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
) throws -> [NDArray: Data] {
    print("Backward")
    // 1) Build the forward graph
    
    let (graph, nodeMap) = try GraphBuilder.buildGraph(from: loss)
    
    // 2) Retrieve MPSGraphTensor for the `loss` node
    guard let lossTensor = nodeMap[loss] else {
        throw NDArrayError.operationError("No final tensor for loss NDArray.")
    }
    
    // 3) Get MPSGraphTensors for all parameters
    let parameterTensors: [MPSGraphTensor] = try parameters.map {
        guard let t = nodeMap[$0] else {
            throw NDArrayError.operationError("Parameter \($0.label ?? "<?>") not found in graph.")
        }
        return t
    }
    
    // 4) Ask MPSGraph for the gradient Tensors
    // NOTE: Apple calls the method `gradients(of:with: name:)` or
    //  `gradients(of:withRespectToTensors: name:)` in some versions.
    // Adjust for your Xcode/SDK version:
    
    let gradientsMap: [MPSGraphTensor : MPSGraphTensor] =
        graph.gradients(of: lossTensor,
                        with: parameterTensors,
                        name: "gradients")
    
    // We want to run the graph to compute:
    //   - The original loss itself (optional if you want it for display)
    //   - Each gradient
    var targets = [lossTensor]
    targets.append(contentsOf: gradientsMap.values)
    
    // 5) Create command queue
    guard let cmdQueue = Alloy.shared.device.makeCommandQueue() else {
        throw NDArrayError.operationError("Failed to create a Metal command queue.")
    }
    
    print("run graph")
    
    // 6) Run the graph
    let resultMap = graph.run(
        with: cmdQueue,
        feeds: feeds,
        targetTensors: targets,
        targetOperations: nil
    )
    
    // 7) Read back the gradients from GPU to CPU
    //    We'll also read back the final loss if you want:
//    if let finalLossData = resultMap[lossTensor]?.mpsndarray() {
//        // Read final loss as float if needed...
//    }
    
    // Gather the resulting gradient data into a dictionary:
    var paramToGradData = [NDArray: Data]()
    
    for param in parameters {
        guard let paramTensor = nodeMap[param] else {
            // Should not happen because we constructed paramTensors earlier
            continue
        }
        guard let gradTensor = gradientsMap[paramTensor] else {
            // If the param didn't actually affect the loss, the gradient might be nil
            // (MPSGraph sometimes omits zero grad nodes).
            // We'll store an all-zero buffer if that’s your preference
            let zeroData = Data(count: param.shape.reduce(1, *) * MemoryLayout<Float>.size)
            paramToGradData[param] = zeroData
            continue
        }
        
        // 8) Convert MPSGraphTensorData → MPSNDArray → CPU Data
        guard let gradMPSNDArray = resultMap[gradTensor]?.mpsndarray() else {
            // Param doesn't affect the loss or something else failed
            let zeroData = Data(count: param.shape.reduce(1, *) * MemoryLayout<Float>.size)
            paramToGradData[param] = zeroData
            continue
        }
        
        let shape = param.shape
        let elementCount = shape.reduce(1, *)
        let totalBytes = elementCount * MemoryLayout<Float>.size
        var bufferData = Data(count: totalBytes)
        
        bufferData.withUnsafeMutableBytes { rawBufferPtr in
            guard let ptr = rawBufferPtr.baseAddress else { return }
            gradMPSNDArray.readBytes(ptr, strideBytes: nil)
        }
        
        paramToGradData[param] = bufferData
    }
    
    return paramToGradData
}
