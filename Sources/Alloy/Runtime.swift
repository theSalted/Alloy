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
    let (graph, finalTensor) = try GraphBuilder.buildGraph(from: root)
    
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
