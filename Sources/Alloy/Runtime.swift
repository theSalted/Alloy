//
//  Runtime.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/28/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders

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
