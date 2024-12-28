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
        throw NDArrayError.operationError("Fail to convert MPSGraphTensorData to MPSNDArray")
    }
    let shape = root.shape
    let elementCount = shape.reduce(1, *)
    var cpuBuffer = [Float](repeating: 0, count: elementCount)
    
    precondition(cpuBuffer.count == elementCount, "CPU buffer size mismatch.")
            
            // Read the bytes from `MPSNDArray` into `cpuBuffer`
    mpsndarray.readBytes(&cpuBuffer, strideBytes: nil)
    
    // Update the NDArray's data
    root.data = cpuBuffer
}
