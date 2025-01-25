//
//  Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/29/24.
//

import MetalPerformanceShadersGraph

public func reshape(
    _ array: NDArray,
    to shape: [Int]) -> NDArray {
    let label = "reshaped to \(shape)"
    return NDArray(
        shape: shape,
        label: label,
        parents: [array]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("`reshaped` expects exactly 1 input.")
        }
        return graph.reshape(inputs[0], shape: shape.toNSNumberArray(), name: nodeLabel)
    }
}

public func broadcast(
    _ array: NDArray,
    to shape: [Int]
) -> NDArray {
    let label = "broadcasted to \(shape)"
    return NDArray(
        shape: shape,
        label: label,
        parents: [array]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("`broadcasted` expects exactly 1 input.")
        }
        return graph.broadcast(inputs[0], shape: shape.toNSNumberArray(), name: nodeLabel)
    }
}



/// Slices the NDArray from start indices to end indices (exclusive).
/// - Parameters:
///   - start: Starting indices for each dimension.
///   - end: Ending indices for each dimension (exclusive).
/// - Returns: A new NDArray representing the sliced tensor.
public func slice(_ x: NDArray, start: [Int], end: [Int], label: String? = nil) throws -> NDArray {
    guard start.count == x.shape.count, end.count == x.shape.count else {
        throw NDArrayError.operationError("Start and end indices must match the number of dimensions. Start count \(start.count); x.shape.count \(x.shape.count); end count \(end.count);")
    }
    
                
    // Assuming MPSGraph has a slicing operation. If not, implement it.
    // Here's a hypothetical implementation:
    return NDArray(
        shape: end.enumerated().map { $0.element - start[$0.offset] },
        label: label ?? "slice",
        parents: [x]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("Slice op expects exactly 1 input.")
        }
        let strides = Array(repeating: 1, count: x.shape.count).toNSNumberArray()
        let slicedTensor = graph.sliceTensor(inputs[0], starts: start.toNSNumberArray(), ends: end.toNSNumberArray(), strides: strides, name: nodeLabel)
        
        return slicedTensor
    }
}

/// Slices the NDArray by applying the same start and end indices to all dimensions.
/// - Parameters:
///   - x: The NDArray to be sliced.
///   - start: The starting index for each dimension.
///   - end: The ending index for each dimension (exclusive).
///   - label: An optional label for the operation.
/// - Returns: A new NDArray representing the sliced tensor.
public func slice(_ x: NDArray, start: Int, end: Int, label: String? = nil) throws -> NDArray {
    // Ensure that start and end are within the valid range for all dimensions
    for (dim, dimSize) in x.shape.enumerated() {
        guard start >= 0, end <= dimSize, start <= end else {
            throw NDArrayError.operationError("""
                Invalid slice indices for dimension \(dim):
                start: \(start), end: \(end), dimension size: \(dimSize)
                """)
        }
    }
    
    // Create start and end arrays by repeating the provided start and end for each dimension
    let startIndices = Array(repeating: start, count: x.shape.count)
    let endIndices = Array(repeating: end, count: x.shape.count)
    
    // Delegate to the existing slice function
    return try slice(x, start: startIndices, end: endIndices, label: label)
}
