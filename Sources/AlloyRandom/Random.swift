//
//  Random.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/13/25.
//

import MetalPerformanceShadersGraph
import Alloy
import Metal
import Foundation

extension NDArray {
    /// Creates an NDArray whose values are drawn from a normal distribution
    /// using MPSGraph’s built-in random generator:
    ///   mean: `mean`, stddev: `std`, shape: `shape`.
    /// - Parameters:
    ///   - shape: Desired shape of the output tensor
    ///   - mean: Mean (μ) of the distribution
    ///   - std: Standard deviation (σ) of the distribution
    ///   - seed: The seed to make random draws reproducible
    ///   - label: Optional debug label
    /// - Returns: A *lazy* NDArray. Real data is produced on `run(...)`.
    public static func randn(
        shape: [Int],
        mean: Float = 0,
        std: Float = 1,
        seed: Int = 0,
        label: String? = nil
    ) -> NDArray {
        // Basic shape checks
        for dim in shape {
            precondition(dim > 0, "Shape dimensions must be > 0. Got \(dim)")
        }
        
        return NDArray(
            shape: shape,
            label: label,
            parents: [],         // No parent NDArrays; it’s a “source” op
            op: { graph, inputs, nodeLabel in
                // We expect 0 inputs since we have no parents
                guard inputs.isEmpty else {
                    throw NDArrayError.operationError("randn op expects 0 inputs.")
                }
                
                // 1. Create a shape tensor as [Int32]
                let intShape = shape.map { Int32($0) }
                
                // Safely convert [Int32] to Data using withUnsafeBufferPointer
                let shapeData = intShape.withUnsafeBufferPointer { buffer -> Data in
                    return Data(buffer: buffer)
                }
                
                // Create the shape tensor
                let shapeTensor = graph.constant(
                    shapeData,
                    shape: [NSNumber(value: intShape.count)],
                    dataType: .int32
                )
                
                // 2. Create descriptor for normal distribution
                guard let desc = MPSGraphRandomOpDescriptor(
                    distribution: .normal,
                    dataType: .float32
                ) else {
                    throw NDArrayError.operationError("Failed to create MPSGraphRandomOpDescriptor.")
                }
                desc.mean = mean
                desc.standardDeviation = std
                
                // 3. Generate random tensor
                let randTensor = graph.randomTensor(
                    withShapeTensor: shapeTensor,
                    descriptor: desc,
                    seed: seed,
                    name: nodeLabel
                )
                
                return randTensor
            }
        )
    }
}
