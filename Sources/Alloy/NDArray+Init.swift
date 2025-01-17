//
//  NDArray+Init.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/1/25.
//

import MetalPerformanceShaders

extension NDArray {
    /// Creates a **leaf node** directly from an `MPSNDArray`.
    /// - Parameters:
    ///   - mpsArray: The source `MPSNDArray`.
    ///   - label: An optional name/label (debugging).
    ///
    /// This initializer reads the GPU-side contents of the `MPSNDArray`,
    /// copies them into CPU memory (`Data`), and sets `shape` accordingly.
    ///
    /// **Note**: This assumes the `MPSNDArray` is in `.float32` format.
    /// If you have a different data type, you must handle conversion before or after creating the NDArray.
    public convenience init(mpsArray: MPSNDArray, shape: [Int]? = nil, label: String? = nil) {
        // Retrieve the descriptor by calling mpsArray.descriptor()
        let desc = mpsArray.descriptor()
        
        // Use numberOfDimensions to figure out how many dimensions are in the MPSNDArray
        let dimensionCount = desc.numberOfDimensions
        
        let shape: [Int] = {
            guard let shape else {
                // Build the shape array by querying each dimension’s length
                var extractedShape = [Int]()
                extractedShape.reserveCapacity(dimensionCount)
                for i in 0..<dimensionCount {
                    extractedShape.append(Int(desc.length(ofDimension: i)))
                }
                return extractedShape
            }
            return shape
        }()
        
        
        // Calculate how many total elements
        let elementCount = shape.reduce(1, *)
        
        // Assume the MPSNDArray data is float32, so total bytes = elementCount * 4
        let byteCount = elementCount * MemoryLayout<Float>.size
        
        // Allocate a Data buffer of the required size
        var buffer = Data(count: byteCount)
        
        // Read the GPU-side bytes into our CPU buffer
        buffer.withUnsafeMutableBytes { rawBufferPtr in
            guard let ptr = rawBufferPtr.baseAddress else { return }
            mpsArray.readBytes(ptr, strideBytes: nil)
        }
        
        // Use the existing NDArray init for a leaf node
        self.init(shape: shape, label: label)
        
        // Store the data we just copied
        self.data = buffer
    }
}
