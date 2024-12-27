//
//  Alloy.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/2/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders
import Metal
import MetalKit

open class NDArray {
    public var label: String?
    public var device: MTLDevice
    public var commandQueue: MTLCommandQueue
    var buffer: MTLBuffer?
    var tensor: MPSGraphTensorData
    
    public init(shape: [Int],
                device: MTLDevice,
                data: [Float]? = nil,
                label: String? = nil)
    {
        self.device = device
        self.label = label
        self.commandQueue = device.makeCommandQueue()!
        
        // Compute total elements and buffer size
        let totalElements = shape.reduce(1, *)
        let dataSize = MemoryLayout<Float>.size * totalElements
        let resourceOptions: MTLResourceOptions = .storageModeShared
        
        // Create the Metal buffer
        guard let buffer = device.makeBuffer(length: dataSize, options: resourceOptions) else {
            fatalError("Failed to create MTLBuffer of size \(dataSize)")
        }
        
        // If data is provided, ensure it matches the shape
        if let data = data {
            guard data.count == totalElements else {
                fatalError("Data count (\(data.count)) does not match shape (\(shape)) with total elements = \(totalElements)")
            }
            
            // Copy data to GPU buffer
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: data.count)
            pointer.update(from: data, count: data.count)
        }
        
        self.buffer = buffer
        
        // Convert Swift [Int] to [NSNumber] if your MPSGraphTensorData initializer requires it
        let mpsShape = shape.map { NSNumber(value: $0) }
        
        self.tensor = MPSGraphTensorData(buffer,
                                         shape: mpsShape,
                                         dataType: .float32)
    }
}

extension NDArray: CustomDebugStringConvertible {
    public var debugDescription: String {
        ""
    }
}
