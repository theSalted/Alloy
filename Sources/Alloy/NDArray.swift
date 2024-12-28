//
//  NDArray.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/27/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders
import Metal

open class NDArray {
    // MARK: - Properties
    
    private var _value: [Float]
    public var shape: [Int]
    public private(set) var buffer: MTLBuffer
    public private(set) var graph: MPSGraph
    public private(set) var tensor: MPSGraphTensor
    public private(set) var tensorData: MPSGraphTensorData
    
    public var label: String?
    public private(set) var device: MTLDevice
    public private(set) var commandQueue: MTLCommandQueue
    
    public var `operator`: String?
    public var neighbors = Set<NDArray>()
    
    private let bufferLock = NSLock()
    
    // Computed property to manage value and buffer synchronization
    public var value: [Float] {
        get {
            bufferLock.lock()
            defer { bufferLock.unlock() }
            return _value
        }
        set {
            setValue(newValue)
        }
    }
    
    // MARK: - Initializer
    
    public init(_ value: [Float],
                shape: [Int],
                device: MTLDevice,
                label: String? = nil)
    {
        self._value = value
        self.shape = shape
        self.device = device
        self.label = label
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create MTLCommandQueue")
        }
        self.commandQueue = commandQueue
        
        // Compute total elements and buffer size
        let totalElements = shape.reduce(1, *)
        let dataSize = MemoryLayout<Float>.size * totalElements
        let resourceOptions: MTLResourceOptions = .storageModeShared
        
        // Create the Metal buffer
        guard let buffer = device.makeBuffer(length: dataSize, options: resourceOptions) else {
            fatalError("Failed to create MTLBuffer of size \(dataSize)")
        }
        
        // Ensure data matches the shape
        guard value.count == totalElements else {
            fatalError("Data count (\(value.count)) does not match shape (\(shape)) with total elements = \(totalElements)")
        }
        
        // Copy data to GPU buffer
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: value.count)
        pointer.update(from: value, count: value.count)
        
        self.buffer = buffer
        
        // Convert Swift [Int] to [NSNumber] for MPSGraph
        let mpsShape = shape.map { NSNumber(value: $0) }
        
        self.graph = MPSGraph()
        self.tensor = graph.placeholder(shape: mpsShape, dataType: .float32, name: label)
        self.tensorData = MPSGraphTensorData(buffer,
                                             shape: mpsShape,
                                             dataType: .float32)
    }
    
    // MARK: - Value Management
    
    /// Sets the value and updates the buffer accordingly.
    /// - Parameter newValue: The new array of Float values.
    public func setValue(_ newValue: [Float]) {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        
        guard newValue.count == _value.count else {
            fatalError("New value count (\(newValue.count)) does not match existing count (\(_value.count))")
        }
        
        // Update the CPU-side value
        _value = newValue
        
        // Update the GPU buffer
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: _value.count)
        pointer.update(from: _value, count: _value.count)
    }
    
    /// Synchronizes the buffer with the current value.
    /// Call this if you modify the buffer directly.
    public func synchronizeBuffer() {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        
        // Read back the buffer contents to _value
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: _value.count)
        _value = Array(UnsafeBufferPointer(start: pointer, count: _value.count))
    }
}

// MARK: - Equatable

extension NDArray: Equatable {
    public static func ==(lhs: NDArray, rhs: NDArray) -> Bool {
        return lhs === rhs // Reference equality
    }
}

// MARK: - Hashable

extension NDArray: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self)) // Use object identity
    }
}
