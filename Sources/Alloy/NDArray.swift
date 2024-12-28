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
    
    /// The shape of the tensor (e.g. [1, 3, 224, 224]).
    public var shape: [Int]
    
    /// Optional GPU buffer. Only allocated/populated once you call `materialize(_:)`.
    public var buffer: MTLBuffer?
    
    /// Optional MPSGraph. Only created after materializing or manually building the graph.
    public var graph: MPSGraph?
    
    /// Optional MPSGraphTensor. Represents the symbolic placeholder or operator output.
    public var tensor: MPSGraphTensor?
    
    /// Optional data wrapper for MPSGraphTensor. Used during graph execution.
    public var tensorData: MPSGraphTensorData?
    
    /// Optional label for debugging or identification.
    public var label: String?
    
    /// The Metal device used for GPU operations.
    public private(set) var device: MTLDevice
    
    /// The command queue used to submit Metal command buffers.
    public private(set) var commandQueue: MTLCommandQueue
    
    /// An optional operator name or type, useful for debugging or identifying what created this NDArray.
    public var `operator`: String?
    
    /// A set of neighbor NDArrays used for graph analysis (e.g., for topological ordering).
    public var neighbors = Set<NDArray>()
    
    // MARK: - Initializer
    
    /// Creates a new NDArray in a “symbolic” (unmaterialized) state.
    ///
    /// - Parameters:
    ///   - shape: The tensor shape.
    ///   - device: The Metal device for GPU ops.
    ///   - label: Optional label for this NDArray.
    public init(shape: [Int],
                device: MTLDevice? = nil,
                label: String? = nil)
    {
        self.shape = shape
        
        var device = device
        if device == nil {
            device = MTLCreateSystemDefaultDevice()
        }
        
        guard let device else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        self.label = label
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create MTLCommandQueue")
        }
        self.commandQueue = commandQueue
        
        // At this point, `buffer`, `graph`, `tensor`, and `tensorData` remain nil.
        // You can materialize later via `materialize(from:)`.
    }
    
    convenience init(_ values: [Float], shape: [Int], device: MTLDevice? = nil, label: String? = nil) {
        self.init(shape: shape, device: device, label: label)
        materialize(from: values)
    }
    
    convenience init(elements: (NDArray?, NDArray?), operator: String, shape: [Int]) {
        self.init(shape: shape)
        
        if let leftChild = elements.0 {
            self.neighbors.insert(leftChild)
        }
        if let rightChild = elements.1 {
            self.neighbors.insert(rightChild) // Thread 1: Fatal error: Duplicate elements of type 'Scalar' were found in a Set.
        }
        self.operator = `operator`
        
    }
    
    // MARK: - Materialization
    
    /// Materializes the NDArray by allocating a GPU buffer and creating MPSGraph components.
    ///
    /// - Parameter values: The actual data to store. Must match the total element count implied by `shape`.
    public func materialize(from values: [Float]) {
        let totalElements = shape.reduce(1, *)
        guard values.count == totalElements else {
            fatalError("Data count (\(values.count)) does not match shape \(shape) with total elements = \(totalElements)")
        }
        
        // Create the buffer if not already present
        if buffer == nil {
            let dataSize = MemoryLayout<Float>.size * totalElements
            let resourceOptions: MTLResourceOptions = .storageModeShared
            
            guard let newBuffer = device.makeBuffer(length: dataSize, options: resourceOptions) else {
                fatalError("Failed to create MTLBuffer of size \(dataSize)")
            }
            self.buffer = newBuffer
        }
        
        // Copy data into buffer
        if let buffer = buffer {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
            pointer.update(from: values, count: values.count)
        }
        
        // Convert shape to [NSNumber]
        let mpsShape = shape.map { NSNumber(value: $0) }
        
        // Create or update the graph objects
        if graph == nil {
            graph = MPSGraph()
        }
        guard let graph = graph else {
            fatalError("Failed to create or retrieve MPSGraph.")
        }
        
        // Create the placeholder or operator tensor if not present
        if tensor == nil {
            tensor = graph.placeholder(shape: mpsShape, dataType: .float32, name: label)
        }
        
        // Create or update the tensorData object
        if let buffer = buffer,
           let _ = tensor
        {
            tensorData = MPSGraphTensorData(buffer,
                                            shape: mpsShape,
                                            dataType: .float32)
        }
    }
    
    // MARK: - Value Management
    
    /// Updates the GPU buffer with new data (if materialized).
    /// - Parameter newValue: The new array of Float values.
    public func updateBuffer(with newValue: [Float]) {
        let totalElements = shape.reduce(1, *)
        guard newValue.count == totalElements else {
            fatalError("New value count (\(newValue.count)) does not match shape (\(shape)) with total elements = \(totalElements)")
        }
        
        // If not materialized yet, materialize now
        if buffer == nil {
            materialize(from: newValue)
            return
        }
        
        // Otherwise, just update existing buffer
        if let buffer = buffer {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: newValue.count)
            pointer.update(from: newValue, count: newValue.count)
        }
    }
    
    /// Retrieves the data from the GPU buffer (if materialized).
    /// - Returns: An array of Float values. Returns empty if there’s no buffer.
    public func fetchData() -> [Float] {
        guard let buffer = buffer else {
            // Not materialized yet, so there’s no actual data on GPU to fetch.
            return []
        }
        
        let totalElements = shape.reduce(1, *)
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        return Array(UnsafeBufferPointer(start: pointer, count: totalElements))
    }
    
    /// Executes the graph operation associated with this NDArray (if materialized).
    /// - Parameter inputs: A dictionary mapping tensor names (or placeholders) to their corresponding tensor data.
    /// - Returns: The output tensor data resulting from the graph execution, or nil if not materialized or missing components.
    public func runGraph(inputs: [String: MPSGraphTensorData] = [:]) -> MPSGraphTensorData? {
        guard
            let graph = graph,
            let tensor = tensor,
            let tensorData = tensorData
        else {
            // If any of these are nil, we can't actually run anything
            return nil
        }
        
        // Merge “self”’s feed with external feeds
        var feeds = [MPSGraphTensor: MPSGraphTensorData]()
        feeds[tensor] = tensorData
        
        // Attempt to match any other placeholders by name if provided
        /*for (inputName, inputData) in inputs {
            // If the graph has placeholders with `name == inputName`, attach them here
            // (Naive approach: we'd need a reference to each input MPSGraphTensor for this to be robust.)
            // For a simplistic approach, skip or handle naming logic as needed.
        }*/
        
        // Actually run the graph
        let result = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [tensor],
            targetOperations: nil
        )
        return result.values.first
    }
    
    // MARK: - Graph Ordering
    
    /// Returns a topologically sorted list of NDArrays that start from `self`.
    public func makeTopologicalOrdered() -> [NDArray] {
        var visited = Set<NDArray>()
        var topo = [NDArray]()
        
        NDArray.buildTopologicalOrder(from: self, visited: &visited, to: &topo)
        
        return topo
    }
    
    fileprivate static func buildTopologicalOrder(from s: NDArray,
                                                  visited: inout Set<NDArray>,
                                                  to topo: inout [NDArray])
    {
        if !visited.contains(s) {
            visited.insert(s)
            for child in s.neighbors {
                buildTopologicalOrder(from: child, visited: &visited, to: &topo)
            }
            topo.append(s)
        }
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
