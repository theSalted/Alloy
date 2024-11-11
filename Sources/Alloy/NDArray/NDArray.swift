import MetalPerformanceShaders
import MetalKit

final public class NDArray {
    public var shape: [Int]
    public var label: String?
    public var device: MTLDevice
    public var commandQueue: MTLCommandQueue
    
    // Data on GPU
    var buffer: MTLBuffer?
    
    // Gradient on GPU
    var gradientBuffer: MTLBuffer?
    
    // For computational graph
    var operation: ((MTLCommandBuffer) -> Void)?
    var inputs: [NDArray] = []
    var _backward: (() -> Void)?
    
    // Metal function cache
    nonisolated(unsafe) static var functionCache: [String: MTLFunction] = [:]
    
    public init(shape: [Int], device: MTLDevice, data: [Float]? = nil, label: String? = nil) {
        self.shape = shape
        self.device = device
        self.label = label
        self.commandQueue = device.makeCommandQueue()!
        
        let dataSize = MemoryLayout<Float>.size * shape.reduce(1, *)
        let resourceOptions: MTLResourceOptions = .storageModeShared
        
        buffer = device.makeBuffer(length: dataSize, options: resourceOptions)
        gradientBuffer = device.makeBuffer(length: dataSize, options: resourceOptions)
        
        if let data = data {
            // Upload data to GPU
            let pointer = buffer!.contents().bindMemory(to: Float.self, capacity: data.count)
            pointer.update(from: data, count: data.count)
        }
    }
    
    // Forward computation (computes the value of this NDArray)
    public func forward() {
        guard let operation = operation else {
            // Leaf node; data is already in buffer
            return
        }
        
        // Ensure inputs are computed
        for input in inputs {
            input.forward()
        }
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Perform the operation
        operation(commandBuffer)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // Backward computation (computes gradients)
    public func backward() {
        // Initialize gradient to 1.0 for the root node
        let count = shape.reduce(1, *)
        let gradientInit = [Float](repeating: 1.0, count: count)
        gradientBuffer?.contents().copyMemory(from: gradientInit, byteCount: count * MemoryLayout<Float>.size)
        
        // Build topological order for backward pass
        let topo = makeTopologicalOrdered()
        for node in topo.reversed() {
            node._backward?()
        }
    }
    
    // Retrieve data from GPU to CPU
    public func evaluate() -> [Float] {
        let count = shape.reduce(1, *)
        let pointer = buffer!.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    // Retrieve gradient data from GPU to CPU
    public func gradient() -> [Float] {
        let count = shape.reduce(1, *)
        let pointer = gradientBuffer!.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    // Broadcast the shape of this array with another shape
    public func broadcast(with otherShape: [Int]) -> NDArray {
        let newShape = computeBroadcastShape(shape, otherShape)
        guard newShape == shape else {
            // Reshape to the new shape if current shape doesn't match the target broadcast shape
            return reshape(to: newShape)
        }
        return self
    }

    private func reshape(to newShape: [Int]) -> NDArray {
        let reshapedArray = NDArray(shape: newShape, device: device, label: label)
        reshapedArray.buffer = buffer // Share the buffer for memory efficiency
        reshapedArray.gradientBuffer = gradientBuffer
        return reshapedArray
    }
    
    // Compute the resulting shape after broadcasting
    private func computeBroadcastShape(_ shape1: [Int], _ shape2: [Int]) -> [Int] {
        var resultShape = [Int]()
        let maxDims = max(shape1.count, shape2.count)
        
        for i in 0..<maxDims {
            let dim1 = i < shape1.count ? shape1[shape1.count - 1 - i] : 1
            let dim2 = i < shape2.count ? shape2[shape2.count - 1 - i] : 1
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                fatalError("Shapes are not compatible for broadcasting: \(shape1) vs \(shape2)")
            }
            resultShape.insert(max(dim1, dim2), at: 0)
        }
        return resultShape
    }
}

extension NDArray: Equatable {
    public static func ==(lhs: NDArray, rhs: NDArray) -> Bool {
        return lhs === rhs
    }
}

extension NDArray: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}
