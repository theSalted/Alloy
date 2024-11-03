import MetalPerformanceShaders
import MetalKit

class NDArray {
    var shape: [Int]
    var stride: [Int]
    var data: [Float]
    var device: MTLDevice
    var buffer: MTLBuffer?
    
    init(shape: [Int], device: MTLDevice) {
        self.shape = shape
        self.device = device
        self.stride = NDArray.calculateStride(shape: shape)
        self.data = Array(repeating: 0.0, count: shape.reduce(1, *))
        
        // Platform-specific memory options
        #if os(iOS)
        let resourceOptions: MTLResourceOptions = .storageModeShared
        #elseif os(macOS)
        let resourceOptions: MTLResourceOptions = .storageModeShared
        #endif
        
        // Allocate GPU memory if Metal is supported on this device
        buffer = device.makeBuffer(length: MemoryLayout<Float>.size * data.count, options: resourceOptions)
    }
    
    static func calculateStride(shape: [Int]) -> [Int] {
        var stride = Array(repeating: 1, count: shape.count)
        for i in stride.indices.dropLast().reversed() {
            stride[i] = stride[i + 1] * shape[i + 1]
        }
        return stride
    }
    
    subscript(indices: [Int]) -> Float {
        get {
            let index = calculateIndex(indices: indices)
            return data[index]
        }
        set {
            let index = calculateIndex(indices: indices)
            data[index] = newValue
        }
    }
    
    private func calculateIndex(indices: [Int]) -> Int {
        zip(indices, stride).map(*).reduce(0, +)
    }
    
    // Method to upload data to GPU buffer
    func uploadToGPU() {
        guard let buffer = buffer else { return }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: data.count)
        pointer.update(from: data, count: data.count)
        
        #if os(macOS)
        buffer.didModifyRange(0..<buffer.length) // Needed for macOS
        #endif
    }
    
    // Placeholder for operations such as matrix multiplication, broadcasting, etc.
}
