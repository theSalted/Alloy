import Metal

class KernelManager {
    private var cache: [String: MTLFunction] = [:]
    private init() {}
    
    nonisolated(unsafe) public static let shared = KernelManager()
    
    // Method to retrieve a function from the cache or load it if not present
    public func getFunction(device: MTLDevice, name: String) -> MTLFunction? {
        if let function = cache[name] {
            return function
        }
        // Load function from the default library
        guard let defaultLibrary = device.makeDefaultLibrary(),
              let function = defaultLibrary.makeFunction(name: name) else {
            fatalError("Failed to load function '\(name)' from default library")
        }
        cache[name] = function
        return function
    }
}
