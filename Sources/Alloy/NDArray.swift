import MetalPerformanceShadersGraph
import Metal
import Foundation
import AlloyUtils

// MARK: - NDArrayOp

/// A closure that can handle N input MPSGraphTensors and produce a single MPSGraphTensor.
/// In real-world production, you could expand this to support multiple outputs, error handling, etc.
public typealias NDArrayOp = (
    _ graph: MPSGraph,
    _ inputs: [MPSGraphTensor],
    _ label: String?
) throws -> MPSGraphTensor

// MARK: - NDArrayError

/// Basic errors for NDArray operations and graph building.
public enum NDArrayError: Error {
    case emptyDAG(String)
    case operationError(String)
}

// MARK: - NDArray

/// `NDArray` is a node in a DAG representing a tensor computation.
/// - **Leaf Nodes**: Have no `op` and may contain raw CPU data (`data`). If `data` is `nil`, it's treated as a placeholder during graph execution.
/// - **Internal Nodes**: Contain an `op`, as well as references to one or more `parents`.
public class NDArray: Hashable {
    
    // MARK: - Public Properties
    
    /// A human-readable label or name for debugging and identification.
    public var label: String?
    
    /// The shape of this NDArray. If you plan on broadcasting or dynamic shapes, you'll need more sophisticated logic.
    public var shape: [Int]
    
    /// If this node is a leaf, you can store CPU-side data here.
    /// If `nil`, and `op == nil`, this node becomes a placeholder during graph building.
    public var data: Data?
    
    public var grad: Data?

    /// A list of parent NDArrays. If this node is an internal operation node,
    /// the operation `op` is applied to all parents’ outputs.
    public var parents: [NDArray] = []
    
    /// An optional operation closure. If `nil`, this node is a leaf.
    public var op: NDArrayOp?
    
    // MARK: - Initializers
    
    /// Creates a **leaf node**.
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - label: An optional name/label (debugging).
    ///   - data: If provided, this node becomes a constant in the graph.
    ///           If `nil`, it's treated as a placeholder.
    public init(
        _ value: [Float]? = nil,
        shape: [Int],
        label: String? = nil
    ) {
        // Basic shape checks
        for dim in shape {
            precondition(dim > 0, "Shape dimensions must be > 0. Got \(dim)")
        }
        if let value {
            let prod = shape.reduce(1, *)
            precondition(value.count == prod,
                         "Data count (\(value.count)) != shape’s element count (\(prod))")
        }
        
        self.shape = shape
        self.label = label
        self.data  = value?.toData(shape: shape)
        self.op    = nil  // Leaf => no op
    }
    
    /// Creates an **internal node**.
    /// - Parameters:
    ///   - shape: The shape of this node’s output. Could be derived from parents (e.g. broadcast).
    ///   - label: Optional debug label.
    ///   - parents: The parent NDArrays.
    ///   - op: The operation to build an `MPSGraphTensor` from these parents.
    public init(
        shape: [Int],
        label: String? = nil,
        parents: [NDArray],
        op: @escaping NDArrayOp
    ) {
        for dim in shape {
            precondition(dim > 0, "Shape dimensions must be > 0. Got \(dim)")
        }
        
        self.shape   = shape
        self.label   = label
        self.parents = parents
        self.op      = op
    }
    
    // MARK: - Hashable & Equatable
    
    public static func == (lhs: NDArray, rhs: NDArray) -> Bool {
        lhs === rhs
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
    
    public func toArray() throws -> [Float] {
        guard let rawData = self.data else {
            return []
        }
        
        let elementCount = shape.reduce(1, *)
        let expectedByteCount = elementCount * MemoryLayout<Float>.size
        
        // Compare the size in bytes, not the size in elements
        if rawData.count != expectedByteCount {
            throw NDArrayError.operationError("""
            NDArray(shape: \(shape), 
                    data byte-count: \(rawData.count) 
                    != expected byte-count: \(expectedByteCount))
            """)
        }
        
        // Now convert bytes to [Float] for representation
        let floatArray = rawData.toFloatArray() ?? []
        
        // If the float array is empty or has a mismatch, bail early
        if floatArray.count != elementCount {
            throw NDArrayError.operationError("""
            NDArray(shape: \(shape), 
                    data float-count: \(floatArray.count) 
                    != expected float-count: \(elementCount))
            """)
        }
        
        return floatArray
    }
}

// MARK: - DAG Helpers

extension NDArray {
    /// Topological sort for the DAG rooted at `self`.
    /// Ensures that parents come before children in the returned array.
    public func topologicalSort() -> [NDArray] {
        var visited = Set<NDArray>()
        var sorted  = [NDArray]()
        
        func dfs(_ node: NDArray) {
            if visited.contains(node) { return }
            visited.insert(node)
            for parent in node.parents {
                dfs(parent)
            }
            sorted.append(node)
        }
        
        dfs(self)
        return sorted
    }
}


extension NDArray: CustomDebugStringConvertible {
    public var debugDescription: String {
        guard let rawData = self.data else {
            return "NDArray(shape: \(shape), data: nil)"
        }
        
        let elementCount = shape.reduce(1, *)
        let expectedByteCount = elementCount * MemoryLayout<Float>.size
        
        // Compare the size in bytes, not the size in elements
        if rawData.count != expectedByteCount {
            return """
            NDArray(shape: \(shape), 
                    data byte-count: \(rawData.count) 
                    != expected byte-count: \(expectedByteCount))
            """
        }
        
        // Now convert bytes to [Float] for representation
        let floatArray = rawData.toFloatArray() ?? []
        
        // If the float array is empty or has a mismatch, bail early
        if floatArray.count != elementCount {
            return """
            NDArray(shape: \(shape), 
                    data float-count: \(floatArray.count) 
                    != expected float-count: \(elementCount))
            """
        }
        
        // Define a max number of elements to show per dimension
        let maxElements = 5
        
        func buildString(from data: [Float], shape: [Int]) -> String {
            guard !shape.isEmpty else { return "\(data[0])" }
            
            let currentDim = shape[0]
            let remainingShape = Array(shape.dropFirst())
            
            let elementsToShow = Swift.min(currentDim, maxElements)
            let showEllipsis = (currentDim > maxElements)
            
            var result = "["
            for i in 0..<elementsToShow {
                let startIndex = i * (data.count / currentDim)
                let endIndex = (i + 1) * (data.count / currentDim)
                let slice = Array(data[startIndex..<endIndex])
                
                result += buildString(from: slice, shape: remainingShape)
                if i < elementsToShow - 1 {
                    result += ", "
                }
            }
            if showEllipsis {
                result += ", …"
            }
            result += "]"
            return result
        }
        
        let debugStr = buildString(from: floatArray, shape: shape)
        return "NDArray(shape: \(shape), data: \(debugStr))"
    }
}

extension Collection where Element == NDArray {
    /// Performs a topological sort on multiple NDArray roots.
    /// The result is guaranteed to have parents before children.
    func multiRootTopologicalSort() -> [NDArray] {
        var visited = Set<NDArray>()
        var sorted  = [NDArray]()
        
        func dfs(_ node: NDArray) {
            if visited.contains(node) { return }
            visited.insert(node)
            for parent in node.parents {
                dfs(parent)
            }
            sorted.append(node)
        }
        
        // For each "root", DFS to collect all ancestors
        for root in self {
            dfs(root)
        }
        
        // Remove duplicates but maintain the partial order.
        // Because we only push nodes when we finish their DFS,
        // the array is already topologically sorted.
        // We just need to keep them in the order encountered.
        // A stable way to do that is:
        var deduplicated = [NDArray]()
        var seen = Set<NDArray>()
        for node in sorted {
            if !seen.contains(node) {
                deduplicated.append(node)
                seen.insert(node)
            }
        }
        
        return deduplicated
    }
}
