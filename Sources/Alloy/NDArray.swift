import MetalPerformanceShadersGraph
import Metal
import Foundation

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
    public var data: [Float]?
    
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
        _ data: [Float]? = nil,
        shape: [Int],
        label: String? = nil
    ) {
        // Basic shape checks
        for dim in shape {
            precondition(dim > 0, "Shape dimensions must be > 0. Got \(dim)")
        }
        if let d = data {
            let prod = shape.reduce(1, *)
            precondition(d.count == prod,
                         "Data count (\(d.count)) != shape’s element count (\(prod))")
        }
        
        self.shape = shape
        self.label = label
        self.data  = data
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
    /// Provides a human-readable debug description of the NDArray.
    public var debugDescription: String {
        guard let flatData = self.data else {
            return "NDArray(shape: \(shape), data: nil)"
        }
        
        // Check if the shape is valid with the data count
        let expectedCount = shape.reduce(1, *)
        if flatData.count != expectedCount {
            return "NDArray(shape: \(shape), data count: \(flatData.count) does not match shape)"
        }
        
        // Define the maximum number of elements to display per dimension to prevent overly long descriptions
        let maxElements = 5
        
        // A helper function to recursively build the nested string
        func buildString(from data: [Float], shape: [Int], depth: Int = 0) -> String {
            // Base case: no more dimensions, return a single element
            if shape.isEmpty {
                return "\(data[0])"
            }
            
            let currentDim = shape[0]
            let remainingShape = Array(shape.dropFirst())
            var result = "["
            
            // Determine how many elements to show
            let elementsToShow = currentDim > maxElements ? maxElements : currentDim
            let showEllipsis = currentDim > maxElements
            
            for i in 0..<elementsToShow {
                let startIndex = i * (data.count / currentDim)
                let endIndex = (i + 1) * (data.count / currentDim)
                let slice = Array(data[startIndex..<endIndex])
                result += buildString(from: slice, shape: remainingShape, depth: depth + 1)
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
        
        // To handle multi-dimensional data, we need to chunk the flat data accordingly
        func reshape(data: [Float], shape: [Int]) -> [Any] {
            if shape.isEmpty {
                return [data.first!]
            }
            let dim = shape[0]
            let subShape = Array(shape.dropFirst())
            let size = subShape.reduce(1, *)
            var result: [Any] = []
            for i in 0..<dim {
                let start = i * size
                let end = start + size
                let chunk = Array(data[start..<end])
                result.append(contentsOf: reshape(data: chunk, shape: subShape))
            }
            return result
        }
        
        // For simplicity, we'll limit the nesting depth and elements displayed
        let description = buildString(from: flatData, shape: self.shape)
        return "NDArray(shape: \(shape), data: \(description))"
    }
}
