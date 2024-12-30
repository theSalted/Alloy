//
//  NDArray+Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/27/24.
//

import MetalPerformanceShadersGraph
import Metal
import Foundation

// MARK: - Operator Overloads (e.g., +, -, *, /)

extension NDArray {
    
    // MARK: - Binary Operators (NDArray ↔ NDArray)
    
    /// Adds two `NDArray` instances element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise sum.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func +(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // Broadcast rhs to match lhs shape if necessary
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") + \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`+` operator expects exactly 2 inputs.")
            }
            return graph.addition(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Subtracts the second `NDArray` from the first element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise difference.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func -(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // Broadcast rhs to match lhs shape if necessary
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") - \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`-` operator expects exactly 2 inputs.")
            }
            return graph.subtraction(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Multiplies two `NDArray` instances element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise product.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func *(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // Broadcast rhs to match lhs shape if necessary
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") * \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`*` operator expects exactly 2 inputs.")
            }
            return graph.multiplication(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Divides the first `NDArray` by the second element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `NDArray`.
    ///   - rhs: The denominator `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise division.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func /(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // Broadcast rhs to match lhs shape if necessary
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") / \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`/` operator expects exactly 2 inputs.")
            }
            return graph.division(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    // MARK: - Binary Operators (NDArray ↔ Float)
    
    /// Adds a `Float` to an `NDArray` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `NDArray`.
    ///   - rhs: The `Float` to add.
    /// - Returns: A new `NDArray` representing the element-wise sum.
    public static func +(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs + NDArray([rhs], shape: [1], label: "rhs")
    }
    
    /// Subtracts a `Float` from an `NDArray` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `NDArray`.
    ///   - rhs: The `Float` to subtract.
    /// - Returns: A new `NDArray` representing the element-wise difference.
    public static func -(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs - NDArray([rhs], shape: [1], label: "rhs")
    }
    
    /// Multiplies an `NDArray` by a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `NDArray`.
    ///   - rhs: The `Float` to multiply.
    /// - Returns: A new `NDArray` representing the element-wise product.
    public static func *(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs * NDArray([rhs], shape: [1], label: "rhs")
    }
    
    /// Divides an `NDArray` by a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `NDArray`.
    ///   - rhs: The `Float` denominator.
    /// - Returns: A new `NDArray` representing the element-wise division.
    public static func /(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs / NDArray([rhs], shape: [1], label: "rhs")
    }
    
    // MARK: - Binary Operators (Float ↔ NDArray)
    
    /// Adds an `NDArray` to a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `Float` to add.
    ///   - rhs: The `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise sum.
    public static func +(_ lhs: Float, _ rhs: NDArray) -> NDArray {
        // Wrap the Float into an NDArray with shape [1]
        let lhsNDArray = NDArray([lhs], shape: [1], label: "lhs")
        return lhsNDArray + rhs
    }
    
    /// Subtracts an `NDArray` from a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `Float` minuend.
    ///   - rhs: The `NDArray` subtrahend.
    /// - Returns: A new `NDArray` representing the element-wise difference.
    public static func -(_ lhs: Float, _ rhs: NDArray) -> NDArray {
        // Wrap the Float into an NDArray with shape [1]
        let lhsNDArray = NDArray([lhs], shape: [1], label: "lhs")
        return lhsNDArray - rhs
    }
    
    /// Multiplies a `Float` by an `NDArray` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `Float` multiplier.
    ///   - rhs: The `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise product.
    public static func *(_ lhs: Float, _ rhs: NDArray) -> NDArray {
        // Wrap the Float into an NDArray with shape [1]
        let lhsNDArray = NDArray([lhs], shape: [1], label: "lhs")
        return lhsNDArray * rhs
    }
    
    /// Divides a `Float` by an `NDArray` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `Float` numerator.
    ///   - rhs: The `NDArray` denominator.
    /// - Returns: A new `NDArray` representing the element-wise division.
    public static func /(_ lhs: Float, _ rhs: NDArray) -> NDArray {
        // Wrap the Float into an NDArray with shape [1]
        let lhsNDArray = NDArray([lhs], shape: [1], label: "lhs")
        return lhsNDArray / rhs
    }
    
    // MARK: - Unary Operators
    
    /// Negates an `NDArray` element-wise.
    ///
    /// - Parameter value: The `NDArray` to negate.
    /// - Returns: A new `NDArray` representing the element-wise negation.
    public static prefix func -(_ value: NDArray) -> NDArray {
        return -1 * value
    }
}

// MARK: - Reshaping and Broadcasting Helpers

extension NDArray {
    /// Reshapes the `NDArray` to a new shape.
    ///
    /// - Parameter shape: The new shape for the `NDArray`.
    /// - Returns: A new `NDArray` with the specified shape.
    public func reshaped(_ shape: [Int]) -> NDArray {
        return reshape(self, to: shape)
    }
    
    /// Broadcasts the `NDArray` to a new shape.
    ///
    /// - Parameter shape: The target shape for broadcasting.
    /// - Returns: A new `NDArray` broadcasted to the specified shape.
    public func broadcasted(_ shape: [Int]) -> NDArray {
        return broadcast(self, to: shape)
    }
}
