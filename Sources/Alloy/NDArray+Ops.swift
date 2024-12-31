//
//  NDArray+Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/27/24.
//

import MetalPerformanceShadersGraph
import Metal
import Foundation

// MARK: - Operator Overloads

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
    
    // MARK: - Exponentiation (^)
    
    /// Raises `lhs` to the power of `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The base `NDArray`.
    ///   - rhs: The exponent `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise exponentiation.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func ^ (lhs: NDArray, rhs: NDArray) -> NDArray {
        // Broadcast if needed
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") ^ \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`^` operator expects exactly 2 inputs.")
            }
            // Typically you’d call a `power` method on your graph backend:
            return graph.power(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Raises `lhs` to the power of a `Float`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The base `NDArray`.
    ///   - rhs: The exponent `Float`.
    /// - Returns: A new `NDArray` representing the element-wise exponentiation.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func ^ (lhs: NDArray, rhs: Float) -> NDArray {
        lhs ^ NDArray([rhs], shape: [1], label: "rhs")
    }
    
    /// Raises a `Float` to the power of `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The base `Float`.
    ///   - rhs: The exponent `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise exponentiation.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func ^ (lhs: Float, rhs: NDArray) -> NDArray {
        NDArray([lhs], shape: [1], label: "lhs") ^ rhs
    }
    
    // MARK: - Modulus (%)
    
    /// Computes the modulus of `lhs` by `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The dividend `NDArray`.
    ///   - rhs: The divisor `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise modulus.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func % (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") % \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`%` operator expects exactly 2 inputs.")
            }
            return graph.modulo(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Computes the modulus of `lhs` by a `Float`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The dividend `NDArray`.
    ///   - rhs: The divisor `Float`.
    /// - Returns: A new `NDArray` representing the element-wise modulus.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func % (lhs: NDArray, rhs: Float) -> NDArray {
        lhs % NDArray([rhs], shape: [1], label: "rhs")
    }
    
    /// Computes the modulus of a `Float` by `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The dividend `Float`.
    ///   - rhs: The divisor `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise modulus.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func % (lhs: Float, rhs: NDArray) -> NDArray {
        NDArray([lhs], shape: [1], label: "lhs") % rhs
    }
    
    // MARK: - Comparisons (>, <, ==, !=)
    //
    // WARNING: Overriding these for NDArray → NDArray
    // returning another NDArray is non-standard in Swift.
    // Proceed with caution.
    
    /// Compares two NDArrays element-wise, returns 1.0 if `lhs` > `rhs`, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise comparison.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func > (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") > \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`>` operator expects exactly 2 inputs.")
            }
            // Suppose your graph has `greater()`:
            return graph.greaterThan(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Compares two NDArrays element-wise, returns 1.0 if `lhs` < `rhs`, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise comparison.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func < (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") < \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`<` operator expects exactly 2 inputs.")
            }
            return graph.lessThan(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Compares two NDArrays element-wise, returns 1.0 if `lhs` == `rhs`, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise comparison.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func == (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") == \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`==` operator expects exactly 2 inputs.")
            }
            return graph.equal(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Compares two NDArrays element-wise, returns 1.0 if `lhs` != `rhs`, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise comparison.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func != (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") != \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`!=` operator expects exactly 2 inputs.")
            }
            return graph.notEqual(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    // MARK: - Logical AND, OR, NOT (&&, ||, !)
    //
    // NOTE: Swift normally does NOT allow overloading `&&` or `||`
    // for custom types due to short-circuiting rules.
    // The code below may not compile in a real Swift project.
    // A more common pattern is to define custom operators
    // like `.&.` or `.|.` for element-wise logical ops.
    
    /// Logical AND (element-wise). Returns 1.0 if both elements are non-zero, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise logical AND.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func && (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") && \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`&&` operator expects exactly 2 inputs.")
            }
            // Typically you'd have something like `graph.logicalAnd(...)`
            return graph.logicalAND(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Logical OR (element-wise). Returns 1.0 if at least one element is non-zero, else 0.0.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand side `NDArray`.
    ///   - rhs: The right-hand side `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise logical OR.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func || (lhs: NDArray, rhs: NDArray) -> NDArray {
        var rhs = rhs
        if lhs.shape != rhs.shape {
            rhs = rhs.broadcasted(lhs.shape)
        }
        
        let label = "\(lhs.label ?? "lhs") || \(rhs.label ?? "rhs")"
        
        return NDArray(
            shape: lhs.shape,
            label: label,
            parents: [lhs, rhs]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 2 else {
                throw NDArrayError.operationError("`||` operator expects exactly 2 inputs.")
            }
            // Typically you'd have something like `graph.logicalOr(...)`
            return graph.logicalOR(inputs[0], inputs[1], name: nodeLabel)
        }
    }
    
    /// Logical NOT (element-wise). Returns 1.0 if original element is 0.0, else 0.0.
    ///
    /// - Parameter value: The `NDArray` to negate logically.
    /// - Returns: A new `NDArray` representing the element-wise logical NOT.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static prefix func ! (value: NDArray) -> NDArray {
        let label = "!(\(value.label ?? "value"))"
        
        return NDArray(
            shape: value.shape,
            label: label,
            parents: [value]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`!` operator expects exactly 1 input.")
            }
            // Typically you'd have something like `graph.logicalNot(...)`
            return graph.not(with: inputs[0], name: nodeLabel)
        }
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
