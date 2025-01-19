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
        lhs + NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
    }
    
    /// Subtracts a `Float` from an `NDArray` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `NDArray`.
    ///   - rhs: The `Float` to subtract.
    /// - Returns: A new `NDArray` representing the element-wise difference.
    public static func -(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs - NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
    }
    
    /// Multiplies an `NDArray` by a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The `NDArray`.
    ///   - rhs: The `Float` to multiply.
    /// - Returns: A new `NDArray` representing the element-wise product.
    public static func *(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs * NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
    }
    
    /// Divides an `NDArray` by a `Float` element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `NDArray`.
    ///   - rhs: The `Float` denominator.
    /// - Returns: A new `NDArray` representing the element-wise division.
    public static func /(_ lhs: NDArray, _ rhs: Float) -> NDArray {
        lhs / NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
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
        return lhsNDArray.broadcasted(rhs.shape) + rhs
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
        return lhsNDArray.broadcasted(rhs.shape) - rhs
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
        return lhsNDArray.broadcasted(rhs.shape) * rhs
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
        return lhsNDArray.broadcasted(rhs.shape) / rhs
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
        lhs ^ NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
    }
    
    /// Raises a `Float` to the power of `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The base `Float`.
    ///   - rhs: The exponent `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise exponentiation.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func ^ (lhs: Float, rhs: NDArray) -> NDArray {
        NDArray([lhs], shape: [1], label: "lhs").broadcasted(rhs.shape) ^ rhs
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
        lhs % NDArray([rhs], shape: [1], label: "rhs").broadcasted(lhs.shape)
    }
    
    /// Computes the modulus of a `Float` by `rhs`, element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The dividend `Float`.
    ///   - rhs: The divisor `NDArray`.
    /// - Returns: A new `NDArray` representing the element-wise modulus.
    /// - Throws: `NDArrayError.operationError` if the number of inputs is incorrect.
    public static func % (lhs: Float, rhs: NDArray) -> NDArray {
        NDArray([lhs], shape: [1], label: "lhs").broadcasted(rhs.shape) % rhs
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

extension NDArray {
    // MARK: - Reduction Operations
    /// Computes the reduction sum of all elements in the `NDArray`.
    public func sum(axis: Int? = nil) -> NDArray {
        let label = "\(self.label ?? "NDArray").sum()"
        
        return NDArray(
            shape: [], // Scalar result
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`sum` operation expects exactly 1 input.")
            }
            if let axis {
                return graph.reductionSum(with: inputs[0], axis: axis, name: nodeLabel)
            }
            return graph.reductionSum(with: inputs[0], axes: [], name: nodeLabel)
        }
    }
    
    /// Computes the sum of all elements in the `NDArray`.
    ///
    /// - Returns: A new `NDArray` representing the sum of all elements.
    /// - Throws: `NDArrayError.operationError` if the operation fails.
    public func cumsum(axis: Int,
                       reverse: Bool = false,
                       inclusive: Bool = true) -> NDArray {
        let label = "\(self.label ?? "NDArray").sum()"
        
        return NDArray(
            shape: [], // Scalar result
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`sum` operation expects exactly 1 input.")
            }
            return graph.cumulativeSum(inputs[0], axis: axis, exclusive: inclusive, reverse: reverse, name: nodeLabel)
        }
    }
    
    /// Computes the maximum value among all elements in the `NDArray`.
    ///
    /// - Returns: A new `NDArray` representing the maximum value.
    /// - Throws: `NDArrayError.operationError` if the operation fails.
    public func max(axis: Int? = nil) -> NDArray {
        let label = "\(self.label ?? "NDArray").max()"
        
        return NDArray(
            shape: [], // Scalar result
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`max` operation expects exactly 1 input.")
            }
            if let axis {
                return graph.reductionMinimum(with: inputs[0], axis: axis, name: nodeLabel)
            }
            return graph.reductionMinimum(with: inputs[0], axes: nil, name: nodeLabel)
        }
    }
    
    /// Computes the minimum value among all elements in the `NDArray`.
    ///
    /// - Returns: A new `NDArray` representing the minimum value.
    /// - Throws: `NDArrayError.operationError` if the operation fails.
    public func min(axis: Int? = nil) -> NDArray {
        let label = "\(self.label ?? "NDArray").min()"
        
        return NDArray(
            shape: [], // Scalar result
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`min` operation expects exactly 1 input.")
            }
            if let axis {
                return graph.reductionMinimum(with: inputs[0], axis: axis, name: nodeLabel)
            }
            return graph.reductionMinimum(with: inputs[0], axes: nil, name: nodeLabel)
        }
    }
    
    /// Computes the index of the maximum value along the specified axis.
    ///
    /// - Parameter axis: The axis along which to compute the `argmax`.
    /// - Returns: A new `NDArray` representing the indices of the maximum values.
    /// - Throws: `NDArrayError.operationError` if the operation fails.
    public func argmax(axis: Int) -> NDArray {
        let label = "\(self.label ?? "NDArray").argmax(axis: \(axis))"
        
        return NDArray(
            shape: self.shape.filter { $0 != axis }, // Reduced shape
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`argmax` operation expects exactly 1 input.")
            }
            return graph.reductionArgMaximum(with: inputs[0], axis: axis, name: nodeLabel)
        }
    }
    
    /// Computes the index of the minimum value along the specified axis.
    ///
    /// - Parameter axis: The axis along which to compute the `argmin`.
    /// - Returns: A new `NDArray` representing the indices of the minimum values.
    /// - Throws: `NDArrayError.operationError` if the operation fails.
    public func argmin(axis: Int) -> NDArray {
        let label = "\(self.label ?? "NDArray").argmin(axis: \(axis))"
        
        return NDArray(
            shape: self.shape.filter { $0 != axis }, // Reduced shape
            label: label,
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("`argmin` operation expects exactly 1 input.")
            }
            return graph.reductionArgMinimum(with: inputs[0], axis: axis, name: nodeLabel)
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


extension NDArray {
    /// Creates an NDArray that is the one-hot encoding of `indices`.
    /// - Parameters:
    ///   - indices: NDArray of integer indices, e.g., shape [N].
    ///   - depth: The size of the one-hot dimension (number of classes).
    ///   - axis: The position at which to insert the new dimension.
    ///     - For example, -1 means append it at the end.
    ///   - dataType: The output MPSDataType (usually .float32).
    ///   - label: Optional name/label.
    /// - Returns: A *lazy* NDArray that will become `[N, depth]` if `axis == -1`.
    public static func oneHot(
        indices: NDArray,
        depth: Int,
        axis: Int = -1,
        dataType: MPSDataType = .float32,
        label: String? = nil
    ) -> NDArray {
        // Determine new shape by inserting 'depth' at the specified axis
        var newShape = indices.shape
        let rank = newShape.count
        let insertAxis = axis >= 0 ? axis : rank + axis + 1
        precondition(insertAxis >= 0 && insertAxis <= rank, "Axis \(axis) out of bounds for shape \(indices.shape)")
        newShape.insert(depth, at: insertAxis)
        
        return NDArray(
            shape: newShape,
            label: label ?? "oneHot",
            parents: [indices],
            op: { graph, inputs, nodeLabel in
                guard inputs.count == 1 else {
                    throw NDArrayError.operationError("oneHot op expects exactly 1 input.")
                }
                
                let indicesTensor = inputs[0]
                
                // Assuming indices are stored as Int32. If not, cast them.
                // Here, we add a cast if necessary.
                let castedIndices: MPSGraphTensor
                if indicesTensor.dataType != .int32 {
                    castedIndices = graph.cast(
                        indicesTensor,
                        to: .int32,
                        name: "\(nodeLabel ?? "oneHot")_cast"
                    )
                } else {
                    castedIndices = indicesTensor
                }
                
                // Create one-hot tensor
                let oneHotTensor = graph.oneHot(
                    withIndicesTensor: castedIndices,
                    depth: depth,
                    axis: axis,
                    dataType: dataType,
                    name: nodeLabel
                )
                
                return oneHotTensor
            }
        )
    }
}


extension NDArray {
    /// Casts the NDArray to a different data type.
    /// - Parameter dataType: The target MPSDataType.
    /// - Returns: A new NDArray with the desired data type.
    public func cast(to dataType: MPSDataType, label: String? = nil) -> NDArray {
        return NDArray(
            shape: self.shape,
            label: label ?? "cast(\(self.label ?? "?"))",
            parents: [self]
        ) { graph, inputs, nodeLabel in
            guard inputs.count == 1 else {
                throw NDArrayError.operationError("cast op expects exactly 1 input.")
            }
            return graph.cast(
                inputs[0],
                to: dataType,
                name: nodeLabel
            )
        }
    }
}
