//
//  NDArray+Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/27/24.
//

// MARK: - Operator Overloads (e.g. +)

extension NDArray {
    /// Simple binary `+` operator for demonstration.
    /// In production, you might do shape inference or broadcasting here.
    public static func +(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // For simplicity, assume shapes match (no broadcasting).
        guard lhs.shape == rhs.shape else {
            fatalError("`+` operator: shapes do not match and no broadcasting is implemented.")
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
}
