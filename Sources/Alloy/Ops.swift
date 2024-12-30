//
//  Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/29/24.
//

public func reshape(
    _ array: NDArray,
    to shape: [Int]) -> NDArray {
    let label = "reshaped to \(shape)"
    return NDArray(
        shape: shape,
        label: label,
        parents: [array]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("`reshaped` expects exactly 1 input.")
        }
        return graph.reshape(inputs[0], shape: shape.toNSNumberArray(), name: nodeLabel)
    }
}

public func broadcast(
    _ array: NDArray,
    to shape: [Int]
) -> NDArray {
    let label = "broadcasted to \(shape)"
    return NDArray(
        shape: shape,
        label: label,
        parents: [array]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("`broadcasted` expects exactly 1 input.")
        }
        return graph.broadcast(inputs[0], shape: shape.toNSNumberArray(), name: nodeLabel)
    }
}
