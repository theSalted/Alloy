//
//  Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/29/24.
//

import MetalPerformanceShadersGraph

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


public func conv2d(
    input: NDArray,
    weights: NDArray,
    bias: NDArray? = nil,
    stride: (Int, Int) = (1, 1),
    padding: (Int, Int, Int, Int) = (0, 0, 0, 0), // (left, right, top, bottom)
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    label: String? = nil
) throws -> NDArray {
    // Validate input dimensions
    guard input.shape.count == 4 else {
        throw NDArrayError.operationError("`conv2d` expects input with 4 dimensions [N, C_in, H, W].")
    }
    
    guard weights.shape.count == 4 else {
        throw NDArrayError.operationError("`conv2d` expects weights with 4 dimensions [C_out, C_in/groups, K_h, K_w].")
    }
    
    if let bias = bias {
        guard bias.shape.count == 1, bias.shape[0] == weights.shape[0] else {
            throw NDArrayError.operationError("`bias` must have shape [C_out].")
        }
    }
    
    // Calculate output shape
    let N = input.shape[0]
    let C_out = weights.shape[0]
    let K_h = weights.shape[2]
    let K_w = weights.shape[3]
    let H_in = input.shape[2]
    let W_in = input.shape[3]
    
    let strideH = stride.0
    let strideW = stride.1
    let dilationH = dilation.0
    let dilationW = dilation.1
    
    let paddingLeft = padding.0
    let paddingRight = padding.1
    let paddingTop = padding.2
    let paddingBottom = padding.3
    
    let H_out = ((H_in + paddingTop + paddingBottom - dilationH * (K_h - 1) - 1) / strideH) + 1
    let W_out = ((W_in + paddingLeft + paddingRight - dilationW * (K_w - 1) - 1) / strideW) + 1
    
    // Define the output shape
    let outputShape = [N, C_out, H_out, W_out]
    
    // Create label if not provided
    let nodeLabel = label ?? "conv2d(\(C_out)x\(input.shape[1])/\(groups) kernel=\(K_h)x\(K_w))"
    
    // Create the convolution descriptor
    let convDesc = MPSGraphConvolution2DOpDescriptor()
    convDesc.strideInX = strideW
    convDesc.strideInY = strideH
    convDesc.dilationRateInX = dilationW
    convDesc.dilationRateInY = dilationH
    convDesc.groups = groups
    convDesc.paddingLeft = paddingLeft
    convDesc.paddingRight = paddingRight
    convDesc.paddingTop = paddingTop
    convDesc.paddingBottom = paddingBottom
    convDesc.paddingStyle = .explicit // Since we're specifying padding manually
    
    // Determine data and weight layouts
    // Assuming NCHW layout
    convDesc.dataLayout = .NCHW
    convDesc.weightsLayout = .NCHW
    
    // Parents: input, weights, (optional) bias
    var parents: [NDArray] = [input, weights]
    if let bias = bias {
        parents.append(bias)
    }
    
    // Create the NDArray node
    let convNDArray = NDArray(
        shape: outputShape,
        label: nodeLabel,
        parents: parents
    ) { graph, inputs, nodeLabel in
        guard inputs.count == (bias == nil ? 2 : 3) else {
            throw NDArrayError.operationError("`conv2d` expects \(bias == nil ? "2" : "3") inputs.")
        }
        
        let inputTensor = inputs[0]
        let weightTensor = inputs[1]
        var outputTensor = graph.convolution2D(
            inputTensor,
            weights: weightTensor,
            descriptor: convDesc,
            name: nodeLabel
        )
        
        if bias != nil {
            let biasReshaped = graph.reshape(
                inputs[2],
                shape: [1, C_out, 1, 1].toNSNumberArray(),
                name: "\(nodeLabel ?? "conv2d")_bias_reshaped"
            )
            outputTensor = graph.addition(
                outputTensor,
                biasReshaped,
                name: "\(nodeLabel ?? "conv2d")_bias_addition"
            )
        }
        
        return outputTensor
    }
    
    return convNDArray
}
