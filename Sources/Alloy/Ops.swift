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

public func softmaxCrossEntropy(logits: NDArray, labels: NDArray) throws -> NDArray {
    guard logits.shape.count == 2 else {
        throw NDArrayError.operationError("Logits must be 2D, got \(logits.shape)")
    }
    guard labels.shape == logits.shape else {
        throw NDArrayError.operationError("Labels shape must match logits. Got \(labels.shape) vs \(logits.shape)")
    }
    
//    let classes = logits.shape[1]
//    let batchSize = logits.shape[0]
    
    // We'll produce a scalar NDArray of shape []
    return NDArray(
        shape: [],
        label: "softmaxCE(\(logits.label ?? "?"))",
        parents: [logits, labels]
    ) { graph, inputs, nodeLabel in
        guard inputs.count == 2 else {
            throw NDArrayError.operationError("softmaxCrossEntropy expects 2 inputs (logits, labels).")
        }
        let ce = graph.softMaxCrossEntropy(inputs[0], labels: inputs[1], axis: 1, reuctionType: .mean, name: nodeLabel)
        return ce
    }
}

public func maxPool2d(
    _ input: NDArray,
    kernelSize: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int, Int, Int) = (0,0,0,0),
    label: String? = nil
) -> NDArray {
    let strideH = stride?.0 ?? kernelSize.0
    let strideW = stride?.1 ?? kernelSize.1
    let (padLeft, padRight, padTop, padBottom) = padding
    
    // Compute output shape manually
    let N = input.shape[0]
    let C = input.shape[1]
    let H_in = input.shape[2]
    let W_in = input.shape[3]
    
    let H_out = ((H_in + padTop + padBottom - kernelSize.0) / strideH) + 1
    let W_out = ((W_in + padLeft + padRight - kernelSize.1) / strideW) + 1
    
    let outShape = [N, C, H_out, W_out]
    let nodeLabel = label ?? "maxPool2d(\(kernelSize))"
    
    return NDArray(
        shape: outShape,
        label: nodeLabel,
        parents: [input]
    ) { graph, inputs, opLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("maxPool2d expects 1 input.")
        }
        let poolDesc = MPSGraphPooling2DOpDescriptor()
        poolDesc.strideInX = strideW
        poolDesc.strideInY = strideH
        
//        poolDesc.windowWidth = kernelSize.1
//        poolDesc.windowHeight = kernelSize.0
        poolDesc.paddingLeft = padLeft
        poolDesc.paddingRight = padRight
        poolDesc.paddingTop = padTop
        poolDesc.paddingBottom = padBottom
        poolDesc.paddingStyle = .explicit
        poolDesc.dataLayout = .NCHW
        
        // MPSGraph has maxPooling2D
        let result = graph.maxPooling2D(withSourceTensor: inputs[0],
                                        descriptor: poolDesc,
                                        name: opLabel)
        return result
    }
}

public func linear(_ x: NDArray, weight: NDArray, bias: NDArray? = nil) throws -> NDArray {
    // x: [N, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features]
    // output => [N, out_features]
    
    if x.shape.count != 2 {
        throw NDArrayError.operationError("linear input must be 2D, got \(x.shape)")
    }
    let (N, inFeatures) = (x.shape[0], x.shape[1])
    let (outFeatures, wInFeatures) = (weight.shape[0], weight.shape[1])
    guard inFeatures == wInFeatures else {
        throw NDArrayError.operationError("Input size mismatch in linear. x: \(x.shape), w: \(weight.shape)")
    }
    if let bias = bias {
        guard bias.shape == [outFeatures] else {
            throw NDArrayError.operationError("Bias must be [\(outFeatures)], got \(bias.shape)")
        }
    }
    
    let outShape = [N, outFeatures]
    
    let nodeLabel = "linear(\(x.label ?? "?"))"
    var parents: [NDArray] = [x, weight]
    if let b = bias {
        parents.append(b)
    }
    
    return NDArray(
        shape: outShape,
        label: nodeLabel,
        parents: parents
    ) { graph, inputs, opLabel in
        guard inputs.count == (bias == nil ? 2 : 3) else {
            throw NDArrayError.operationError("linear expects \(bias == nil ? "2" : "3") inputs.")
        }
        let xTensor = inputs[0]
        let wTensor = inputs[1]
        
        // x: [N, inFeatures], w: [outFeatures, inFeatures]
        // MPSGraph matmul expects shapes: x: [N, K], w: [K, M] => out: [N, M]
        // We can do a transpose on w if needed.
        let wTransposed = graph.transposeTensor(wTensor, dimension: 0, withDimension: 1, name: "wT")
        
        var yTensor = graph.matrixMultiplication(primary: xTensor, secondary: wTransposed, name: opLabel)
        
        if bias != nil {
            let biasTensor = inputs[2]
            // reshape bias to [1, outFeatures]
            let reshapedBias = graph.reshape(biasTensor, shape: [1, NSNumber(value: outFeatures)], name: "bias_reshape")
            yTensor = graph.addition(yTensor, reshapedBias, name: "add_bias")
        }
        
        return yTensor
    }
}

public func relu(_ x: NDArray, label: String? = nil) -> NDArray {
    return NDArray(
        shape: x.shape,
        label: label ?? "relu(\(x.label ?? "?"))",
        parents: [x]
    ) { graph, inputs, opLabel in
        guard inputs.count == 1 else {
            throw NDArrayError.operationError("ReLU expects 1 input.")
        }
        return graph.reLU(with: inputs[0], name: opLabel)
    }
}

public func flatten(_ x: NDArray) -> NDArray {
    let (N, C, H, W) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return x.reshaped([N, C*H*W])
}

