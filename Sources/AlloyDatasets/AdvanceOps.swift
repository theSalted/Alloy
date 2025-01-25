//
//  AdvanceOps.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/25/25.
//

import Alloy
import MetalPerformanceShadersGraph

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
    /**
     Expects:
       - `input` in NHWC shape:    [N, H_in, W_in, C_in]
       - `weights` in NHWC shape:  [K_h, K_w, C_in/groups, C_out]
       - `bias` in shape:          [C_out]
     Produces output in NHWC:      [N, H_out, W_out, C_out]
     */
    
    // 1) Validate shapes & rank
    guard input.shape.count == 4 else {
        throw NDArrayError.operationError(
            "`conv2d` expects input with 4D shape [N, H_in, W_in, C_in], got \(input.shape)."
        )
    }
    guard weights.shape.count == 4 else {
        throw NDArrayError.operationError(
            "`conv2d` expects 4D weights [K_h, K_w, C_in/groups, C_out], got \(weights.shape)."
        )
    }
    if let bias = bias {
        guard bias.shape.count == 1 else {
            throw NDArrayError.operationError(
                "`bias` must be 1D, got shape \(bias.shape)."
            )
        }
    }
    
    // 2) Parse input shape (NHWC)
    let N = input.shape[0]
    let H_in = input.shape[1]
    let W_in = input.shape[2]
    let C_in = input.shape[3]
    
    // 3) Parse weight shape (NHWC => [K_h, K_w, C_inOverGroups, C_out])
    let K_h = weights.shape[0]
    let K_w = weights.shape[1]
    _ = weights.shape[2]
    let C_out = weights.shape[3]
    
    // If groups > 1, you might check cInOverGroups == C_in/groups here
    
    // 4) Validate bias shape
    if let bias = bias {
        guard bias.shape[0] == C_out else {
            throw NDArrayError.operationError(
                "`bias` must have shape [\(C_out)], got \(bias.shape)."
            )
        }
    }
    
    // 5) Unpack stride/padding/dilation
    let (strideH, strideW) = stride
    let (padLeft, padRight, padTop, padBottom) = padding
    let (dilationH, dilationW) = dilation
    
    // 6) Compute output spatial dims
    let H_out = ((H_in + padTop + padBottom - dilationH * (K_h - 1) - 1) / strideH) + 1
    let W_out = ((W_in + padLeft + padRight - dilationW * (K_w - 1) - 1) / strideW) + 1
    
    // Ensure H_out, W_out are valid
    if H_out <= 0 || W_out <= 0 {
        throw NDArrayError.operationError(
            "Invalid H_out=\(H_out), W_out=\(W_out). Check kernel/stride/padding/dilation."
        )
    }
    
    // 7) Build the output shape [N, H_out, W_out, C_out]
    let outputShape = [N, H_out, W_out, C_out]
    
    // 8) Create label if not provided
    let nodeLabel = label ?? "conv2d(\(C_out)x\(C_in)/\(groups) kernel=\(K_h)x\(K_w))"
    
    // 9) Create MPSGraphConvolution2DOpDescriptor
    let convDesc = MPSGraphConvolution2DOpDescriptor()
    convDesc.strideInX         = strideW
    convDesc.strideInY         = strideH
    convDesc.dilationRateInX   = dilationW
    convDesc.dilationRateInY   = dilationH
    convDesc.groups            = groups
    convDesc.paddingLeft       = padLeft
    convDesc.paddingRight      = padRight
    convDesc.paddingTop        = padTop
    convDesc.paddingBottom     = padBottom
    convDesc.paddingStyle      = .explicit
    convDesc.dataLayout        = .NHWC
    convDesc.weightsLayout     = .NHWC
    
    // 10) Build the NDArray for the graph
    var parents: [NDArray] = [input, weights]
    if let bias = bias {
        parents.append(bias)
    }
    
    let convNDArray = NDArray(
        shape: outputShape,
        label: nodeLabel,
        parents: parents
    ) { graph, inputs, opLabel in
        guard inputs.count == (bias == nil ? 2 : 3) else {
            throw NDArrayError.operationError(
                "`conv2d` expects \(bias == nil ? "2" : "3") inputs."
            )
        }
        
        let inputTensor  = inputs[0]
        let weightTensor = inputs[1]
        
        // Convolution
        var outputTensor = graph.convolution2D(
            inputTensor,
            weights: weightTensor,
            descriptor: convDesc,
            name: opLabel
        )
        
        // Add bias if present
        if bias != nil {
            let biasTensor = inputs[2]
            // Reshape bias [C_out] => [1, 1, 1, C_out] in NHWC
            let biasReshaped = graph.reshape(
                biasTensor,
                shape: [1, 1, 1, NSNumber(value: C_out)],
                name: "\(opLabel ?? "conv2d")_bias_reshaped"
            )
            outputTensor = graph.addition(
                outputTensor,
                biasReshaped,
                name: "\(opLabel ?? "conv2d")_bias_addition"
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
    let (kernelH, kernelW) = kernelSize
    let (padLeft, padRight, padTop, padBottom) = padding
    let strideH = stride?.0 ?? kernelH
    let strideW = stride?.1 ?? kernelW

    // For NHWC:
    let N = input.shape[0]
    let H_in = input.shape[1]
    let W_in = input.shape[2]
    let C_in = input.shape[3]

    let H_out = ((H_in + padTop + padBottom - kernelH) / strideH) + 1
    let W_out = ((W_in + padLeft + padRight - kernelW) / strideW) + 1

    let outShape = [N, H_out, W_out, C_in]

    return NDArray(
        shape: outShape,
        label: label ?? "maxPool2d(\(kernelSize))",
        parents: [input]
    ) { graph, inputs, opLabel in
        let poolDesc = MPSGraphPooling2DOpDescriptor()
        poolDesc.kernelHeight = kernelH
        poolDesc.kernelWidth  = kernelW
        poolDesc.strideInY = strideH
        poolDesc.strideInX = strideW
        poolDesc.paddingLeft   = padLeft
        poolDesc.paddingRight  = padRight
        poolDesc.paddingTop    = padTop
        poolDesc.paddingBottom = padBottom
        poolDesc.paddingStyle  = .TF_VALID
        poolDesc.dilationRateInX = 1
        poolDesc.dilationRateInY = 1

        // **NHWC** layout:
        poolDesc.dataLayout = .NHWC

        return graph.maxPooling2D(
            withSourceTensor: inputs[0],
            descriptor: poolDesc,
            name: opLabel
        )
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
    let (N, H, W, C) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return x.reshaped([N, H*W*C])
}

