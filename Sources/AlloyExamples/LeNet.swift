//
//  LeNet.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/15/25.
//

import Alloy
import AlloyDatasets
import AlloyRandom
import MetalPerformanceShadersGraph
import Metal
import Foundation

public func accuracy(logits: NDArray, trueLabels: [Int]) throws -> Float {
    // 1) Argmax on GPU side -> NDArray of shape [N]
    let predIndices = logits.argmax(axis: 1)
    
    // 2) Evaluate the DAG to get the predicted indices on CPU
    try run(predIndices)
    let predArray = try predIndices.toArray().map { Int($0) } // [Float] → [Int]
    
    // 3) Compare with `trueLabels`
    guard predArray.count == trueLabels.count else {
        throw NDArrayError.operationError(
            "Mismatch in predicted count (\(predArray.count)) vs trueLabels count (\(trueLabels.count))."
        )
    }
    
    var correct = 0
    for i in 0..<trueLabels.count {
        if predArray[i] == trueLabels[i] {
            correct += 1
        }
    }
    return Float(correct) / Float(trueLabels.count)
}

/// Returns a dictionary of trainable parameters for LeNet.
public func buildLeNet() -> [String: NDArray] {
    // 1) Conv1 weights & bias: [6, 1, 5, 5]
    let conv1_w = NDArray.randn(shape: [6, 1, 5, 5], std: 0.1, label: "conv1_w")
    let conv1_b = NDArray.randn(shape: [6], std: 0.1, label: "conv1_b")
    
    // 2) Conv2 weights & bias: [16, 6, 5, 5]
    let conv2_w = NDArray.randn(shape: [16, 6, 5, 5], std: 0.1, label: "conv2_w")
    let conv2_b = NDArray.randn(shape: [16], std: 0.1, label: "conv2_b")
    
    // 3) FC1: input dim=16*5*5=400, output=120
    let fc1_w = NDArray.randn(shape: [120, 400], std: 0.1, label: "fc1_w")
    let fc1_b = NDArray.randn(shape: [120], std: 0.1, label: "fc1_b")
    
    // 4) FC2: 120 -> 84
    let fc2_w = NDArray.randn(shape: [84, 120], std: 0.1, label: "fc2_w")
    let fc2_b = NDArray.randn(shape: [84], std: 0.1, label: "fc2_b")
    
    // 5) FC3: 84 -> 10
    let fc3_w = NDArray.randn(shape: [10, 84], std: 0.1, label: "fc3_w")
    let fc3_b = NDArray.randn(shape: [10], std: 0.1, label: "fc3_b")
    
    return [
        "conv1_w": conv1_w, "conv1_b": conv1_b,
        "conv2_w": conv2_w, "conv2_b": conv2_b,
        "fc1_w": fc1_w,     "fc1_b": fc1_b,
        "fc2_w": fc2_w,     "fc2_b": fc2_b,
        "fc3_w": fc3_w,     "fc3_b": fc3_b,
    ]
}

/// Forward pass of LeNet. Expects input shape: [N, 1, 28, 28].
public func lenetForward(_ x: NDArray, _ p: [String: NDArray]) throws -> NDArray {
    print("lenetForward input shape:", x.shape)
    // Conv1 + relu + pool
    let c1 = try conv2d(
        input: x,
        weights: p["conv1_w"]!,
        bias: p["conv1_b"]!,
        stride: (1,1),
        padding: (2,2,2,2), // pad left=2,right=2,top=2,bottom=2
        dilation: (1,1),
        groups: 1,
        label: "conv1"
    )
    let r1 = relu(c1, label: "relu1")
    let p1 = maxPool2d(r1, kernelSize: (2,2), stride: (2,2), label: "pool1")
    
    // Conv2 + relu + pool
    let c2 = try conv2d(
        input: p1,
        weights: p["conv2_w"]!,
        bias: p["conv2_b"]!,
        stride: (1,1),
        padding: (0,0,0,0),
        dilation: (1,1),
        groups: 1,
        label: "conv2"
    )
    let r2 = relu(c2, label: "relu2")
    let p2 = maxPool2d(r2, kernelSize: (2,2), stride: (2,2), label: "pool2")
    
    // Flatten
    let f = flatten(p2)
    
    // FC1 + relu
    let fc1 = try linear(f, weight: p["fc1_w"]!, bias: p["fc1_b"]!)
    let r3 = relu(fc1, label: "relu3")
    
    // FC2 + relu
    let fc2 = try linear(r3, weight: p["fc2_w"]!, bias: p["fc2_b"]!)
    let r4 = relu(fc2, label: "relu4")
    
    // FC3 => logits
    let fc3 = try linear(r4, weight: p["fc3_w"]!, bias: p["fc3_b"]!)
    // No activation => these are final logits
    return fc3
}

public func trainLeNet(
    trainX: NDArray,
    trainY: NDArray,
    testX: NDArray,
    testY: NDArray,
    batchSize: Int,
    epochs: Int,
    learningRate: Float
) throws {
    // 1) Build parameter dictionary
    let params = buildLeNet() // Make it mutable if needed for updates
    
    // 2) Determine the number of training samples
    let trainCount = trainY.shape[0]
    let stepsPerEpoch = trainCount / batchSize
    
    // 3) Iterate over epochs
    for epoch in 1...epochs {
        var avgLoss: Float = 0
        
        // Optionally shuffle the training data at the start of each epoch
        // Implement shuffling if your framework supports it
        // Example: (trainX, trainY) = shuffle(trainX, trainY)
        
        // 4) Iterate over mini-batches
        for step in 0..<stepsPerEpoch {
            let startIdx = step * batchSize
            let endIdx = min(startIdx + batchSize, trainCount)
            
            // 5) Slice out the mini-batch from trainX and trainY
            let xBatch = try slice(trainX, start: [startIdx, 0, 0, 0], end: [endIdx, trainX.shape[1], trainX.shape[2], trainX.shape[3]])
            let yBatch = try slice(trainY, start: [startIdx], end: [endIdx])
            
            // 6) Convert labels to one-hot encoding using MPSGraph's oneHot
            let oneHotLabels = NDArray.oneHot(
                indices: yBatch,
                depth: 10,
                axis: -1,
                dataType: .float32,
                label: "oneHotLabels"
            )
            
            // 7) Forward pass
            let logits = try lenetForward(xBatch, params)
            
            // 8) Compute loss
            let loss = try softmaxCrossEntropy(logits: logits, labels: oneHotLabels)
            
            // 9) Perform SGD update
            // Assuming SGD updates params in-place or returns updated params
            try SGD(loss: loss, params: params.values.map { $0 }, learningRate: learningRate)
            
            // 10) Optionally fetch the float loss for logging
            try run(loss) // Evaluate the loss tensor
            let lossVal = try loss.toArray().first ?? 0
            avgLoss += lossVal
        }
        
        avgLoss /= Float(stepsPerEpoch)
        
        // 11) Evaluate on the test set
        let testLogits = try lenetForward(testX, params)
        let testAcc = try accuracy(logits: testLogits, trueLabels: try testY.toArray().map { Int($0) })
        
        print("Epoch \(epoch), loss=\(avgLoss), testAcc=\(testAcc)")
    }
}
