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
import AlloyRandom

/// Returns a dictionary of trainable parameters for LeNet, in NHWC layout.
public func buildLeNet() -> [String: NDArray] {
    // For MNIST: input is [N, 28, 28, 1] => kernel: [5, 5, inC, outC]
    let conv1_w = NDArray.randn(shape: [5, 5, 1, 6], std: 0.1, label: "conv1_w")
    let conv1_b = NDArray.randn(shape: [6], std: 0.1, label: "conv1_b")
    
    // The output of conv1 is [N, 28, 28, 6].
    // Then we typically do pool => [N, 14, 14, 6].
    //
    // Next conv: [5, 5, 6, 16] => out shape [N, 14, 14, 16] (if padding=2).
    // But if we want the classic LeNet shape transitions, we might do no padding:
    // so shape after conv2 would be [N, 10, 10, 16] if stride=1 and no padding,
    // then pool => [N, 5, 5, 16].
    let conv2_w = NDArray.randn(shape: [5, 5, 6, 16], std: 0.1, label: "conv2_w")
    let conv2_b = NDArray.randn(shape: [16], std: 0.1, label: "conv2_b")
    
    // Flatten => shape [N, 5*5*16] = [N, 400].
    // Then fully-connected 1 => out=120
    let fc1_w = NDArray.randn(shape: [120, 400], std: 0.1, label: "fc1_w")
    let fc1_b = NDArray.randn(shape: [120], std: 0.1, label: "fc1_b")
    
    // Fully-connected 2 => out=84
    let fc2_w = NDArray.randn(shape: [84, 120], std: 0.1, label: "fc2_w")
    let fc2_b = NDArray.randn(shape: [84], std: 0.1, label: "fc2_b")
    
    // Fully-connected 3 => out=10 (final logits)
    let fc3_w = NDArray.randn(shape: [10, 84], std: 0.1, label: "fc3_w")
    let fc3_b = NDArray.randn(shape: [10], std: 0.1, label: "fc3_b")
    
    return [
        "conv1_w": conv1_w, "conv1_b": conv1_b,
        "conv2_w": conv2_w, "conv2_b": conv2_b,
        "fc1_w":   fc1_w,   "fc1_b":   fc1_b,
        "fc2_w":   fc2_w,   "fc2_b":   fc2_b,
        "fc3_w":   fc3_w,   "fc3_b":   fc3_b,
    ]
}

/// Forward pass of LeNet with NHWC shapes.
/// - Parameter x: The input NDArray, shape [N, 28, 28, 1].
/// - Parameter p: A dictionary of parameters from `buildLeNet()`.
/// - Returns: The logits of shape [N, 10].
public func lenetForward(_ x: NDArray, _ p: [String: NDArray]) throws -> NDArray {
    print("lenetForward input shape:", x.shape)
    // Should be [N, 28, 28, 1]
    
    // 1) Conv1: kernel=5x5, padding=2 => output is [N, 28, 28, 6]
    let c1 = try conv2d(
        input: x,
        weights: p["conv1_w"]!,
        bias:    p["conv1_b"]!,
        stride: (1,1),
        padding: (2,2,2,2), // left=2, right=2, top=2, bottom=2
        label:   "conv1"
    )
    // ReLU => same shape
    let r1 = relu(c1, label: "relu1")
    print("r1 shape:", r1.shape) // expect [N, 28, 28, 6]
    
    // 2) Pool => typical 2×2 max pool => [N, 14, 14, 6]
    let p1 = maxPool2d(r1, kernelSize: (2,2), stride: (2,2), label: "pool1")
    print("p1 shape:", p1.shape) // expect [N, 14, 14, 6]
    
    // 3) Conv2: kernel=5×5, no padding => output [N, 10, 10, 16]
    let c2 = try conv2d(
        input: p1,
        weights: p["conv2_w"]!,
        bias:    p["conv2_b"]!,
        stride: (1,1),
        // If you want no padding, do (0,0,0,0):
        padding: (0,0,0,0),
        label:   "conv2"
    )
    let r2 = relu(c2, label: "relu2")
    print("r2 shape:", r2.shape) // expect [N, 10, 10, 16]
    
    // 4) Pool => 2×2 => [N, 5, 5, 16]
    let p2 = maxPool2d(r2, kernelSize: (2,2), stride: (2,2), label: "pool2")
    print("p2 shape:", p2.shape) // expect [N, 5, 5, 16]
    
    // 5) Flatten => [N, 5*5*16] = [N, 400]
    let f = flatten(p2)
    print("f shape:", f.shape) // [N, 400]
    
    // 6) FC1 => [N, 120]
    let fc1 = try linear(f, weight: p["fc1_w"]!, bias: p["fc1_b"]!)
    let r3 = relu(fc1, label: "relu3")
    print("r3 shape:", r3.shape) // [N, 120]
    
    // 7) FC2 => [N, 84]
    let fc2 = try linear(r3, weight: p["fc2_w"]!, bias: p["fc2_b"]!)
    let r4 = relu(fc2, label: "relu4")
    print("r4 shape:", r4.shape) // [N, 84]
    
    // 8) FC3 => [N, 10] (logits)
    let fc3 = try linear(r4, weight: p["fc3_w"]!, bias: p["fc3_b"]!)
    print("fc3 shape:", fc3.shape) // [N, 10]
    
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
    
    print("Start training")
    // 3) Iterate over epochs
    for epoch in 1...epochs {
        var avgLoss: Float = 0
        print("Epoch: \(epoch)")
        
        // Optionally shuffle the training data at the start of each epoch
        // Implement shuffling if your framework supports it
        // Example: (trainX, trainY) = shuffle(trainX, trainY)
        
        // 4) Iterate over mini-batches
        for step in 0..<stepsPerEpoch {
            
            print("Mini-batch: \(step)")
            let startIdx = step * batchSize
            let endIdx = min(startIdx + batchSize, trainCount)
            
            // 5) Slice out the mini-batch from trainX and trainY
            let xBatch = try slice(trainX, start: [startIdx, 0, 0, 0], end: [endIdx, trainX.shape[1], trainX.shape[2], trainX.shape[3]])
            print("LeNet trainY shape: ", trainY.shape)
            let yBatch = try slice(trainY,
                                   start: [startIdx, 0],
                                   end:   [endIdx,   10])
            
            // 6) Convert labels to one-hot encoding using MPSGraph's oneHot
//            let oneHotLabels = NDArray.oneHot(
//                indices: yBatch,
//                depth: 10,
//                axis: -1,
//                dataType: .float32,
//                label: "oneHotLabels"
//            )
            
            let oneHotLabels = yBatch
            
            // 7) Forward pass
            let logits = try lenetForward(xBatch, params)
            
            // 8) Compute loss
            print("Compute Loss")
            let loss = try softmaxCrossEntropy(logits: logits, labels: oneHotLabels)
            
            // 9) Perform SGD update
            // Assuming SGD updates params in-place or returns updated params
            print("Perform SGD update")
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
