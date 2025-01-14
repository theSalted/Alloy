//
//  main.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/10/24.
//
import Alloy
import AlloyDatasets
import AlloyRandom

let a = NDArray([1, 1, 1, 1], shape: [2, 2])
let b = NDArray([2], shape: [1])

let c = a + b
let d = c * 4
let e = d / 12

do {
    try run(c, d, e)
    print("Success (c): \(c)")
    print("Success (d): \(d)")
    print("Success (e): \(e)")

} catch {
    print("Error: \(error)")
}

// MNIST
let mnist = try MNIST()
let (trainX, trainY) = try mnist.getTrainingBatch()
let (testX, testY) = try mnist.getTestingBatch()

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
    trainX: [Float],
    trainY: [Int],
    testX: [Float],
    testY: [Int],
    batchSize: Int,
    epochs: Int,
    learningRate: Float
) throws {
    // 1) Build param dictionary
    var params = buildLeNet()
    
    let trainCount = trainY.count
    let stepsPerEpoch = trainCount / batchSize
    
    for epoch in 1...epochs {
        var avgLoss: Float = 0
        
        for step in 0..<stepsPerEpoch {
            let startIdx = step * batchSize
            let endIdx = startIdx + batchSize
            
            // 2) Slice out the batch
            let xBatch = Array(trainX[(startIdx*28*28)..<(endIdx*28*28)])
            let yBatch = Array(trainY[startIdx..<endIdx])
            
            // 3) Wrap in NDArray
            let xNDArray = NDArray(xBatch, shape: [batchSize, 1, 28, 28], label: "input")
            
            let oneHotLabels = yBatch.toOneHot(classCount: 10)
            let yNDArray = NDArray(oneHotLabels, shape: [batchSize, 10], label: "labels")
            
            // 4) Forward pass
            let logits = try lenetForward(xNDArray, params)
            
            // 5) Loss
            let loss = try softmaxCrossEntropy(logits: logits, labels: yNDArray)
            
            // 6) Update
            
//            try sgdUpdate(loss: loss, params: &params, learningRate: learningRate)
            
            // 7) Optionally fetch the float loss for logging
            try run(loss) // to fill .data
            let lossVal = try loss.toArray().first ?? 0
            avgLoss += lossVal
        }
        
        avgLoss /= Float(stepsPerEpoch)
        
        // Evaluate on test batch or entire test set
        // For brevity, let's just do 1 batch:
        let testBatchSize = min(1000, testY.count) // e.g. 1000
        let testXSlice = Array(testX[0..<(testBatchSize * 28 * 28)])
        let testYSlice = Array(testY[0..<testBatchSize])
        
        let testInput = NDArray(testXSlice, shape: [testBatchSize, 1, 28, 28])
        let testLogits = try lenetForward(testInput, params)
        
        let testAcc = try accuracy(logits: testLogits, trueLabels: testYSlice)
        
        print("Epoch \(epoch), loss=\(avgLoss), testAcc=\(testAcc)")
    }
}

extension Array where Element == Int {
    /// Converts an array of integer labels into one-hot arrays.
    /// - Parameter classCount: Number of classes (e.g. 10 for MNIST).
    /// - Returns: A flat [Float] array containing the one-hot encodings.
    public func toOneHot(classCount: Int) -> [Float] {
        var result = [Float](repeating: 0, count: self.count * classCount)
        for (i, label) in self.enumerated() {
            // one-hot index
            result[i * classCount + label] = 1.0
        }
        return result
    }
}
