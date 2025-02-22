# Alloy

Alloy is an array framework for deep learning on Metal. Alloy is built with Swift and Metal Performance Shader Graph to provide a ease to use, type-safe machine learning with best in class performance on Apple devices that suppot MPSGraph.

>[!Caution]
>This framework is in Pre-Alpha stage of development, only suite for research. API WILL change. And production not recommended.

```Swift
import Alloy
import AlloyDatasets

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
do {
    print("Load MNIST")
    // Load the MNIST dataset
    let mnist = try MNIST()
    
    print("Getting Training Batch")
    // For simplicity, fetch entire train & test as one big batch.
    // Real training typically goes in small mini-batches.
    let (trainX, trainY) = try mnist.getTrainingBatch()   // e.g., [60000 * 28 * 28], [60000]
    print("Getting Testing Batch")
    let (testX, testY)   = try mnist.getTestingBatch()    // e.g., [10000 * 28 * 28], [10000]
    
    // Let's train for 1 epoch with batchSize=128 just to demonstrate:
    let batchSize = 128
    let epochs = 1
    let lr: Float = 0.01
    
    print("Main trainY Shape: ", trainY.shape)
    
    print("Start training LeNet")
    try trainLeNet(
        trainX: trainX,
        trainY: trainY,
        testX: testX,
        testY: testY,
        batchSize: batchSize,
        epochs: epochs,
        learningRate: lr
    )
    
} catch {
    print("MNIST/LeNet Error: \(error)")
}
```
