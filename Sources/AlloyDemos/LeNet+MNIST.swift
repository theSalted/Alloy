//
//  LeNet+MNIST.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/25/25.
//

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

func testConv2DSingleChannel3x3() {
    // A single image (N=1), single channel (C=1), height=3, width=3
    // Values laid out in row-major order: 1, 2, 3, 4, 5, 6, 7, 8, 9
    let inputValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    ]
    let input = NDArray(inputValues, shape: [1, 1, 3, 3], label: "input")
    
    // A single output channel (C_out=1), single input channel (C_in=1),
    // kernel height=3, kernel width=3.
    // We'll pick a kernel pattern that’s easy to verify:
    //   1 2 1
    //   2 4 2
    //   1 2 1
    let kernelValues: [Float] = [
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    ]
    let kernel = NDArray(kernelValues, shape: [1, 1, 3, 3], label: "kernel")
    
    // Expected shape => [1, 1, (3 - 3 + 1), (3 - 3 + 1)] = [1, 1, 1, 1]
    // Let's compute the expected value by hand:
    //
    // Convolution is sum of elementwise products:
    //   1*1 + 2*2 + 3*1 +
    //   4*2 + 5*4 + 6*2 +
    //   7*1 + 8*2 + 9*1
    //
    // = 1 + 4 + 3 + 8 + 20 + 12 + 7 + 16 + 9 = 80
    let _: [Float] = [80.0]
    
    do {
        // Perform the 2D convolution with no padding, stride=1, etc.
        let output = try conv2d(
            input: input,
            weights: kernel,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            label: "conv2d_3x3_singleChannel"
        )
        
        // Run the graph to materialize `output.data`.
        try run(output)
        
        // Verify the shape and values
        
        
    } catch {
        print("conv2d test threw an error: \(error)")
    }
}


