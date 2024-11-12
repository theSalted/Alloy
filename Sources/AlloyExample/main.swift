//
//  main.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/10/24.
//

import Foundation
import MetalPerformanceShaders
import Alloy

// Initialize Metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Create NDArrays
let dataA = [Float](repeating: 2.0, count: 1024)
let dataB = [Float](repeating: 3.0, count: 1024)
let a = NDArray(shape: [1024], device: device, data: dataA, label: "A")
let b = NDArray(shape: [1024], device: device, data: dataB, label: "B")

// Build computational graph
let c = a * b + a

// Perform forward computation
c.forward()

// Retrieve result
let resultData = c.evaluate()

// Perform backward computation
c.backward()

// Retrieve gradients
let gradA = a.gradient()
let gradB = b.gradient()

print("Result data:", resultData.prefix(5))
print("Gradient w.r.t A:", gradA.prefix(5))
print("Gradient w.r.t B:", gradB.prefix(5))
