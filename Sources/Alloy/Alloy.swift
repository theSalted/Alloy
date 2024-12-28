//
//  Alloy.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/2/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders
import Metal
import MetalKit

open class NeuralNetwork {
    public var device: MTLDevice
    public var commandQueue: MTLCommandQueue
    public var graph: MPSGraph
    
    public init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        self.graph = MPSGraph()
    }
    
    
}

/*
extension NDArray: CustomDebugStringConvertible {
    public var debugDescription: String {
        ""
    }
}
 */
