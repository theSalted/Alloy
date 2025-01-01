//
//  Alloy.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/2/24.
//

import MetalPerformanceShadersGraph
import MetalPerformanceShaders
@preconcurrency import Metal
import MetalKit

public struct Alloy: Sendable {
    public static let shared = Alloy()
    
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    
    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device found") }
        guard let commandQueue = device.makeCommandQueue() else { fatalError("No command queue found") }
        self.device = device
        self.commandQueue = commandQueue
    }
}
