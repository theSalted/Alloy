//
//  Alloy.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/2/24.
//

import MetalPerformanceShaders

func isMetalPerformanceShadersAvailable() -> Bool {
    #if targetEnvironment(simulator)
    // MetalPerformanceShaders is not available on iOS simulators
    return false
    #else
    if let device = MTLCreateSystemDefaultDevice() {
        #if os(iOS)
        return device.supportsFeatureSet(.iOS_GPUFamily2_v1)
        #elseif os(macOS)
        return device.supportsFamily(.mac2)
        #endif
    }
    return false
    #endif
}
