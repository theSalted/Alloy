//
//  NDArray+Arithmetic.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/11/24.
//

import Metal

extension NDArray {
    // Overloaded operators with deferred computation
    public static func +(lhs: NDArray, rhs: NDArray) -> NDArray {
        let result = NDArray(shape: lhs.shape, device: lhs.device)
        result.inputs = [lhs, rhs]
        
        // Deferred operation using custom Metal shader for element-wise addition
        result.operation = { commandBuffer in
            guard let function = lhs.getFunction(name: "elementwise_add") else { return }
            let pipelineState = try! lhs.device.makeComputePipelineState(function: function)
            let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            commandEncoder.setComputePipelineState(pipelineState)
            commandEncoder.setBuffer(lhs.buffer, offset: 0, index: 0)
            commandEncoder.setBuffer(rhs.buffer, offset: 0, index: 1)
            commandEncoder.setBuffer(result.buffer, offset: 0, index: 2)
            
            let vectorLength = lhs.shape.reduce(1, *)
            let threadsPerThreadgroup = MTLSize(width: min(256, vectorLength), height: 1, depth: 1)
            let threadgroupsPerGrid = MTLSize(width: (vectorLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            commandEncoder.endEncoding()
        }
        
        // Define backward function
        result._backward = {
            let commandBuffer = lhs.commandQueue.makeCommandBuffer()!
            if let function = lhs.getFunction(name: "elementwise_add_grad") {
                let pipelineState = try! lhs.device.makeComputePipelineState(function: function)
                let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
                commandEncoder.setComputePipelineState(pipelineState)
                
                // Set buffers
                commandEncoder.setBuffer(result.gradientBuffer, offset: 0, index: 0)
                commandEncoder.setBuffer(lhs.gradientBuffer, offset: 0, index: 1)
                commandEncoder.setBuffer(rhs.gradientBuffer, offset: 0, index: 2)
                
                let vectorLength = lhs.shape.reduce(1, *)
                let threadsPerThreadgroup = MTLSize(width: min(256, vectorLength), height: 1, depth: 1)
                let threadgroupsPerGrid = MTLSize(width: (vectorLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
                commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                commandEncoder.endEncoding()
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        return result
    }
    
    public static func *(lhs: NDArray, rhs: NDArray) -> NDArray {
        let result = NDArray(shape: lhs.shape, device: lhs.device)
        result.inputs = [lhs, rhs]
        
        // Deferred operation using custom Metal shader for element-wise multiplication
        result.operation = { commandBuffer in
            guard let function = lhs.getFunction(name: "elementwise_multiply") else { return }
            let pipelineState = try! lhs.device.makeComputePipelineState(function: function)
            let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            commandEncoder.setComputePipelineState(pipelineState)
            commandEncoder.setBuffer(lhs.buffer, offset: 0, index: 0)
            commandEncoder.setBuffer(rhs.buffer, offset: 0, index: 1)
            commandEncoder.setBuffer(result.buffer, offset: 0, index: 2)
            
            let vectorLength = lhs.shape.reduce(1, *)
            let threadsPerThreadgroup = MTLSize(width: min(256, vectorLength), height: 1, depth: 1)
            let threadgroupsPerGrid = MTLSize(width: (vectorLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            commandEncoder.endEncoding()
        }
        
        // Define backward function
        result._backward = {
            let commandBuffer = lhs.commandQueue.makeCommandBuffer()!
            
            // lhs.gradient += rhs.data * result.gradient
            if let function = lhs.getFunction(name: "elementwise_mul_grad_lhs") {
                let pipelineState = try! lhs.device.makeComputePipelineState(function: function)
                let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
                commandEncoder.setComputePipelineState(pipelineState)
                commandEncoder.setBuffer(rhs.buffer, offset: 0, index: 0)
                commandEncoder.setBuffer(result.gradientBuffer, offset: 0, index: 1)
                commandEncoder.setBuffer(lhs.gradientBuffer, offset: 0, index: 2)
                
                let vectorLength = lhs.shape.reduce(1, *)
                let threadsPerThreadgroup = MTLSize(width: min(256, vectorLength), height: 1, depth: 1)
                let threadgroupsPerGrid = MTLSize(width: (vectorLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
                commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                commandEncoder.endEncoding()
            }
            
            // rhs.gradient += lhs.data * result.gradient
            if let function = lhs.getFunction(name: "elementwise_mul_grad_rhs") {
                let pipelineState = try! lhs.device.makeComputePipelineState(function: function)
                let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
                commandEncoder.setComputePipelineState(pipelineState)
                commandEncoder.setBuffer(lhs.buffer, offset: 0, index: 0)
                commandEncoder.setBuffer(result.gradientBuffer, offset: 0, index: 1)
                commandEncoder.setBuffer(rhs.gradientBuffer, offset: 0, index: 2)
                
                let vectorLength = lhs.shape.reduce(1, *)
                let threadsPerThreadgroup = MTLSize(width: min(256, vectorLength), height: 1, depth: 1)
                let threadgroupsPerGrid = MTLSize(width: (vectorLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
                commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                commandEncoder.endEncoding()
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        return result
    }
}
