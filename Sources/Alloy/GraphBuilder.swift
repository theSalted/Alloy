//
//  GraphBuilder.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/28/24.
//

import MetalPerformanceShadersGraph
import Metal
import Foundation

/// A private utility that constructs (and optionally executes) an MPSGraph from an NDArray DAG.
struct GraphBuilder {
    
    /// Build an MPSGraph from the DAG, returning the final MPSGraphTensor for the `root`.
    static func buildGraph(from root: NDArray) throws -> (MPSGraph, MPSGraphTensor) {
        let sortedNodes = root.topologicalSort()
        guard !sortedNodes.isEmpty else {
            throw NDArrayError.emptyDAG("Root NDArray produced an empty DAG.")
        }
        
        let graph = MPSGraph()
        var nodeToTensor = [NDArray: MPSGraphTensor]()
        
        // Build MPSGraphTensors from leaf -> internal nodes.
        for node in sortedNodes {
            if let op = node.op {
                // Internal node
                let parentTensors = try node.parents.map { parent in
                    guard let t = nodeToTensor[parent] else {
                        throw NDArrayError.operationError(
                            "Missing parent tensor for node \(node.label ?? "<?>")."
                        )
                    }
                    return t
                }
                let newTensor = try op(graph, parentTensors, node.label)
                nodeToTensor[node] = newTensor
                
            } else {
                // Leaf node => constant or placeholder
                if let rawData = node.data {
                    // Constant
                    let data = rawData.withUnsafeBytes { Data($0) }
                    let t = graph.constant(data, shape: node.shape.map { NSNumber(value: $0) }, dataType: .float32)// Candidate expects value of type 'Data' for parameter #1 (got '[Float]') (MetalPerformanceShadersGraph.MPSGraph)
                    nodeToTensor[node] = t
                } else {
                    // Placeholder
                    let t = graph.placeholder(
                        shape: node.shape.map { NSNumber(value: $0) },
                        dataType: .float32,
                        name: node.label
                    )
                    nodeToTensor[node] = t
                }
            }
        }
        
        // The final node’s MPSGraphTensor
        guard let finalTensor = nodeToTensor[root] else {
            throw NDArrayError.operationError("No final tensor for root NDArray.")
        }
        
        return (graph, finalTensor)
    }
}

