//
//  NDArray+Helper.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/11/24.
//

import Metal

extension NDArray {
    // Helper function to load Metal functions
    func getFunction(name: String) -> MTLFunction? {
        // Check if function is already cached
        if let cachedFunction = NDArray.functionCache[name] {
            return cachedFunction
        }
        
        // Load function from the default library
        guard let defaultLibrary = try? device.makeDefaultLibrary(bundle: Bundle.module),
              let function = defaultLibrary.makeFunction(name: name)
              else { fatalError("Unable to create default library") }
        
        // Cache the function for future use
        NDArray.functionCache[name] = function
        return function
    }
    
    // Build topological order for computational graph
    func makeTopologicalOrdered() -> [NDArray] {
        var visited = Set<NDArray>()
        var topo = [NDArray]()
        
        func buildOrder(_ node: NDArray) {
            if !visited.contains(node) {
                visited.insert(node)
                for input in node.inputs {
                    buildOrder(input)
                }
                topo.append(node)
            }
        }
        
        buildOrder(self)
        return topo
    }
}
