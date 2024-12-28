//
//  NDArray+Ops.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/27/24.
//

extension NDArray {
    public static func +(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
        // -TODO: Implement broadcasting here
        let out = NDArray(elements: (lhs, rhs), operator: "+", shape: lhs.shape)
        
        return out
    }
    
}
