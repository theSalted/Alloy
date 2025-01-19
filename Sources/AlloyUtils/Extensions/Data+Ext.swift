//
//  Data+Ext.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/1/25.
//
import Foundation

extension Data {
    /// Converts `Data` into `[Float]`, returning `nil` if `Data` is `nil`
    /// or if the size is not a multiple of the size of `Float`.
    public func toFloatArray() -> [Float]? {
        // Optionally, you can enforce that the data size must be a multiple
        // of `MemoryLayout<Float>.size`. If the Data can be partially filled,
        // you might omit this and just convert as-is.
        guard count % MemoryLayout<Float>.size == 0 else {
            return nil
        }
            
        return withUnsafeBytes {
            let floatPtr = $0.bindMemory(to: Float.self)
            return Array(floatPtr)
        }
    }
}
