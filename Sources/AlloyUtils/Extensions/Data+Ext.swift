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
        guard count % MemoryLayout<Float>.size == 0 else {
            return nil
        }
        
        let floatArray = withUnsafeBytes {
            let floatPtr = $0.bindMemory(to: Float.self)
            return Array(floatPtr)
        }
        
        // If necessary, convert endianness
        return floatArray.map { Float(bitPattern: $0.bitPattern.bigEndian) }
    }
}
