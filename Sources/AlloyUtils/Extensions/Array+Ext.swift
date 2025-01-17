//
//  Array+Ext.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/29/24.
//

import Foundation

public extension Array where Element == Int {
    /// Converts an array of Ints to an array of NSNumbers.
    func toNSNumberArray() -> [NSNumber] {
        return self.map { NSNumber(value: $0) }
    }
}

public extension Array where Element == NSNumber {
    /// Converts an array of NSNumbers to an array of Ints.
    func toIntArray() -> [Int] {
        return self.compactMap { $0.intValue }
    }
}

public extension Array where Element == Float {
    /// Converts an array of `Float` back to `Data`.
    /// - Parameter shape: The shape (dimensions) for which the array is valid.
    /// - Throws: A runtime error (via `preconditionFailure`) if the array count
    ///   does not match the product of the shape dimensions.
    func toData(shape: [Int]) -> Data {
        // Compute total elements from the shape
        let prod = shape.reduce(1, *)
        
        // Validate that our array's count matches the shape's product
        precondition(
            count == prod,
            "Data count (\(count)) != shape’s element count (\(prod))"
        )
        
        // Convert the array's memory to Data
        return withUnsafeBytes { Data($0) }
    }
}
