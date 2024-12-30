//
//  Array+Ext.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/29/24.
//

import Foundation

extension Array where Element == Int {
    /// Converts an array of Ints to an array of NSNumbers.
    func toNSNumberArray() -> [NSNumber] {
        return self.map { NSNumber(value: $0) }
    }
}

extension Array where Element == NSNumber {
    /// Converts an array of NSNumbers to an array of Ints.
    func toIntArray() -> [Int] {
        return self.compactMap { $0.intValue }
    }
}
