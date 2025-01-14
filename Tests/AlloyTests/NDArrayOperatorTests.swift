//
//  NDArrayOperatorTests.swift
//  Alloy
//
//  Created by Yuhao Chen on 1/13/25.
//

import Foundation
import Testing
@testable import Alloy

extension NDArray {
    var toFloatArray: [Float] {
        (try? self.toArray()) ?? []
    }
}

func ndArraysEqual(_ lhs: NDArray, _ rhs: NDArray) -> Bool {
    lhs.shape == rhs.shape
    && lhs.toFloatArray.elementsEqual(rhs.toFloatArray)
}

struct NDArrayOperatorTests {
    
    // MARK: - NDArray + NDArray
    @Test(
        "Test NDArray addition with matching shapes",
        arguments: [
            // Note the explicit `as [Float]`
            ([1.0, 2.0] as [Float],
             [3.0, 4.0] as [Float],
             [4.0, 6.0] as [Float]),
            
            ([0.0, -1.0] as [Float],
             [0.5, 2.5] as [Float],
             [0.5, 1.5] as [Float])
        ]
    )
    func testAddNDArrays(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs + rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - NDArray + Float
    @Test(
        "Test NDArray + Float",
        arguments: [
            // Cast scalars to `Float` too
            ([1.0, 2.0] as [Float], 3.0 as Float, [4.0, 5.0] as [Float]),
            ([0.0, -1.0] as [Float], 2.0 as Float, [2.0, 1.0] as [Float])
        ]
    )
    func testAddNDArrayFloat(lhsValues: [Float], scalar: Float, expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs + scalar
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - Float + NDArray
    @Test(
        "Test Float + NDArray",
        arguments: [
            (10.0 as Float, [1.0, 2.0] as [Float], [11.0, 12.0] as [Float]),
            (5.0 as Float, [0.0, -1.0] as [Float], [5.0, 4.0] as [Float])
        ]
    )
    func testAddFloatNDArray(scalar: Float, rhsValues: [Float], expectedValues: [Float]) {
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = scalar + rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - NDArray - NDArray
    @Test(
        "Test NDArray subtraction",
        arguments: [
            ([4.0, 5.0] as [Float], [1.0, 2.0] as [Float], [3.0, 3.0] as [Float]),
            ([0.0, 0.0] as [Float], [2.0, -2.0] as [Float], [-2.0, 2.0] as [Float])
        ]
    )
    func testSubtractNDArrays(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs - rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - NDArray * NDArray
    @Test(
        "Test NDArray multiplication",
        arguments: [
            ([2.0, 3.0] as [Float], [4.0, 5.0] as [Float], [8.0, 15.0] as [Float]),
            ([0.0, 1.0] as [Float], [1.0, 10.0] as [Float], [0.0, 10.0] as [Float])
        ]
    )
    func testMultiplyNDArrays(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs * rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - NDArray / NDArray
    @Test(
        "Test NDArray division",
        arguments: [
            ([8.0, 4.0] as [Float], [2.0, 2.0] as [Float], [4.0, 2.0] as [Float]),
            ([10.0, 0.5] as [Float], [5.0, 0.5] as [Float], [2.0, 1.0] as [Float])
        ]
    )
    func testDivideNDArrays(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs / rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - Exponent (^)
    @Test(
        "Test NDArray exponent",
        arguments: [
            ([2.0, 3.0] as [Float], [2.0, 2.0] as [Float], [4.0, 9.0] as [Float]),
            ([4.0, 9.0] as [Float], [0.5, 0.5] as [Float], [2.0, 3.0] as [Float])
        ]
    )
    func testExponent(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs ^ rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - Modulus (%)
    @Test(
        "Test NDArray modulus",
        arguments: [
            ([5.0, 7.0] as [Float], [2.0, 3.0] as [Float], [1.0, 1.0] as [Float]),
            ([10.0, 12.0] as [Float], [5.0, 5.0] as [Float], [0.0, 2.0] as [Float])
        ]
    )
    func testModulus(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs % rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - Comparisons
    @Test(
        "Test NDArray comparisons (>)",
        arguments: [
            ([1.0, 3.0] as [Float], [2.0, 1.0] as [Float], [0.0, 1.0] as [Float]),
            ([2.0, 2.0] as [Float], [2.0, 10.0] as [Float], [0.0, 0.0] as [Float])
        ]
    )
    func testGreater(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs > rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    @Test(
        "Test NDArray comparisons (<)",
        arguments: [
            ([1.0, 3.0] as [Float], [2.0, 1.0] as [Float], [1.0, 0.0] as [Float]),
            ([2.0, -1.0] as [Float], [2.0, 0.0] as [Float], [0.0, 1.0] as [Float])
        ]
    )
    func testLess(lhsValues: [Float], rhsValues: [Float], expectedValues: [Float]) {
        let lhs = NDArray(lhsValues, shape: [lhsValues.count])
        let rhs = NDArray(rhsValues, shape: [rhsValues.count])
        let expected = NDArray(expectedValues, shape: [expectedValues.count])
        
        let result = lhs < rhs
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        #expect(
            ndArraysEqual(result, expected),
            "Expected \(expectedValues), got \(result.toFloatArray)"
        )
    }

    // MARK: - Reductions
    @Test(
        "Test NDArray sum",
        arguments: [
            ([1.0, 2.0, 3.0] as [Float], 6.0 as Float),
            ([2.0, 2.0, 2.0, 2.0] as [Float], 8.0 as Float)
        ]
    )
    func testSum(values: [Float], expectedSum: Float) {
        let array = NDArray(values, shape: [values.count])
        let result = array.sum()
        do { try run(result) } catch {
            Issue.record("run() threw an error: \(error)")
        }
        
        let actual = result.toFloatArray.first ?? Float.nan
        #expect(
            actual == expectedSum,
            "Expected sum \(expectedSum), got \(actual)"
        )
    }
}
