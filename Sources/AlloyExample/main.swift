//
//  main.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/10/24.
//
import Alloy

let a = NDArray([1, 1, 1, 1], shape: [2, 2])
let b = NDArray([2], shape: [1])

let c = a + b
let d = c * 4
let e = d / 12

do {
    try run(c, d, e)
    print("Success (c): \(c)")
    print("Success (d): \(d)")
    print("Success (e): \(e)")

} catch {
    print("Error: \(error)")
}

