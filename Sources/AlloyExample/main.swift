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
    try run(c)
    print("Success (c): \(c)")
    try run(d)
    print("Success (d): \(d)")
    try run(e)
    print("Success (e): \(e)")

} catch {
    print("Error: \(error)")
}
