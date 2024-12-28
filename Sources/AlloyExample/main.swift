//
//  main.swift
//  Alloy
//
//  Created by Yuhao Chen on 11/10/24.
//
import Alloy

let a = NDArray([1], shape: [1])
let b = NDArray([2], shape: [1])
let c = a + b
do {
    try run(c)
    print("Success: \(String(describing: c.data))")
} catch {
    print("Error: \(error)")
}
