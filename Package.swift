// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Alloy",
    platforms: [
        .macOS(.v15),
        .iOS(.v17),
        .visionOS(.v1)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Alloy",
            targets: ["Alloy"]),
        .library(
            name: "AlloyDatasets",
            targets: ["AlloyDatasets"]),
        .library(
            name: "AlloyNN",
            targets: ["AlloyNN"]),
        .executable(name: "AlloyExample",
                    targets: ["AlloyExample"])
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "Alloy"),
        .target(
            name: "AlloyDatasets",
            dependencies: ["Alloy"]),
        .target(
            name: "AlloyNN",
            dependencies: ["Alloy"]),
        .executableTarget(
            name: "AlloyExample",
            dependencies: ["Alloy", "AlloyDatasets"]
        )

    ]
)
