// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Alloy",
    platforms: [
        .macOS(.v15),
        .iOS(.v17),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "Alloy", targets: ["Alloy"]),
        .library(name: "AlloyRandom", targets: ["AlloyRandom"]),
        .library(name: "AlloyDatasets", targets: ["AlloyDatasets"]),
        .library(name: "AlloyNN", targets: ["AlloyNN"]),
        .library(name: "AlloyUtils", targets: ["AlloyUtils"]),
        .executable(name: "AlloyExamples", targets: ["AlloyExamples"])
    ],
    targets: [
        .target(name: "Alloy", dependencies: ["AlloyUtils"]),
        .target(name: "AlloyRandom", dependencies: ["Alloy"]),
        .target(name: "AlloyDatasets", dependencies: ["Alloy"]),
        .target(name: "AlloyNN", dependencies: ["Alloy"]),
        .target(name: "AlloyUtils"),
        .executableTarget(name: "AlloyExamples", dependencies: ["Alloy", "AlloyDatasets", "AlloyRandom", "AlloyUtils"]),
        
        // Add a test target that depends on "Alloy" and the "Testing" framework
        .testTarget(
            name: "AlloyTests",
            dependencies: [
                "Alloy",
            ]
        )
    ]
)
