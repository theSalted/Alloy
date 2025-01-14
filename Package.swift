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
        .executable(name: "AlloyExample", targets: ["AlloyExample"])
    ],
    targets: [
        .target(name: "Alloy"),
        .target(name: "AlloyRandom", dependencies: ["Alloy"]),
        .target(name: "AlloyDatasets", dependencies: ["Alloy"]),
        .target(name: "AlloyNN", dependencies: ["Alloy"]),
        .executableTarget(name: "AlloyExample", dependencies: ["Alloy", "AlloyDatasets", "AlloyRandom"]),
        
        // Add a test target that depends on "Alloy" and the "Testing" framework
        .testTarget(
            name: "AlloyTests",
            dependencies: [
                "Alloy",
            ]
        )
    ]
)
