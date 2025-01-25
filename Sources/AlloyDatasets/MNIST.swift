//
//  MNIST.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/31/24.
//

import Alloy
import Foundation
import zlib
import MetalPerformanceShaders

fileprivate let MNISTImageMetadataPrefixSize = 16
fileprivate let MNISTLabelsMetadataPrefixSize = 8
fileprivate let MNISTSize = 28
fileprivate let MNISTNumClasses = 10

fileprivate let trainImagesURL = "https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-images-idx3-ubyte.gz"
fileprivate let trainLabelsURL = "https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-labels-idx1-ubyte.gz"
fileprivate let testImagesURL = "https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/t10k-images-idx3-ubyte.gz"
fileprivate let testLabelsURL = "https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/t10k-labels-idx1-ubyte.gz"

enum MNISTError: Error {
    case dataNotLoaded
    case downloadFailed(String)
    case dataUnzipFailed
}

public class MNIST {
    private var trainImages: Data?
    private var trainLabels: Data?
    private var testImages: Data?
    private var testLabels: Data?
    private var device: MTLDevice

    /// The number of training samples (images/labels) in the dataset
    private var numTrainSamples: Int {
        guard let labels = trainLabels else { return 0 }
        return labels.count - MNISTLabelsMetadataPrefixSize
    }

    /// The number of testing samples (images/labels) in the dataset
    private var numTestSamples: Int {
        guard let labels = testLabels else { return 0 }
        return labels.count - MNISTLabelsMetadataPrefixSize
    }

    public init(device: MTLDevice? = nil) throws {
        self.device = device ?? Alloy.shared.device
        try loadDataset()
    }

    private func downloadAndUnzip(url: String) throws -> Data {
        guard let remoteURL = URL(string: url),
              let compressedData = try? Data(contentsOf: remoteURL)
        else {
            throw MNISTError.downloadFailed(url)
        }
        
        guard let unzippedData = compressedData.gunzippedData() else {
            throw MNISTError.dataUnzipFailed
        }
        
        return unzippedData
    }

    private func loadDataset() throws {
        trainImages = try downloadAndUnzip(url: trainImagesURL)
        trainLabels = try downloadAndUnzip(url: trainLabelsURL)
        testImages  = try downloadAndUnzip(url: testImagesURL)
        testLabels  = try downloadAndUnzip(url: testLabelsURL)
    }

    /// Retrieves a batch of data. If `batchSize` is nil, returns the entire dataset.
    /// The returned images are in **NHWC** layout: [batchSize, height=28, width=28, channels=1].
    ///
    /// - Parameters:
    ///   - images: Raw image data (may be training or testing).
    ///   - labels: Raw label data (may be training or testing).
    ///   - batchSize: The number of samples to retrieve. Defaults to the entire dataset if nil.
    /// - Returns: A tuple containing `(imagesNDArray, labelsNDArray)`.
    public func getBatch(
        from images: Data?,
        labels: Data?,
        batchSize: Int? = nil
    ) throws -> (NDArray, NDArray) {
        guard let images = images, let labels = labels else {
            throw MNISTError.dataNotLoaded
        }

        let imageCount = labels.count - MNISTLabelsMetadataPrefixSize
        let actualBatchSize = batchSize ?? imageCount

        // Ensure the batch size does not exceed the dataset size
        guard actualBatchSize > 0 && actualBatchSize <= imageCount else {
            throw MNISTError.downloadFailed("""
            Invalid batch size: \(actualBatchSize). Must be between 1 and \(imageCount).
            """)
        }

        // Set shape to [batchSize, 28, 28, 1] for NHWC
        let imageShape: [NSNumber] = [
            NSNumber(value: actualBatchSize),
            NSNumber(value: MNISTSize),
            NSNumber(value: MNISTSize),
            NSNumber(value: 1)
        ]
        
        // Labels remain [batchSize, 10]
        let labelShape: [NSNumber] = [
            NSNumber(value: actualBatchSize),
            NSNumber(value: MNISTNumClasses)
        ]

        // Create MPSNDArray descriptors
        let inputDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: imageShape
        )
        let labelDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: labelShape
        )

        // Allocate GPU buffers
        let inputs      = MPSNDArray(device: device, descriptor: inputDesc)
        let labelsArray = MPSNDArray(device: device, descriptor: labelDesc)

        // Allocate CPU-side buffers with matching total element count
        let imageElements = actualBatchSize * MNISTSize * MNISTSize * 1
        var inputBuffer = [Float](repeating: 0.0, count: imageElements)

        let labelElements = actualBatchSize * MNISTNumClasses
        var labelBuffer = [Float](repeating: 0.0, count: labelElements)

        // Fill them from random indices in the dataset
        for i in 0..<actualBatchSize {
            let index = Int.random(in: 0..<imageCount)
            
            // Each image is MNISTSize * MNISTSize = 784 bytes (for grayscale)
            let imageOffset = MNISTImageMetadataPrefixSize + index * MNISTSize * MNISTSize
            // Each label is a single byte
            let labelOffset = MNISTLabelsMetadataPrefixSize + index

            // Fill the image row, normalizing to [0..1]
            // For NHWC, we store each image in row-major order:
            // dimension 0 = batch index
            // dimension 1 = y
            // dimension 2 = x
            // dimension 3 = channel
            //
            // Because there's only 1 channel, we can just flatten it:
            let base = i * (MNISTSize * MNISTSize)
            for pixel in 0..<(MNISTSize * MNISTSize) {
                let gray = Float(images[imageOffset + pixel]) / 255.0
                inputBuffer[base + pixel] = gray
            }

            // Fill the label row as one-hot
            let labelVal = labels[labelOffset]
            for classIndex in 0..<MNISTNumClasses {
                // row i => i * MNISTNumClasses
                let labelIdx = i * MNISTNumClasses + classIndex
                labelBuffer[labelIdx] = (classIndex == labelVal ? 1.0 : 0.0)
            }
        }

        // Upload CPU buffers → GPU arrays
        inputs.writeBytes(&inputBuffer, strideBytes: nil)
        labelsArray.writeBytes(&labelBuffer, strideBytes: nil)

        // Finally create the NDArray wrappers with our desired shape (NHWC for images)
        let imagesNDArray = NDArray(
            mpsArray: inputs,
            shape: imageShape.map { $0.intValue }, // [batch, 28, 28, 1]
            label: "mnist_images"
        )
        let labelsNDArray = NDArray(
            mpsArray: labelsArray,
            shape: labelShape.map { $0.intValue }, // [batch, 10]
            label: "mnist_labels"
        )

        return (imagesNDArray, labelsNDArray)
    }

    /// Retrieves a training batch. If `batchSize` is nil, returns the entire training set.
    ///
    /// - Parameter batchSize: The number of samples in the batch (defaults to the entire training set).
    /// - Returns: A tuple `(images, labels)` shaped `[N, 28, 28, 1]` and `[N, 10]`.
    public func getTrainingBatch(batchSize: Int? = nil) throws -> (NDArray, NDArray) {
        return try getBatch(
            from: trainImages,
            labels: trainLabels,
            batchSize: batchSize ?? numTrainSamples
        )
    }

    /// Retrieves a testing batch. If `batchSize` is nil, returns the entire testing set.
    ///
    /// - Parameter batchSize: The number of samples in the batch (defaults to the entire test set).
    /// - Returns: A tuple `(images, labels)` shaped `[N, 28, 28, 1]` and `[N, 10]`.
    public func getTestingBatch(batchSize: Int? = nil) throws -> (NDArray, NDArray) {
        return try getBatch(
            from: testImages,
            labels: testLabels,
            batchSize: batchSize ?? numTestSamples
        )
    }
}
