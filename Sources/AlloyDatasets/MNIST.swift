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

    private var numTrainSamples: Int {
        guard let labels = trainLabels else { return 0 }
        return labels.count - MNISTLabelsMetadataPrefixSize
    }

    private var numTestSamples: Int {
        guard let labels = testLabels else { return 0 }
        return labels.count - MNISTLabelsMetadataPrefixSize
    }

    public init(device: MTLDevice? = nil) throws {
        self.device = device ?? Alloy.shared.device
        try loadDataset()
    }

    private func downloadAndUnzip(url: String) throws -> Data {
        guard let remoteURL = URL(string: url), let compressedData = try? Data(contentsOf: remoteURL) else {
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
        testImages = try downloadAndUnzip(url: testImagesURL)
        testLabels = try downloadAndUnzip(url: testLabelsURL)
    }

    /// Retrieves a batch of data. If `batchSize` is nil, returns the entire dataset.
    /// - Parameters:
    ///   - images: Image data.
    ///   - labels: Label data.
    ///   - batchSize: The number of samples to retrieve. Defaults to the entire dataset if nil.
    /// - Returns: A tuple containing input and label NDArrays.
    public func getBatch(from images: Data?, labels: Data?, batchSize: Int? = nil) throws -> (NDArray, NDArray) {
        guard let images = images, let labels = labels else {
            throw MNISTError.dataNotLoaded
        }

        let imageCount = labels.count - MNISTLabelsMetadataPrefixSize
        let actualBatchSize = batchSize ?? imageCount

        // Ensure the batch size does not exceed the dataset size
        guard actualBatchSize > 0 && actualBatchSize <= imageCount else {
            throw MNISTError.downloadFailed("Invalid batch size: \(actualBatchSize). Must be between 1 and \(imageCount).")
        }

        // Updated imageShape to include the channel dimension
        let imageShape = [actualBatchSize as NSNumber, 1 as NSNumber, MNISTSize as NSNumber, MNISTSize as NSNumber]
        let labelShape = [actualBatchSize as NSNumber, MNISTNumClasses as NSNumber]

        let inputDesc = MPSNDArrayDescriptor(dataType: .float32, shape: imageShape)
        let labelDesc = MPSNDArrayDescriptor(dataType: .float32, shape: labelShape)

        let inputs = MPSNDArray(device: device, descriptor: inputDesc)
        let labelsArray = MPSNDArray(device: device, descriptor: labelDesc)

        // Adjust the buffer size to account for the additional channel dimension
        var inputBuffer = [Float](repeating: 0, count: actualBatchSize * 1 * MNISTSize * MNISTSize)
        var labelBuffer = [Float](repeating: 0, count: actualBatchSize * MNISTNumClasses)

        for i in 0..<actualBatchSize {
            let index = Int.random(in: 0..<imageCount)
            let imageOffset = MNISTImageMetadataPrefixSize + index * MNISTSize * MNISTSize
            let labelOffset = MNISTLabelsMetadataPrefixSize + index

            for pixel in 0..<(MNISTSize * MNISTSize) {
                inputBuffer[i * 1 * MNISTSize * MNISTSize + pixel] = Float(images[imageOffset + pixel]) / 255.0
            }

            let label = labels[labelOffset]
            for classIndex in 0..<MNISTNumClasses {
                labelBuffer[i * MNISTNumClasses + classIndex] = (classIndex == label ? 1.0 : 0.0)
            }
        }

        inputs.writeBytes(&inputBuffer, strideBytes: nil)
        labelsArray.writeBytes(&labelBuffer, strideBytes: nil)

        return (NDArray(mpsArray: inputs, shape: imageShape.toIntArray()), NDArray(mpsArray: labelsArray))
    }

    /// Retrieves a training batch. If `batchSize` is nil, returns the entire training dataset.
    /// - Parameter batchSize: The number of samples to retrieve. Defaults to the entire training set if nil.
    /// - Returns: A tuple containing input and label NDArrays.
    public func getTrainingBatch(batchSize: Int? = nil) throws -> (NDArray, NDArray) {
        return try getBatch(from: trainImages, labels: trainLabels, batchSize: batchSize ?? numTrainSamples)
    }

    /// Retrieves a testing batch. If `batchSize` is nil, returns the entire testing dataset.
    /// - Parameter batchSize: The number of samples to retrieve. Defaults to the entire testing set if nil.
    /// - Returns: A tuple containing input and label NDArrays.
    public func getTestingBatch(batchSize: Int? = nil) throws -> (NDArray, NDArray) {
        return try getBatch(from: testImages, labels: testLabels, batchSize: batchSize ?? numTestSamples)
    }
}
