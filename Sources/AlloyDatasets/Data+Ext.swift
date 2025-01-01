//
//  Data+Ext.swift
//  Alloy
//
//  Created by Yuhao Chen on 12/31/24.
//
import Foundation
import zlib

extension Data {
    
    func gunzippedData() -> Data {
        var stream = z_stream()
        var status: Int32
        
        status = inflateInit2_(&stream, 47, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
        
        var data = Data(capacity: self.count * 2)
        repeat {
            if Int(stream.total_out) >= data.count {
                data.count += self.count / 2
            }
            
            let inputCount = self.count
            let outputCount = data.count
            
            self.withUnsafeBytes { (inputPointer: UnsafeRawBufferPointer) in
                stream.next_in =
                    UnsafeMutablePointer<Bytef>(mutating: inputPointer.bindMemory(to: Bytef.self).baseAddress!).advanced(by: Int(stream.total_in))
                stream.avail_in = uint(inputCount) - uInt(stream.total_in)
                
                data.withUnsafeMutableBytes { (outputPointer: UnsafeMutableRawBufferPointer) in
                    stream.next_out = outputPointer.bindMemory(to: Bytef.self).baseAddress!.advanced(by: Int(stream.total_out))
                    stream.avail_out = uInt(outputCount) - uInt(stream.total_out)
                    
                    status = inflate(&stream, Z_SYNC_FLUSH)
                    
                    stream.next_out = nil
                }
                
                stream.next_in = nil
            }
            
        } while status == Z_OK
        
        if inflateEnd(&stream) == Z_OK, status == Z_STREAM_END {
            data.count = Int(stream.total_out)
        }
        
        return data
    }
}
