// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ORBIT_VULKAN_LAYER_WRITER_H_
#define ORBIT_VULKAN_LAYER_WRITER_H_

#include <fstream>
#include <iosfwd>
#include <string>
#include <utility>

#include "OrbitBase/Logging.h"
#include "absl/synchronization/mutex.h"
#include "capture.grpc.pb.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"

namespace orbit_vulkan_layer {
class Writer {
 public:
  explicit Writer(std::string filename) : filename_(std::move(filename)) {}

  void WriteQueueSubmission(const orbit_grpc_protos::GpuQueueSubmisssion& submisssion) {
    LOG("WriteQueueSubmission");
    WriteMessage(submisssion);
  }

 private:
  void WriteMessage(const google::protobuf::Message& message) {
    LOG("WriteMessage");
    std::ofstream stream(filename_, std::ios::app | std::ios::binary);
    google::protobuf::io::OstreamOutputStream out_stream(&stream);
    google::protobuf::io::CodedOutputStream coded_output(&out_stream);
    uint32_t message_size = message.ByteSizeLong();
    coded_output.WriteLittleEndian32(message_size);
    message.SerializeToCodedStream(&coded_output);
  }

  std::string filename_;
};

}  // namespace orbit_vulkan_layer

#endif  // ORBIT_VULKAN_LAYER_WRITER_H_
