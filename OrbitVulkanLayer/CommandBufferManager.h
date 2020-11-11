// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ORBIT_VULKAN_LAYER_COMMAND_BUFFER_MANAGER_H_
#define ORBIT_VULKAN_LAYER_COMMAND_BUFFER_MANAGER_H_

#include <stack>

#include "DispatchTable.h"
#include "OrbitBase/Logging.h"
#include "OrbitConnector.h"
#include "TimerQueryPool.h"
#include "Writer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "vulkan/vulkan.h"

namespace orbit_vulkan_layer {

/*
 * This class is responsible to track command buffers and command pools.
 * TODO: So far it only tracks the allocation/de-allocation of buffers and pools.
 *  It should probably also track the timestamps inside the buffers (and the markers?)
 *
 * It also tracks which command buffer belongs to which device, which can be used
 * in the `DispatchTable` for function look-up.
 *
 * Thread-Safety: This class is internally synchronized (using read/write locks), and can be
 * safely accessed from different threads.
 */
class CommandBufferManager {
 public:
  explicit CommandBufferManager(DispatchTable* dispatch_table, TimerQueryPool* timer_query_pool,
                                PhysicalDeviceManager* physical_device_manager, Writer* writer,
                                const OrbitConnector* connector)
      : dispatch_table_(dispatch_table),
        timer_query_pool_(timer_query_pool),
        physical_device_manager_(physical_device_manager),
        writer_(writer),
        connector_(connector) {}
  void TrackCommandBuffers(VkDevice device, VkCommandPool pool,
                           const VkCommandBuffer* command_buffers, uint32_t count);
  void UntrackCommandBuffers(VkDevice device, VkCommandPool pool,
                             const VkCommandBuffer* command_buffers, uint32_t count);

  void MarkCommandBufferBegin(const VkCommandBuffer& command_buffer);

  void MarkCommandBufferEnd(const VkCommandBuffer& command_buffer);

  void DoPreSubmitQueue(VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits);
  void DoPostSubmitQueue(const VkQueue& queue);

  void CompleteSubmits(const VkDevice& device);

  void ResetCommandBuffer(const VkCommandBuffer& command_buffer);

  void ResetCommandPool(const VkCommandPool& command_pool);

 private:
  enum MarkerType { kCommandBuffer = 0, kDebugMarker };

  struct MarkerState {
    MarkerType type;
    std::string text;
    uint32_t slot_index;
  };

  struct CommandBufferState {
    MarkerState command_buffer_marker;
  };

  struct SubmittedCommandBuffer {
    MarkerState command_buffer_marker;
  };

  struct SubmitInfo {
    std::vector<SubmittedCommandBuffer> command_buffers;
  };

  struct QueueSubmission {
    uint64_t pre_submission_cpu_timestamp;
    uint64_t post_submission_cpu_timestamp;
    int32_t thread_id;
    std::vector<SubmitInfo> submit_infos;
  };

  absl::Mutex mutex_;
  absl::flat_hash_map<VkCommandPool, absl::flat_hash_set<VkCommandBuffer>> pool_to_command_buffers_;
  absl::flat_hash_map<VkCommandBuffer, VkDevice> command_buffer_to_device_;

  absl::flat_hash_map<VkCommandBuffer, CommandBufferState> command_buffer_to_state_;
  absl::flat_hash_map<VkQueue, std::vector<QueueSubmission>> queue_to_submissions_;

  DispatchTable* dispatch_table_;
  TimerQueryPool* timer_query_pool_;
  PhysicalDeviceManager* physical_device_manager_;
  Writer* writer_;

  const OrbitConnector* connector_;
};

}  // namespace orbit_vulkan_layer

#endif  // ORBIT_VULKAN_LAYER_COMMAND_BUFFER_MANAGER_H_