// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ORBIT_VULKAN_LAYER_TIMER_QUERY_POOL_H_
#define ORBIT_VULKAN_LAYER_TIMER_QUERY_POOL_H_

#include "DispatchTable.h"
#include "OrbitBase/Logging.h"
#include "QueueFamilyInfoManager.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace orbit_vulkan_layer {

class TimerQueryPool {
 public:
  explicit TimerQueryPool(DispatchTable* dispatch_table,
                          QueueFamilyInfoManager* queue_family_info_manager,
                          PhysicalDeviceManager* physical_device_manager)
      : dispatch_table_(dispatch_table),
        queue_family_info_manager_(queue_family_info_manager),
        physical_device_manager_(physical_device_manager) {}
  void InitializeTimerQueryPool(const VkDevice& device, const VkPhysicalDevice& physical_device);
  [[nodiscard]] VkQueryPool GetQueryPool(const VkDevice& device);
  [[nodiscard]] bool NextReadyQuerySlot(const VkDevice& device, uint32_t* allocated_index);

  void ResetQuerySlots(const VkDevice& device, const std::vector<uint32_t>& physical_slot_indices);

  void RollbackPendingQuerySlots(const VkDevice& device,
                                 const std::vector<uint32_t>& physical_slot_indices);

 private:
  enum SlotState {
    kReadyForQueryIssue = 0,
    kQueryPendingOnGPU,
  };

  static constexpr uint32_t kNumPhysicalTimerQuerySlots = 65536;

  void CalibrateGPUTimeStamps(const VkDevice& device, const VkPhysicalDevice& physical_device,
                              const VkQueryPool& query_pool);

  DispatchTable* dispatch_table_;
  QueueFamilyInfoManager* queue_family_info_manager_;
  PhysicalDeviceManager* physical_device_manager_;
  absl::Mutex mutex_;
  absl::flat_hash_map<VkDevice, VkQueryPool> device_to_query_pool_;

  absl::flat_hash_map<VkDevice, std::array<SlotState, kNumPhysicalTimerQuerySlots>>
      device_to_query_slots_;
  absl::flat_hash_map<VkDevice, uint32_t> device_to_potential_next_free_index_;

  absl::flat_hash_map<VkDevice, std::vector<uint32_t>> device_to_pending_reset_slots_;
};
}  // namespace orbit_vulkan_layer

#endif  // ORBIT_VULKAN_LAYER_TIMER_QUERY_POOL_H_
