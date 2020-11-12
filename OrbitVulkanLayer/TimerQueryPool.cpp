// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "TimerQueryPool.h"

#include <thread>

#include "OrbitBase/Profiling.h"

namespace orbit_vulkan_layer {
void orbit_vulkan_layer::TimerQueryPool::InitializeTimerQueryPool(
    const VkDevice& device, const VkPhysicalDevice& physical_device) {
  LOG("InitializeTimerQueryPool");
  VkQueryPool query_pool;

  VkQueryPoolCreateInfo create_info = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .pNext = nullptr,
                                       .flags = 0,
                                       .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                       .queryCount = kNumPhysicalTimerQuerySlots,
                                       .pipelineStatistics = 0};

  VkResult result =
      dispatch_table_->CreateQueryPool(device)(device, &create_info, nullptr, &query_pool);
  CHECK(result == VK_SUCCESS);

  dispatch_table_->ResetQueryPoolEXT(device)(device, query_pool, 0, kNumPhysicalTimerQuerySlots);

  CalibrateGPUTimeStamps(device, physical_device, query_pool);

  {
    absl::WriterMutexLock lock(&mutex_);
    device_to_query_pool_[device] = query_pool;
    std::array<SlotState, kNumPhysicalTimerQuerySlots> slots{};
    slots.fill(kReadyForQueryIssue);
    device_to_query_slots_[device] = slots;
    device_to_pending_reset_slots_[device] = {};
    device_to_potential_next_free_index_[device] = 0;
  }
}

VkQueryPool TimerQueryPool::GetQueryPool(const VkDevice& device) {
  LOG("GetQueryPool");
  absl::ReaderMutexLock lock(&mutex_);
  CHECK(device_to_query_pool_.contains(device));
  return device_to_query_pool_.at(device);
}

// TODO: Rename
void TimerQueryPool::CalibrateGPUTimeStamps(const VkDevice& device,
                                            const VkPhysicalDevice& physical_device,
                                            const VkQueryPool& query_pool) {
  LOG("CalibrateGPUTimeStamps");

  uint32_t timestamp_queue_family_index =
      queue_family_info_manager_->SuitableQueueFamilyIndexForQueryPoolReset(physical_device);

  VkCommandPoolCreateInfo pool_create_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      // "Command buffers allocated from the pool will be short-lived:"
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = timestamp_queue_family_index};

  VkCommandPool pool = VK_NULL_HANDLE;
  VkResult result =
      dispatch_table_->CreateCommandPool(device)(device, &pool_create_info, nullptr, &pool);
  CHECK(result == VK_SUCCESS);

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo command_buffer_allocate_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};

  result = dispatch_table_->AllocateCommandBuffers(device)(device, &command_buffer_allocate_info,
                                                           &command_buffer);
  CHECK(result == VK_SUCCESS);

  VkMemoryBarrier memory_barrier = {
      .pNext = nullptr,
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask =
          VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
          VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
          VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
          VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask =
          VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
          VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
          VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
          VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT};

  VkEventCreateInfo event_create_info = {
      .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO, .pNext = nullptr, .flags = 0};

  VkEvent gpu_wait_event = VK_NULL_HANDLE;
  result =
      dispatch_table_->CreateEvent(device)(device, &event_create_info, nullptr, &gpu_wait_event);
  CHECK(result == VK_SUCCESS);

  VkEvent ready_to_write_timestamp_event_0 = VK_NULL_HANDLE;
  result = dispatch_table_->CreateEvent(device)(device, &event_create_info, nullptr,
                                                &ready_to_write_timestamp_event_0);
  CHECK(result == VK_SUCCESS);

  VkEvent ready_to_write_timestamp_event_1 = VK_NULL_HANDLE;
  result = dispatch_table_->CreateEvent(device)(device, &event_create_info, nullptr,
                                                &ready_to_write_timestamp_event_1);
  CHECK(result == VK_SUCCESS);

  VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .pNext = nullptr,
                                      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                      .pInheritanceInfo = nullptr};

  result = dispatch_table_->BeginCommandBuffer(command_buffer)(command_buffer, &begin_info);
  CHECK(result == VK_SUCCESS);

  dispatch_table_->CmdSetEvent(device)(command_buffer, gpu_wait_event,
                                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  dispatch_table_->CmdPipelineBarrier(device)(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1,
                                              &memory_barrier, 0, nullptr, 0, nullptr);
  dispatch_table_->CmdWaitEvents(device)(
      command_buffer, 1, &ready_to_write_timestamp_event_0, VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, nullptr, 0, nullptr, 0,
      nullptr);
  dispatch_table_->CmdWriteTimestamp(command_buffer)(
      command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
  dispatch_table_->CmdPipelineBarrier(device)(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1,
                                              &memory_barrier, 0, nullptr, 0, nullptr);
  dispatch_table_->CmdWaitEvents(device)(
      command_buffer, 1, &ready_to_write_timestamp_event_1, VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
  dispatch_table_->CmdPipelineBarrier(device)(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1,
                                              &memory_barrier, 0, nullptr, 0, nullptr);
  dispatch_table_->CmdWriteTimestamp(command_buffer)(
      command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 1);
  dispatch_table_->CmdPipelineBarrier(device)(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1,
                                              &memory_barrier, 0, nullptr, 0, nullptr);
  result = dispatch_table_->EndCommandBuffer(command_buffer)(command_buffer);
  CHECK(result == VK_SUCCESS);

  VkFence reset_fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};

  result = dispatch_table_->CreateFence(device)(device, &fence_info, nullptr, &reset_fence);
  CHECK(result == VK_SUCCESS);

  VkQueue reset_queue = VK_NULL_HANDLE;
  dispatch_table_->GetDeviceQueue(device)(device, timestamp_queue_family_index, 0, &reset_queue);

  VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .pNext = nullptr,
                           .waitSemaphoreCount = 0,
                           .pWaitSemaphores = nullptr,
                           .pWaitDstStageMask = nullptr,
                           .commandBufferCount = 1,
                           .pCommandBuffers = &command_buffer,
                           .signalSemaphoreCount = 0,
                           .pSignalSemaphores = nullptr};

  result = dispatch_table_->QueueSubmit(device)(reset_queue, 1, &submit_info, reset_fence);
  CHECK(result == VK_SUCCESS);

  while (true) {
    VkResult event_status = dispatch_table_->GetEventStatus(device)(device, gpu_wait_event);
    if (event_status == VK_EVENT_SET) {
      break;
    }
    CHECK(event_status == VK_EVENT_RESET);
  }

  uint64_t cpu_timestamp_0 = MonotonicTimestampNs();
  result = dispatch_table_->SetEvent(device)(device, ready_to_write_timestamp_event_0);
  CHECK(result == VK_SUCCESS);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  uint64_t cpu_timestamp_1 = MonotonicTimestampNs();
  result = dispatch_table_->SetEvent(device)(device, ready_to_write_timestamp_event_1);
  CHECK(result == VK_SUCCESS);

  while (true) {
    VkResult fence_status = dispatch_table_->GetFenceStatus(device)(device, reset_fence);
    if (fence_status == VK_SUCCESS) {
      break;
    }
  }

  uint64_t gpu_timestamp_0;
  uint64_t gpu_timestamp_1;
  VkDeviceSize result_stride = sizeof(uint64_t);
  result = dispatch_table_->GetQueryPoolResults(device)(device, query_pool, 0, 1,
                                                        sizeof(gpu_timestamp_0), &gpu_timestamp_0,
                                                        result_stride, VK_QUERY_RESULT_64_BIT);

  CHECK(result == VK_SUCCESS);

  result = dispatch_table_->GetQueryPoolResults(device)(device, query_pool, 1, 1,
                                                        sizeof(gpu_timestamp_1), &gpu_timestamp_1,
                                                        result_stride, VK_QUERY_RESULT_64_BIT);
  CHECK(result == VK_SUCCESS);

  dispatch_table_->DestroyFence(device)(device, reset_fence, nullptr);
  dispatch_table_->FreeCommandBuffers(device)(device, pool, 1, &command_buffer);
  dispatch_table_->DestroyCommandPool(device)(device, pool, nullptr);
  dispatch_table_->DestroyEvent(device)(device, gpu_wait_event, nullptr);
  dispatch_table_->DestroyEvent(device)(device, ready_to_write_timestamp_event_0, nullptr);
  dispatch_table_->DestroyEvent(device)(device, ready_to_write_timestamp_event_1, nullptr);

  VkPhysicalDeviceProperties properties =
      physical_device_manager_->GetPhysicalDeviceProperties(physical_device);
  gpu_timestamp_0 = static_cast<uint64_t>(static_cast<double>(gpu_timestamp_0) *
                                          properties.limits.timestampPeriod);
  gpu_timestamp_1 = static_cast<uint64_t>(static_cast<double>(gpu_timestamp_1) *
                                          properties.limits.timestampPeriod);

  int64_t offset_0 = cpu_timestamp_0 - gpu_timestamp_0;
  int64_t offset_1 = cpu_timestamp_1 - gpu_timestamp_1;
  int64_t approx_offset = (offset_0 + offset_1) / 2;

  physical_device_manager_->RegisterApproxCpuTimestampOffset(physical_device, approx_offset);

  LOG("DIFF GPU / CPU 1: %ld ns", cpu_timestamp_0 - gpu_timestamp_0);
  LOG("DIFF GPU / CPU 2: %ld ns", cpu_timestamp_1 - gpu_timestamp_1);
  LOG("GPU Timerange: %lu ns", gpu_timestamp_1 - gpu_timestamp_0);
  LOG("CPU Timerange: %lu ns", cpu_timestamp_1 - cpu_timestamp_0);

  dispatch_table_->ResetQueryPoolEXT(device)(device, query_pool, 0, 2);
}

bool TimerQueryPool::NextReadyQuerySlot(const VkDevice& device, uint32_t* allocated_index) {
  LOG("NextReadyQuerySlot");
  absl::WriterMutexLock lock(&mutex_);
  CHECK(device_to_potential_next_free_index_.contains(device));
  CHECK(device_to_query_slots_.contains(device));
  uint32_t potential_next_free_slot = device_to_potential_next_free_index_.at(device);
  std::array<SlotState, kNumPhysicalTimerQuerySlots>& slots = device_to_query_slots_.at(device);
  uint32_t current_slot = potential_next_free_slot;
  do {
    if (slots.at(current_slot) == kReadyForQueryIssue) {
      device_to_potential_next_free_index_[device] =
          (current_slot + 1) % kNumPhysicalTimerQuerySlots;
      slots.at(current_slot) = kQueryPendingOnGPU;
      *allocated_index = current_slot;
      return true;
    }
    current_slot = (current_slot + 1) % kNumPhysicalTimerQuerySlots;
  } while (current_slot != potential_next_free_slot);

  return false;
}

void TimerQueryPool::ResetQuerySlots(const VkDevice& device,
                                     const std::vector<uint32_t>& physical_slot_indices) {
  if (physical_slot_indices.empty()) {
    return;
  }
  LOG("ResetQuerySlots");
  absl::WriterMutexLock lock(&mutex_);
  std::array<SlotState, kNumPhysicalTimerQuerySlots>& slot_states =
      device_to_query_slots_.at(device);
  for (uint32_t physical_slot_index : physical_slot_indices) {
    CHECK(physical_slot_index < kNumPhysicalTimerQuerySlots);
    const SlotState& current_state = slot_states.at(physical_slot_index);
    CHECK(current_state == kQueryPendingOnGPU);
    const VkQueryPool& query_pool = device_to_query_pool_.at(device);
    dispatch_table_->ResetQueryPoolEXT(device)(device, query_pool, physical_slot_index, 1);
    slot_states.at(physical_slot_index) = kReadyForQueryIssue;
  }
}

void TimerQueryPool::RollbackPendingQuerySlots(const VkDevice& device,
                                               const std::vector<uint32_t>& physical_slot_indices) {
  if (physical_slot_indices.empty()) {
    return;
  }
  LOG("RollbackPendingQuerySlots");
  absl::WriterMutexLock lock(&mutex_);
  std::array<SlotState, kNumPhysicalTimerQuerySlots>& slot_states =
      device_to_query_slots_.at(device);
  for (uint32_t physical_slot_index : physical_slot_indices) {
    CHECK(physical_slot_index < kNumPhysicalTimerQuerySlots);
    const SlotState& current_state = slot_states.at(physical_slot_index);
    CHECK(current_state == kQueryPendingOnGPU);
    slot_states.at(physical_slot_index) = kReadyForQueryIssue;
  }
}

}  // namespace orbit_vulkan_layer