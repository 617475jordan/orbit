// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ORBIT_VULKAN_LAYER_DISPATCH_TABLE_H_
#define ORBIT_VULKAN_LAYER_DISPATCH_TABLE_H_

#include "OrbitBase/Logging.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "vulkan/vk_layer.h"
#include "vulkan/vk_layer_dispatch_table.h"
#include "vulkan/vulkan.h"

namespace orbit_vulkan_layer {
/*
 * A thread-safe dispatch table for vulkan function look-up.
 *
 * It computes/stores the vulkan dispatch tables for concrete devices/instances and provides
 * accessors to the functions.
 *
 * Thread-Safety: This class is internally synchronized (using read/write locks) and can be safely
 * accessed from different threads.
 */
class DispatchTable {
 public:
  DispatchTable() = default;

  void CreateInstanceDispatchTable(
      const VkInstance& instance,
      const PFN_vkGetInstanceProcAddr& next_get_instance_proc_addr_function) {
    VkLayerInstanceDispatchTable dispatch_table;
    dispatch_table.DestroyInstance = absl::bit_cast<PFN_vkDestroyInstance>(
        next_get_instance_proc_addr_function(instance, "vkDestroyInstance"));
    dispatch_table.GetInstanceProcAddr = absl::bit_cast<PFN_vkGetInstanceProcAddr>(
        next_get_instance_proc_addr_function(instance, "vkGetInstanceProcAddr"));
    dispatch_table.EnumerateDeviceExtensionProperties =
        absl::bit_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            next_get_instance_proc_addr_function(instance, "vkEnumerateDeviceExtensionProperties"));
    dispatch_table.GetPhysicalDeviceQueueFamilyProperties =
        absl::bit_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            next_get_instance_proc_addr_function(instance,
                                                 "vkGetPhysicalDeviceQueueFamilyProperties"));
    dispatch_table.GetPhysicalDeviceProperties = absl::bit_cast<PFN_vkGetPhysicalDeviceProperties>(
        next_get_instance_proc_addr_function(instance, "vkGetPhysicalDeviceProperties"));

    {
      absl::WriterMutexLock lock(&mutex_);
      instance_dispatch_table_[GetDispatchTableKey(instance)] = dispatch_table;
    }
  }

  void RemoveInstanceDispatchTable(const VkInstance& instance) {
    absl::WriterMutexLock lock(&mutex_);
    instance_dispatch_table_.erase(GetDispatchTableKey(instance));
  }

  void CreateDeviceDispatchTable(const VkDevice& device,
                                 const PFN_vkGetDeviceProcAddr& next_get_device_proc_add_function) {
    VkLayerDispatchTable dispatch_table;

    dispatch_table.GetDeviceProcAddr = absl::bit_cast<PFN_vkGetDeviceProcAddr>(
        next_get_device_proc_add_function(device, "vkGetDeviceProcAddr"));

    dispatch_table.CreateCommandPool = absl::bit_cast<PFN_vkCreateCommandPool>(
        next_get_device_proc_add_function(device, "vkCreateCommandPool"));
    dispatch_table.DestroyCommandPool = absl::bit_cast<PFN_vkDestroyCommandPool>(
        next_get_device_proc_add_function(device, "vkDestroyCommandPool"));
    dispatch_table.ResetCommandPool = absl::bit_cast<PFN_vkResetCommandPool>(
        next_get_device_proc_add_function(device, "vkResetCommandPool"));

    dispatch_table.AllocateCommandBuffers = absl::bit_cast<PFN_vkAllocateCommandBuffers>(
        next_get_device_proc_add_function(device, "vkAllocateCommandBuffers"));
    dispatch_table.FreeCommandBuffers = absl::bit_cast<PFN_vkFreeCommandBuffers>(
        next_get_device_proc_add_function(device, "vkFreeCommandBuffers"));
    dispatch_table.BeginCommandBuffer = absl::bit_cast<PFN_vkBeginCommandBuffer>(
        next_get_device_proc_add_function(device, "vkBeginCommandBuffer"));
    dispatch_table.EndCommandBuffer = absl::bit_cast<PFN_vkEndCommandBuffer>(
        next_get_device_proc_add_function(device, "vkEndCommandBuffer"));
    dispatch_table.ResetCommandBuffer = absl::bit_cast<PFN_vkResetCommandBuffer>(
        next_get_device_proc_add_function(device, "vkResetCommandBuffer"));

    dispatch_table.QueueSubmit = absl::bit_cast<PFN_vkQueueSubmit>(
        next_get_device_proc_add_function(device, "vkQueueSubmit"));
    dispatch_table.QueuePresentKHR = absl::bit_cast<PFN_vkQueuePresentKHR>(
        next_get_device_proc_add_function(device, "vkQueuePresentKHR"));

    dispatch_table.GetDeviceQueue = absl::bit_cast<PFN_vkGetDeviceQueue>(
        next_get_device_proc_add_function(device, "vkGetDeviceQueue"));
    dispatch_table.GetDeviceQueue2 = absl::bit_cast<PFN_vkGetDeviceQueue2>(
        next_get_device_proc_add_function(device, "vkGetDeviceQueue2"));

    dispatch_table.CreateQueryPool = absl::bit_cast<PFN_vkCreateQueryPool>(
        next_get_device_proc_add_function(device, "vkCreateQueryPool"));
    dispatch_table.CmdResetQueryPool = absl::bit_cast<PFN_vkCmdResetQueryPool>(
        next_get_device_proc_add_function(device, "vkCmdResetQueryPool"));
    dispatch_table.ResetQueryPoolEXT = absl::bit_cast<PFN_vkResetQueryPoolEXT>(
        next_get_device_proc_add_function(device, "vkResetQueryPoolEXT"));

    dispatch_table.CmdWriteTimestamp = absl::bit_cast<PFN_vkCmdWriteTimestamp>(
        next_get_device_proc_add_function(device, "vkCmdWriteTimestamp"));

    dispatch_table.CmdBeginQuery = absl::bit_cast<PFN_vkCmdBeginQuery>(
        next_get_device_proc_add_function(device, "vkCmdBeginQuery"));
    dispatch_table.CmdEndQuery = absl::bit_cast<PFN_vkCmdEndQuery>(
        next_get_device_proc_add_function(device, "vkCmdEndQuery"));
    dispatch_table.GetQueryPoolResults = absl::bit_cast<PFN_vkGetQueryPoolResults>(
        next_get_device_proc_add_function(device, "vkGetQueryPoolResults"));

    dispatch_table.CreateFence = absl::bit_cast<PFN_vkCreateFence>(
        next_get_device_proc_add_function(device, "vkCreateFence"));
    dispatch_table.DestroyFence = absl::bit_cast<PFN_vkDestroyFence>(
        next_get_device_proc_add_function(device, "vkDestroyFence"));
    dispatch_table.GetFenceStatus = absl::bit_cast<PFN_vkGetFenceStatus>(
        next_get_device_proc_add_function(device, "vkGetFenceStatus"));

    dispatch_table.CreateEvent = absl::bit_cast<PFN_vkCreateEvent>(
        next_get_device_proc_add_function(device, "vkCreateEvent"));
    dispatch_table.DestroyEvent = absl::bit_cast<PFN_vkDestroyEvent>(
        next_get_device_proc_add_function(device, "vkDestroyEvent"));
    dispatch_table.SetEvent =
        absl::bit_cast<PFN_vkSetEvent>(next_get_device_proc_add_function(device, "vkSetEvent"));
    dispatch_table.GetEventStatus = absl::bit_cast<PFN_vkGetEventStatus>(
        next_get_device_proc_add_function(device, "vkGetEventStatus"));
    dispatch_table.CmdWaitEvents = absl::bit_cast<PFN_vkCmdWaitEvents>(
        next_get_device_proc_add_function(device, "vkCmdWaitEvents"));
    dispatch_table.CmdSetEvent = absl::bit_cast<PFN_vkCmdSetEvent>(
        next_get_device_proc_add_function(device, "vkCmdSetEvent"));

    dispatch_table.CmdPipelineBarrier = absl::bit_cast<PFN_vkCmdPipelineBarrier>(
        next_get_device_proc_add_function(device, "vkCmdPipelineBarrier"));

    dispatch_table.CmdBeginDebugUtilsLabelEXT = absl::bit_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        next_get_device_proc_add_function(device, "vkCmdBeginDebugUtilsLabelEXT"));
    dispatch_table.CmdEndDebugUtilsLabelEXT = absl::bit_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        next_get_device_proc_add_function(device, "vkCmdEndDebugUtilsLabelEXT"));

    dispatch_table.CmdDebugMarkerBeginEXT = absl::bit_cast<PFN_vkCmdDebugMarkerBeginEXT>(
        next_get_device_proc_add_function(device, "vkCmdDebugMarkerBeginEXT"));
    dispatch_table.CmdDebugMarkerEndEXT = absl::bit_cast<PFN_vkCmdDebugMarkerEndEXT>(
        next_get_device_proc_add_function(device, "vkCmdDebugMarkerEndEXT"));

    {
      absl::WriterMutexLock lock(&mutex_);
      device_dispatch_table_[GetDispatchTableKey(device)] = dispatch_table;
    }
  }

  void RemoveDeviceDispatchTable(const VkDevice& device) {
    absl::WriterMutexLock lock(&mutex_);
    device_dispatch_table_.erase(GetDispatchTableKey(device));
  }

  template <typename DispatchableType>
  PFN_vkDestroyDevice DestroyDevice(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).DestroyDevice;
  }

  template <typename DispatchableType>
  PFN_vkDestroyInstance DestroyInstance(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(instance_dispatch_table_.contains(key));
    return instance_dispatch_table_.at(key).DestroyInstance;
  }

  template <typename DispatchableType>
  PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties(
      const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(instance_dispatch_table_.contains(key));
    return instance_dispatch_table_.at(key).EnumerateDeviceExtensionProperties;
  }

  template <typename DispatchableType>
  PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties(
      const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(instance_dispatch_table_.contains(key));
    return instance_dispatch_table_.at(key).GetPhysicalDeviceQueueFamilyProperties;
  }

  template <typename DispatchableType>
  PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties(
      const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(instance_dispatch_table_.contains(key));
    return instance_dispatch_table_.at(key).GetPhysicalDeviceProperties;
  }

  template <typename DispatchableType>
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(instance_dispatch_table_.contains(key));
    return instance_dispatch_table_.at(key).GetInstanceProcAddr;
  }

  template <typename DispatchableType>
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetDeviceProcAddr;
  }

  template <typename DispatchableType>
  PFN_vkCreateCommandPool CreateCommandPool(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CreateCommandPool;
  }

  template <typename DispatchableType>
  PFN_vkDestroyCommandPool DestroyCommandPool(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).DestroyCommandPool;
  }

  template <typename DispatchableType>
  PFN_vkResetCommandPool ResetCommandPool(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).ResetCommandPool;
  }

  template <typename DispatchableType>
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).AllocateCommandBuffers;
  }

  template <typename DispatchableType>
  PFN_vkFreeCommandBuffers FreeCommandBuffers(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).FreeCommandBuffers;
  }

  template <typename DispatchableType>
  PFN_vkBeginCommandBuffer BeginCommandBuffer(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).BeginCommandBuffer;
  }

  template <typename DispatchableType>
  PFN_vkEndCommandBuffer EndCommandBuffer(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).EndCommandBuffer;
  }

  template <typename DispatchableType>
  PFN_vkResetCommandBuffer ResetCommandBuffer(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).ResetCommandBuffer;
  }

  template <typename DispatchableType>
  PFN_vkGetDeviceQueue GetDeviceQueue(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetDeviceQueue;
  }

  template <typename DispatchableType>
  PFN_vkGetDeviceQueue2 GetDeviceQueue2(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetDeviceQueue2;
  }

  template <typename DispatchableType>
  PFN_vkQueueSubmit QueueSubmit(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).QueueSubmit;
  }

  template <typename DispatchableType>
  PFN_vkQueuePresentKHR QueuePresentKHR(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).QueuePresentKHR;
  }

  template <typename DispatchableType>
  PFN_vkCreateQueryPool CreateQueryPool(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CreateQueryPool;
  }

  template <typename DispatchableType>
  PFN_vkCmdResetQueryPool CmdResetQueryPool(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdResetQueryPool;
  }

  template <typename DispatchableType>
  PFN_vkResetQueryPoolEXT ResetQueryPoolEXT(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).ResetQueryPoolEXT;
  }

  template <typename DispatchableType>
  PFN_vkGetQueryPoolResults GetQueryPoolResults(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetQueryPoolResults;
  }

  template <typename DispatchableType>
  PFN_vkCreateFence CreateFence(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CreateFence;
  }

  template <typename DispatchableType>
  PFN_vkDestroyFence DestroyFence(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).DestroyFence;
  }

  template <typename DispatchableType>
  PFN_vkGetFenceStatus GetFenceStatus(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetFenceStatus;
  }

  template <typename DispatchableType>
  PFN_vkCmdWriteTimestamp CmdWriteTimestamp(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdWriteTimestamp;
  }

  template <typename DispatchableType>
  PFN_vkCreateEvent CreateEvent(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CreateEvent;
  }

  template <typename DispatchableType>
  PFN_vkDestroyEvent DestroyEvent(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).DestroyEvent;
  }

  template <typename DispatchableType>
  PFN_vkSetEvent SetEvent(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).SetEvent;
  }

  template <typename DispatchableType>
  PFN_vkGetEventStatus GetEventStatus(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).GetEventStatus;
  }

  template <typename DispatchableType>
  PFN_vkCmdWaitEvents CmdWaitEvents(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdWaitEvents;
  }

  template <typename DispatchableType>
  PFN_vkCmdSetEvent CmdSetEvent(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdSetEvent;
  }

  template <typename DispatchableType>
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdPipelineBarrier;
  }

  template <typename DispatchableType>
  PFN_vkCmdBeginDebugUtilsLabelEXT CmdBeginDebugUtilsLabelEXT(
      const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdBeginDebugUtilsLabelEXT;
  }

  template <typename DispatchableType>
  PFN_vkCmdEndDebugUtilsLabelEXT CmdEndDebugUtilsLabelEXT(
      const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdEndDebugUtilsLabelEXT;
  }

  template <typename DispatchableType>
  PFN_vkCmdDebugMarkerBeginEXT CmdDebugMarkerBeginEXT(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdDebugMarkerBeginEXT;
  }

  template <typename DispatchableType>
  PFN_vkCmdDebugMarkerEndEXT CmdDebugMarkerEndEXT(const DispatchableType& dispatchable_object) {
    absl::ReaderMutexLock lock(&mutex_);
    void* key = GetDispatchTableKey(dispatchable_object);
    CHECK(device_dispatch_table_.contains(key));
    return device_dispatch_table_.at(key).CmdDebugMarkerEndEXT;
  }

 private:
  /*
   * In vulkan, every "dispatchable type" has as a very first field in memory a pointer to the
   * internal dispatch table. This pointer is unique per device/instance. So for example
   * for a command buffer allocated on a certain device, this pointer is the same for the buffer
   * and for the device. So we can use that pointer to uniquely map dispatchable types to their
   * dispatch table.
   */
  template <typename DispatchableType>
  void* GetDispatchTableKey(DispatchableType dispatchable_object) {
    return *absl::bit_cast<void**>(dispatchable_object);
  }

  // Dispatch tables required for routing instance and device calls onto the next
  // layer in the dispatch chain among our handling of functions we intercept.
  absl::flat_hash_map<void*, VkLayerInstanceDispatchTable> instance_dispatch_table_;
  absl::flat_hash_map<void*, VkLayerDispatchTable> device_dispatch_table_;

  // Must protect access to dispatch tables above by mutex since the Vulkan
  // application may be calling these functions from different threads.
  // However, they are usually filled once (per device/instance) at the beginning
  // and afterwards we only read that data. So we use a read/write lock.
  absl::Mutex mutex_;
};

}  // namespace orbit_vulkan_layer

#endif  // ORBIT_VULKAN_LAYER_DISPATCH_TABLE_H_
