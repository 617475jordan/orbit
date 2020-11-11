// Copyright (c) 2020 The Orbit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "LayerLogic.h"

#include "OrbitBase/Logging.h"

namespace orbit_vulkan_layer {

VkResult LayerLogic::PreCallAndCallCreateInstance(const VkInstanceCreateInfo* create_info,
                                                  const VkAllocationCallbacks* allocator,
                                                  VkInstance* instance) {
  LOG("PreCallAndCallCreateInstance");
  auto* layer_create_info = absl::bit_cast<VkLayerInstanceCreateInfo*>(create_info->pNext);

  while (layer_create_info != nullptr &&
         (layer_create_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
          layer_create_info->function != VK_LAYER_LINK_INFO)) {
    layer_create_info = absl::bit_cast<VkLayerInstanceCreateInfo*>(layer_create_info->pNext);
  }

  if (layer_create_info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PFN_vkGetInstanceProcAddr next_get_instance_proc_addr_function =
      layer_create_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;

  // Advance linkage for next layer
  layer_create_info->u.pLayerInfo = layer_create_info->u.pLayerInfo->pNext;

  // Need to call vkCreateInstance down the chain to actually create the
  // instance, as we need it to be alive in the create instance dispatch table.
  auto create_instance = absl::bit_cast<PFN_vkCreateInstance>(
      next_get_instance_proc_addr_function(VK_NULL_HANDLE, "vkCreateInstance"));
  VkResult result = create_instance(create_info, allocator, instance);

  dispatch_table_.CreateInstanceDispatchTable(*instance, next_get_instance_proc_addr_function);

  return result;
}

void LayerLogic::PostCallCreateInstance(const VkInstanceCreateInfo* /*create_info*/,
                                        const VkAllocationCallbacks* /*allocator*/,
                                        VkInstance* /*instance*/) {
  LOG("PostCallCreateInstance");
}

void LayerLogic::PostCallDestroyInstance(VkInstance instance,
                                         const VkAllocationCallbacks* /*allocator*/) {
  LOG("PostCallDestroyInstance");
  dispatch_table_.RemoveInstanceDispatchTable(instance);
}

VkResult LayerLogic::PreCallAndCallCreateDevice(
    VkPhysicalDevice /*physical_device*/ physical_device, const VkDeviceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator /*allocator*/, VkDevice* device) {
  LOG("PreCallAndCallCreateDevice");
  auto* layer_create_info = absl::bit_cast<VkLayerDeviceCreateInfo*>(create_info->pNext);

  while (layer_create_info != nullptr &&
         (layer_create_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
          layer_create_info->function != VK_LAYER_LINK_INFO)) {
    layer_create_info = absl::bit_cast<VkLayerDeviceCreateInfo*>(layer_create_info->pNext);
  }

  if (layer_create_info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PFN_vkGetInstanceProcAddr next_get_instance_proc_addr_function =
      layer_create_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr next_get_device_proc_addr_function =
      layer_create_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;

  // Advance linkage for next layer
  layer_create_info->u.pLayerInfo = layer_create_info->u.pLayerInfo->pNext;

  // Need to call vkCreateInstance down the chain to actually create the
  // instance, as we need it to be alive in the create instance dispatch table.
  auto create_device_function = absl::bit_cast<PFN_vkCreateDevice>(
      next_get_instance_proc_addr_function(VK_NULL_HANDLE, "vkCreateDevice"));
  VkResult result = create_device_function(physical_device, create_info, allocator, device);

  dispatch_table_.CreateDeviceDispatchTable(*device, next_get_device_proc_addr_function);

  return result;
}

void LayerLogic::PostCallCreateDevice(VkPhysicalDevice physical_device,
                                      const VkDeviceCreateInfo* /*create_info*/,
                                      const VkAllocationCallbacks* /*allocator*/,
                                      VkDevice* device) {
  LOG("PostCallCreateDevice");
  physical_device_manager_.TrackPhysicalDevice(physical_device, *device);
  queue_family_info_manager_.InitializeQueueFamilyInfo(physical_device);
  timer_query_pool_.InitializeTimerQueryPool(*device, physical_device);
}

void LayerLogic::PostCallDestroyDevice(VkDevice device,
                                       const VkAllocationCallbacks* /*allocator*/) {
  LOG("PostCallDestroyDevice");
  queue_family_info_manager_.RemoveQueueFamilyInfo(
      physical_device_manager_.GetPhysicalDeviceOfLogicalDevice(device));
  physical_device_manager_.UntrackLogicalDevice(device);
  dispatch_table_.RemoveDeviceDispatchTable(device);
}

void LayerLogic::PostCallResetCommandPool(VkDevice /*device*/, VkCommandPool command_pool,
                                          VkCommandPoolResetFlags /*flags*/) {
  LOG("PostCallResetCommandPool");
  command_buffer_manager_.ResetCommandPool(command_pool);
}

void LayerLogic::PostCallAllocateCommandBuffers(VkDevice device,
                                                const VkCommandBufferAllocateInfo* allocate_info,
                                                VkCommandBuffer* command_buffers) {
  LOG("PostCallAllocateCommandBuffers");
  const VkCommandPool& pool = allocate_info->commandPool;
  const uint32_t command_buffer_count = allocate_info->commandBufferCount;
  command_buffer_manager_.TrackCommandBuffers(device, pool, command_buffers, command_buffer_count);
}

void LayerLogic::PostCallFreeCommandBuffers(VkDevice device, VkCommandPool command_pool,
                                            uint32_t command_buffer_count,
                                            const VkCommandBuffer* command_buffers) {
  LOG("PostCallFreeCommandBuffers");
  command_buffer_manager_.UntrackCommandBuffers(device, command_pool, command_buffers,
                                                command_buffer_count);
}

void LayerLogic::PostCallBeginCommandBuffer(VkCommandBuffer command_buffer,
                                            const VkCommandBufferBeginInfo* /*begin_info*/) {
  LOG("PostCallBeginCommandBuffer");
  command_buffer_manager_.MarkCommandBufferBegin(command_buffer);
}

void LayerLogic::PreCallEndCommandBuffer(VkCommandBuffer command_buffer) {
  LOG("PreCallEndCommandBuffer");
  command_buffer_manager_.MarkCommandBufferEnd(command_buffer);
}

void LayerLogic::PreCallResetCommandBuffer(VkCommandBuffer command_buffer,
                                           VkCommandBufferResetFlags /*flags*/) {
  command_buffer_manager_.ResetCommandBuffer(command_buffer);
  LOG("PreCallResetCommandBuffer");
}

void LayerLogic::PreCallQueueSubmit(VkQueue queue, uint32_t submit_count,
                                    const VkSubmitInfo* submits, VkFence /*fence*/) {
  LOG("PostCallQueueSubmit");
  command_buffer_manager_.DoPreSubmitQueue(queue, submit_count, submits);
}
void LayerLogic::PostCallQueueSubmit(VkQueue queue, uint32_t /*submit_count*/,
                                     const VkSubmitInfo* /*submits*/, VkFence /*fence*/) {
  LOG("PostCallQueueSubmit");
  command_buffer_manager_.DoPostSubmitQueue(queue);
}

void LayerLogic::PostCallQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* /*present_info*/) {
  LOG("PostCallQueuePresentKHR");
  command_buffer_manager_.CompleteSubmits(queue_manager_.GetDeviceOfQueue(queue));
}

void LayerLogic::PostCallGetDeviceQueue(VkDevice device, uint32_t /*queue_family_index*/,
                                        uint32_t /*queue_index*/, VkQueue* queue) {
  LOG("PostCallGetDeviceQueue");
  queue_manager_.TrackQueue(*queue, device);
}
void LayerLogic::PostCallGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* /*queue_info*/,
                                         VkQueue* queue) {
  LOG("PostCallGetDeviceQueue2");
  queue_manager_.TrackQueue(*queue, device);
}
void LayerLogic::PostCallCmdBeginDebugUtilsLabelEXT(VkCommandBuffer /*command_buffer*/,
                                                    const VkDebugUtilsLabelEXT* /*label_info*/) {
  LOG("PostCallCmdBeginDebugUtilsLabelEXT");
}
void LayerLogic::PreCallCmdEndDebugUtilsLabelEXT(VkCommandBuffer /*command_buffer*/) {
  LOG("PreCallCmdEndDebugUtilsLabelEXT");
}

void LayerLogic::PostCallCmdDebugMarkerBeginEXT(VkCommandBuffer /*command_buffer*/,
                                                const VkDebugMarkerMarkerInfoEXT* /*marker_info*/) {
  LOG("PostCallCmdDebugMarkerBeginEXT");
}

void LayerLogic::PreCallCmdDebugMarkerEndEXT(VkCommandBuffer /*command_buffer*/) {
  LOG("PreCallCmdDebugMarkerEndEXT");
}

}  // namespace orbit_vulkan_layer