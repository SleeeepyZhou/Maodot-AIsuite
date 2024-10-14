#include "deviceinfo.h"

#include "ggml-kompute.h"
#include "util.h"

DeviceInfo::DeviceInfo() {
    physical_Core = get_num_physical_cores();
    get_vk_available_devices();
}

DeviceInfo::get_vk_available_devices() {
    size_t device_count = 0;
    ggml_vk_device *devices = ggml_vk_available_devices(1024 * 1024 * 1024, &device_count); 
    if (devices != nullptr) {
        for (size_t i = 0; i < device_count; ++i) {
            Dictionary vk_device;
            vk_device["index"] = devices[i].index;
            vk_device["type"] = devices[i].type;
            vk_device["heapSize"] = devices[i].heapSize;
            vk_device["vendor"] = String(devices[i].vendor);
            vk_device["subgroupSize"] = devices[i].subgroupSize;
            vk_device["bufferAlignment"] = devices[i].bufferAlignment;
            vk_device["maxAlloc"] = devices[i].maxAlloc;
            vk_device["name"] = String(devices[i].name);
            vk_available_devices.push_back(vk_device);
            vk_devices_idx.push_back(devices[i].index);
        }
        free(devices);
    } else {
        ERR_PRINT(vformat("No Vulkan devices available or insufficient memory."));
    }
}
