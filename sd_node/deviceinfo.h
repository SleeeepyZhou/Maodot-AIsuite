/*       By SleeeepyZhou        */

#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#include "core/object/object.h"

class DeviceInfo {

public:
    static DeviceInfo& getInstance() {
        static DeviceInfo instance;
        return instance;
    }

    int get_core_count() const { return physical_Core; };
    const Array get_available_devices() const { return vk_available_devices;};
    const Array get_devices_idx() const { return vk_devices_idx;};
    void get_vk_available_devices();

private:
    DeviceInfo();
    ~DeviceInfo() = default;

    int physical_Core;
    static Array vk_available_devices;
    static Array vk_devices_idx;

};


#endif // DEVICE_INFO_H
