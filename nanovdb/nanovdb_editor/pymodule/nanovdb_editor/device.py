# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from ctypes import *

from .utils import load_library

COMPUTE_LIB = "pnanovdbcompute"


class LogLevel(c_uint32):
    ERROR = 0
    WARNING = 1
    INFO = 2


LOG_LEVEL = LogLevel.WARNING
LOG_FUNC = CFUNCTYPE(None, LogLevel, c_char_p)


def pnanovdb_compute_log_print(level, format_str):
    """
    Does not support the variadic arguments of the original function.
    """
    level_value = level.value
    if level_value <= LOG_LEVEL:
        prefix = {0: "Error", 1: "Warning", 2: "Info"}.get(level_value, "Unknown")
        print(f"{prefix}: {format_str.decode('utf-8')}")


class pnanovdb_DeviceDesc(Structure):
    """Definition equivalent to pnanovdb_compute_device_desc_t."""
    _fields_ = [
        ("device_index", c_uint32),
        ("enable_external_usage", c_bool),
        ("log_print", LOG_FUNC)
    ]


class pnanovdb_Device(Structure):
    """Definition equivalent to pnanovdb_compute_device_t."""


class pnanovdb_DeviceManager(Structure):
    """Definition equivalent to pnanovdb_compute_device_manager_t."""


class pnanovdb_ComputeQueue(Structure):
    """Definition equivalent to pnanovdb_compute_queue_t."""


class pnanovdb_DeviceInterface(Structure):
    """Definition equivalent to pnanovdb_compute_device_interface_t."""
    _fields_ = [
        ("device_interface", c_void_p),
        ("create_device_manager", CFUNCTYPE(POINTER(pnanovdb_DeviceManager), c_bool)),
        ("destroy_device_manager",
            CFUNCTYPE(None, POINTER(pnanovdb_DeviceManager))),
        ("enumerate_devices",
            CFUNCTYPE(c_bool, POINTER(pnanovdb_DeviceManager), c_uint32,
                     POINTER(c_void_p))),
        ("create_device",
            CFUNCTYPE(POINTER(pnanovdb_Device), POINTER(pnanovdb_DeviceManager),
                     POINTER(pnanovdb_DeviceDesc))),
        ("destroy_device",
            CFUNCTYPE(None, POINTER(pnanovdb_DeviceManager),
                     POINTER(pnanovdb_Device))),
        ("get_memory_stats",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), POINTER(c_void_p))),
        ("create_semaphore",
            CFUNCTYPE(POINTER(c_void_p), POINTER(pnanovdb_Device))),
        ("destroy_semaphore", CFUNCTYPE(None, POINTER(c_void_p))),
        ("get_semaphore_external_handle",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), POINTER(c_void_p),
                     c_uint64)),
        ("close_semaphore_external_handle",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), POINTER(c_void_p),
                     c_uint64)),
        ("get_device_queue", CFUNCTYPE(POINTER(pnanovdb_ComputeQueue), POINTER(pnanovdb_Device))),
        ("get_compute_queue", CFUNCTYPE(POINTER(pnanovdb_ComputeQueue), POINTER(pnanovdb_Device))),
        ("flush",
            CFUNCTYPE(c_int, POINTER(pnanovdb_Device), POINTER(c_uint64),
                     POINTER(c_void_p), POINTER(c_void_p))),
        ("get_frame_global_completed",
            CFUNCTYPE(c_uint64, POINTER(pnanovdb_Device))),
        ("wait_for_frame",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), c_uint64)),
        ("wait_idle", CFUNCTYPE(None, POINTER(pnanovdb_Device))),
        ("get_compute_interface", CFUNCTYPE(c_void_p, c_void_p)),
        ("get_compute_context", CFUNCTYPE(c_void_p, c_void_p)),
        ("create_swapchain",
            CFUNCTYPE(c_void_p, POINTER(pnanovdb_Device), POINTER(c_void_p))),
        ("destroy_swapchain", CFUNCTYPE(None, POINTER(pnanovdb_Device))),
        ("resize_swapchain",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), c_uint32, c_uint32)),
        ("present_swapchain",
            CFUNCTYPE(c_int, POINTER(pnanovdb_Device), c_bool,
                     POINTER(c_uint64))),
        ("get_swapchain_front_texture",
            CFUNCTYPE(c_void_p, POINTER(pnanovdb_Device))),
        ("enable_profiler",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), POINTER(c_void_p),
                     CFUNCTYPE(None, POINTER(pnanovdb_Device), c_uint64,
                              c_uint32, POINTER(c_void_p)))),
        ("disable_profiler", CFUNCTYPE(None, POINTER(pnanovdb_Device))),
        ("set_resource_min_lifetime",
            CFUNCTYPE(None, POINTER(pnanovdb_Device), c_uint64))
    ]


class DeviceInterface:
    """Python wrapper for pnanovdb_compute_device_interface_t."""
    def __init__(self, api: int):
        lib = load_library(COMPUTE_LIB)

        get_device_interface = lib.pnanovdb_get_compute_device_interface
        get_device_interface.restype = POINTER(pnanovdb_DeviceInterface)
        get_device_interface.argtypes = []

        self._device_interface = get_device_interface(api)
        if not self._device_interface:
            raise RuntimeError("Failed to get device interface")

        self._device_manager = None
        self._devices = []

        self._callback_func = LOG_FUNC(pnanovdb_compute_log_print)

    def get_device_interface(self) -> POINTER(pnanovdb_DeviceInterface):
        return self._device_interface

    def get_device(self, index=0) -> POINTER(pnanovdb_Device):
        if len(self._devices) == 0:
            raise RuntimeError("Device not created")

        return self._devices[index]

    def create_device_manager(self, enable_validation=False) -> None:
        create_func = self._device_interface.contents.create_device_manager
        self._device_manager = create_func(enable_validation)
        if not self._device_manager:
            raise RuntimeError("Failed to create device manager")

    def create_device(self, device_index=0, enable_external_usage=False) -> POINTER(pnanovdb_Device):
        if not self._device_manager:
            raise RuntimeError("Device manager not created")

        desc = pnanovdb_DeviceDesc(device_index=device_index,
                                   enable_external_usage=enable_external_usage,
                                   log_print=self._callback_func)

        create_func = self._device_interface.contents.create_device
        device = create_func(self._device_manager, byref(desc))
        if not device:
            raise RuntimeError("Failed to create device")
        self._devices.append(device)

        return device

    def get_compute_queue(self, device: pnanovdb_Device) -> POINTER(pnanovdb_ComputeQueue):
        if not device:
            raise RuntimeError("Device does not exist")

        get_compute_queue = self._device_interface.contents.get_compute_queue
        return get_compute_queue(device)

    def __del__(self):
        for device in self._devices:
            destroy_func = self._device_interface.contents.destroy_device
            destroy_func(self._device_manager, device)

        if self._device_manager:
            destroy_func = self._device_interface.contents.destroy_device_manager
            destroy_func(self._device_manager)

            self._device_manager = None

        self._device_interface = None

