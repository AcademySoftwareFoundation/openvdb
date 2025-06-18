
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   DeviceVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#include <vector>
#include <algorithm>
#include <memory>

namespace pnanovdb_vulkan
{
    static const pnanovdb_uint32_t kMaxFramesInFlight = 3u;
    static const pnanovdb_uint32_t kMaxSwapchainImages = 8u;

    struct DeviceManager;
    struct Device;
    struct DeviceSemaphore;
    struct DeviceQueue;
    struct Swapchain;
    struct Context;

    struct Buffer;
    struct BufferTransient;
    struct BufferAcquire;
    struct Texture;
    struct TextureTransient;
    struct TextureAcquire;
    struct Sampler;

    struct ComputePipeline;

    PNANOVDB_CAST_PAIR(pnanovdb_compute_device_manager_t, DeviceManager)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_device_t, Device)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_semaphore_t, DeviceSemaphore)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_queue_t, DeviceQueue)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_swapchain_t, Swapchain)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_context_t, Context)

    PNANOVDB_CAST_PAIR(pnanovdb_compute_buffer_t, Buffer)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_buffer_transient_t, BufferTransient)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_buffer_acquire_t, BufferAcquire)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_texture_t, Texture)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_texture_transient_t, TextureTransient)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_texture_acquire_t, TextureAcquire)
    PNANOVDB_CAST_PAIR(pnanovdb_compute_sampler_t, Sampler)

    PNANOVDB_CAST_PAIR(pnanovdb_compute_pipeline_t, ComputePipeline)

    struct Fence;
    struct FormatConverter;

    struct DeviceManager
    {
        void* vulkan_module = nullptr;

        VkInstance vulkanInstance = nullptr;

        std::vector<VkPhysicalDevice> physicalDevices;
        std::vector<VkPhysicalDeviceProperties> deviceProps;
        std::vector<pnanovdb_compute_physical_device_desc_t> physicalDeviceDescs;

        pnanovdb_vulkan_enabled_instance_extensions_t enabledExtensions = { };
        pnanovdb_vulkan_instance_loader_t loader = {};
    };

    pnanovdb_compute_device_manager_t* createDeviceManager(pnanovdb_bool_t enableValidationOnDebugBuild);
    void destroyDeviceManager(pnanovdb_compute_device_manager_t* manager);
    pnanovdb_bool_t enumerateDevices(pnanovdb_compute_device_manager_t* manager, pnanovdb_uint32_t deviceIndex, pnanovdb_compute_physical_device_desc_t* pDesc);

    struct Device
    {
        pnanovdb_compute_device_desc_t desc = {};
        pnanovdb_compute_log_print_t logPrint = nullptr;

        DeviceManager* deviceManager = nullptr;
        FormatConverter* formatConverter = nullptr;

        VkDevice vulkanDevice = nullptr;

        VkPhysicalDevice physicalDevice = nullptr;
        VkPhysicalDeviceProperties physicalDeviceProperties = {};
        VkPhysicalDeviceMemoryProperties memoryProperties = {};

        uint32_t graphicsQueueFamilyIdx = 0u;
        VkQueue graphicsQueueVk = nullptr;
        uint32_t computeQueueFamilyIdx = 0u;
        VkQueue computeQueueVk = nullptr;

        DeviceQueue* deviceQueue = nullptr;
        DeviceQueue* computeQueue = nullptr;

        pnanovdb_vulkan_enabled_features_t enabledFeatures = {};
        pnanovdb_vulkan_enabled_device_extensions_t enabledExtensions = { };
        pnanovdb_vulkan_device_loader_t loader = {};

        pnanovdb_compute_device_memory_stats_t memoryStats = {};
    };

    pnanovdb_compute_device_t* createDevice(pnanovdb_compute_device_manager_t* deviceManager, const pnanovdb_compute_device_desc_t* desc);
    void destroyDevice(pnanovdb_compute_device_manager_t* deviceManager, pnanovdb_compute_device_t* device);
    pnanovdb_compute_queue_t* getDeviceQueue(const pnanovdb_compute_device_t* device);
    pnanovdb_compute_queue_t* getComputeQueue(const pnanovdb_compute_device_t* device);
    void getMemoryStats(pnanovdb_compute_device_t* device, pnanovdb_compute_device_memory_stats_t* dstStats);

    void device_reportMemoryAllocate(Device* device, pnanovdb_compute_memory_type_t type, pnanovdb_uint64_t bytes);
    void device_reportMemoryFree(Device* device, pnanovdb_compute_memory_type_t type, pnanovdb_uint64_t bytes);

    struct DeviceSemaphore
    {
        Device* device = nullptr;

        VkSemaphore semaphoreVk;
    };

    pnanovdb_compute_semaphore_t* createSemaphore(pnanovdb_compute_device_t* device);
    void destroySemaphore(pnanovdb_compute_semaphore_t* semaphore);
    void getSemaphoreExternalHandle(pnanovdb_compute_semaphore_t* semaphore, void* dstHandle, pnanovdb_uint64_t dstHandleSize);
    void closeSemaphoreExternalHandle(pnanovdb_compute_semaphore_t* semaphore, const void* srcHandle, pnanovdb_uint64_t srcHandleSize);

    struct Fence
    {
        VkFence fence;
        pnanovdb_bool_t active;
        pnanovdb_uint64_t value;
    };

    struct DeviceQueue
    {
        Device* device = nullptr;

        VkDevice vulkanDevice = nullptr;
        pnanovdb_uint32_t queueFamilyIdx = 0u;
        VkQueue queueVk = nullptr;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        int commandBufferIdx = 0u;
        VkCommandBuffer commandBuffers[kMaxFramesInFlight] = { nullptr };
        Fence fences[kMaxFramesInFlight] = { {VK_NULL_HANDLE, 0, 0u} };
        VkCommandBuffer commandBuffer = nullptr;

        VkSemaphore beginFrameSemaphore = VK_NULL_HANDLE;
        VkSemaphore endFrameSemaphore = VK_NULL_HANDLE;

        VkSemaphore currentBeginFrameSemaphore = VK_NULL_HANDLE;
        VkSemaphore currentEndFrameSemaphore = VK_NULL_HANDLE;

        pnanovdb_uint64_t lastFenceCompleted = 1u;
        pnanovdb_uint64_t nextFenceValue = 2u;

        Context* context = nullptr;
    };

    int flush(pnanovdb_compute_queue_t* ptr, pnanovdb_uint64_t* flushedFrameID, pnanovdb_compute_semaphore_t* waitSemaphore, pnanovdb_compute_semaphore_t* signalSemaphore);
    pnanovdb_uint64_t getLastFrameCompleted(pnanovdb_compute_queue_t* queue);
    void waitForFrame(pnanovdb_compute_queue_t* ptr, pnanovdb_uint64_t frameID);
    void waitIdle(pnanovdb_compute_queue_t* ptr);
    pnanovdb_compute_interface_t* getContextInterface(const pnanovdb_compute_queue_t* ptr);
    pnanovdb_compute_context_t* getContext(const pnanovdb_compute_queue_t* ptr);

    DeviceQueue* deviceQueue_create(Device* device, uint32_t queueFamilyIdx, VkQueue queue);
    void deviceQueue_destroy(DeviceQueue* deviceQueue);
    int flushStepA(DeviceQueue* ptr, pnanovdb_compute_semaphore_t* waitSemaphore, pnanovdb_compute_semaphore_t* signalSemaphore);
    void flushStepB(DeviceQueue* ptr);
    void deviceQueue_fenceUpdate(DeviceQueue* ptr, pnanovdb_uint32_t fenceIdx, pnanovdb_bool_t blocking);

    struct Swapchain
    {
        pnanovdb_compute_swapchain_desc_t desc = {};

        DeviceQueue* deviceQueue = nullptr;

#if defined(_WIN32)
#elif defined(__APPLE__)
#else
        void* moduleX11 = nullptr;
        decltype(&XGetWindowAttributes) getWindowAttrib = nullptr;
#endif

        pnanovdb_bool_t valid = PNANOVDB_FALSE;

        pnanovdb_bool_t vsync = PNANOVDB_TRUE;
        int width = 0;
        int height = 0;

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;

        VkSurfaceKHR surfaceVulkan = VK_NULL_HANDLE;
        VkSwapchainKHR swapchainVulkan = VK_NULL_HANDLE;
        unsigned int numSwapchainImages = 0u;
        VkImage swapchainImages[kMaxSwapchainImages] = { VK_NULL_HANDLE };
        unsigned int currentSwapchainIdx = 0u;

        pnanovdb_compute_texture_t* textures[kMaxSwapchainImages] = { };
    };

    pnanovdb_compute_swapchain_t* createSwapchain(pnanovdb_compute_queue_t* queue, const pnanovdb_compute_swapchain_desc_t* desc);
    void destroySwapchain(pnanovdb_compute_swapchain_t* swapchain);
    void resizeSwapchain(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_uint32_t width, pnanovdb_uint32_t height);
    int presentSwapchain(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_bool_t vsync, pnanovdb_uint64_t* flushedFrameID);
    pnanovdb_compute_texture_t* getSwapchainFrontTexture(pnanovdb_compute_swapchain_t* swapchain);

    void swapchain_getWindowSize(Swapchain* ptr, pnanovdb_uint32_t* width, pnanovdb_uint32_t* height);
    void swapchain_initSwapchain(Swapchain* ptr);
    void swapchain_destroySwapchain(Swapchain* ptr);

    struct Context;

    struct Buffer
    {
        int refCount = 0;
        pnanovdb_uint64_t lastActive = PNANOVDB_FALSE;
        pnanovdb_compute_buffer_desc_t desc = {};
        pnanovdb_compute_memory_type_t memory_type = PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE;
        pnanovdb_uint64_t allocationBytes = 0llu;

        VkDeviceMemory memoryVk = VK_NULL_HANDLE;
        VkBuffer bufferVk = VK_NULL_HANDLE;
        VkBufferView bufferViewVk = VK_NULL_HANDLE;
        std::vector<VkBufferView> aliasBufferViews;
        std::vector<pnanovdb_compute_format_t> aliasFormats;
        void* mappedData = nullptr;

        VkBufferMemoryBarrier restoreBarrier = {};
        VkBufferMemoryBarrier currentBarrier = {};
    };

    struct BufferTransient
    {
        pnanovdb_compute_buffer_desc_t desc = {};
        Buffer* buffer = nullptr;
        BufferTransient* aliasBuffer = nullptr;
        pnanovdb_compute_format_t aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        int nodeBegin = 0;
        int nodeEnd = 0;
    };

    struct BufferAcquire
    {
        BufferTransient* bufferTransient = nullptr;
        Buffer* buffer = nullptr;
    };

    pnanovdb_compute_buffer_t* createBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc);
    void destroyBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);
    pnanovdb_compute_buffer_transient_t* getBufferTransient(pnanovdb_compute_context_t* context, const pnanovdb_compute_buffer_desc_t* desc);
    pnanovdb_compute_buffer_transient_t* registerBufferAsTransient(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);
    pnanovdb_compute_buffer_transient_t* aliasBufferTransient(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_transient_t* buffer, pnanovdb_compute_format_t format, pnanovdb_uint32_t structureStride);
    pnanovdb_compute_buffer_acquire_t* enqueueAcquireBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_transient_t* buffer);
    pnanovdb_bool_t getAcquiredBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_acquire_t* acquire, pnanovdb_compute_buffer_t** outBuffer);
    void* mapBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);
    void unmapBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);
    void getBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, pnanovdb_compute_interop_handle_t* dstHandle);
    void closeBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, const pnanovdb_compute_interop_handle_t* srcHandle);
    pnanovdb_compute_buffer_t* createBufferFromExternalHandle(pnanovdb_compute_context_t* context, const pnanovdb_compute_buffer_desc_t* desc, const pnanovdb_compute_interop_handle_t* interopHandle);
    void device_getBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, void* dstHandle, pnanovdb_uint64_t dstHandleSize, pnanovdb_uint64_t* pBufferSizeInBytes);
    void device_closeBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, const void* srcHandle, pnanovdb_uint64_t srcHandleSize);

    Buffer* buffer_create(Context* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc, const pnanovdb_compute_interop_handle_t* interopHandle);
    void buffer_destroy(Context* context, Buffer* buffer);
    void context_destroyBuffers(Context* context);
    VkBufferView buffer_getBufferView(Context* context, Buffer* ptr, pnanovdb_compute_format_t aliasFormat);

    struct Texture
    {
        int refCount = 0;
        pnanovdb_uint64_t lastActive = PNANOVDB_FALSE;
        pnanovdb_compute_texture_desc_t desc = {};
        VkImageAspectFlags imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;
        pnanovdb_uint64_t allocationBytes = 0llu;

        VkDeviceMemory memoryVk = VK_NULL_HANDLE;
        VkImage imageVk = VK_NULL_HANDLE;
        VkImageView imageViewVk_mipLevel = VK_NULL_HANDLE;
        VkImageView imageViewVk_all = VK_NULL_HANDLE;
        std::vector<VkImageView> aliasImageViewAlls;
        std::vector<VkImageView> aliasImageViewMipLevels;
        std::vector<pnanovdb_compute_format_t> aliasFormats;

        VkImageMemoryBarrier restoreBarrier = {};
        VkImageMemoryBarrier currentBarrier = {};
    };

    struct TextureTransient
    {
        pnanovdb_compute_texture_desc_t desc = {};
        Texture* texture = nullptr;
        TextureTransient* aliasTexture = nullptr;
        pnanovdb_compute_format_t aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        int nodeBegin = 0;
        int nodeEnd = 0;
    };

    struct TextureAcquire
    {
        TextureTransient* textureTransient = nullptr;
        Texture* texture = nullptr;
    };

    pnanovdb_compute_texture_t* createTexture(pnanovdb_compute_context_t* context, const pnanovdb_compute_texture_desc_t* desc);
    pnanovdb_compute_texture_t* createTextureExternal(pnanovdb_compute_context_t* context, const pnanovdb_compute_texture_desc_t* desc, VkImage externalImage, VkImageLayout defaultLayout);
    void destroyTexture(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_t* texture);
    pnanovdb_compute_texture_transient_t* getTextureTransient(pnanovdb_compute_context_t* context, const pnanovdb_compute_texture_desc_t* desc);
    pnanovdb_compute_texture_transient_t* registerTextureAsTransient(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_t* texture);
    pnanovdb_compute_texture_transient_t* aliasTextureTransient(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_transient_t* texture, pnanovdb_compute_format_t format);
    pnanovdb_compute_texture_acquire_t* enqueueAcquireTexture(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_transient_t* texture);
    pnanovdb_bool_t getAcquiredTexture(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_acquire_t* acquire, pnanovdb_compute_texture_t** outTexture);

    Texture* texture_create(Context* context, const pnanovdb_compute_texture_desc_t* desc);
    Texture* texture_createExternal(Context* context, const pnanovdb_compute_texture_desc_t* desc, VkImage externalImage, VkImageLayout defaultLayout);
    void texture_destroy(Context* context, Texture* texture);
    void context_destroyTextures(Context* context);
    VkImageView texture_getImageViewAll(Context* context, Texture* ptr, pnanovdb_compute_format_t aliasFormat);
    VkImageView texture_getImageViewMipLevel(Context* context, Texture* ptr,  pnanovdb_compute_format_t aliasFormat);

    struct Sampler
    {
        pnanovdb_bool_t isActive = PNANOVDB_FALSE;
        pnanovdb_uint64_t lastActive = PNANOVDB_FALSE;
        VkSampler sampler = VK_NULL_HANDLE;
    };

    pnanovdb_compute_sampler_t* createSampler(pnanovdb_compute_context_t* context, const pnanovdb_compute_sampler_desc_t* desc);
    pnanovdb_compute_sampler_t* getDefaultSampler(pnanovdb_compute_context_t* context);
    void destroySampler(pnanovdb_compute_context_t* context, pnanovdb_compute_sampler_t* sampler);

    Sampler* sampler_create(Context* context, const pnanovdb_compute_sampler_desc_t* desc);
    void sampler_destroy(Context* context, Sampler* sampler);
    void context_destroySamplers(Context* context);

    struct DescriptorPool
    {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        pnanovdb_uint32_t allocSetIdx = 0u;
        pnanovdb_uint64_t fenceValue = 0llu;
    };

    struct ComputePipeline
    {
        pnanovdb_compute_pipeline_desc_t desc = {};
        pnanovdb_uint32_t totalDescriptors = 0u;

        VkShaderModule module = VK_NULL_HANDLE;

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        std::vector<DescriptorPool> pools;
        pnanovdb_uint64_t frontIdx = 0u;

        std::vector<VkDescriptorSetLayoutBinding> bindings;

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        std::vector<VkDescriptorBufferInfo> bufferInfos;
        std::vector<VkBufferView> bufferViews;
        std::vector<VkDescriptorImageInfo> imageInfos;

        pnanovdb_uint32_t poolSizeCount = 0u;
        pnanovdb_uint32_t setsPerPool = 0u;

        VkDescriptorType pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_COUNT] = {};
        pnanovdb_uint32_t descriptorCounts[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_COUNT] = {};
        VkDescriptorPoolSize poolSizes[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_COUNT] = {};
    };

    pnanovdb_compute_pipeline_t* createComputePipeline(pnanovdb_compute_context_t* context, const pnanovdb_compute_pipeline_desc_t* desc);
    void destroyComputePipeline(pnanovdb_compute_context_t* context, pnanovdb_compute_pipeline_t* pipeline);

    VkDescriptorSet computePipeline_allocate(Context* context, ComputePipeline* pipeline);
    void computePipeline_dispatch(Context* context, const pnanovdb_compute_dispatch_params_t* params);

    struct ProfilerEntry
    {
        const char* label;
        pnanovdb_uint64_t cpuValue;
        pnanovdb_uint64_t gpuValue;
    };

    struct ProfilerCapture
    {
        VkQueryPool queryPool = VK_NULL_HANDLE;
        VkBuffer queryBuffer = VK_NULL_HANDLE;
        VkDeviceMemory queryMemory = VK_NULL_HANDLE;
        pnanovdb_uint64_t* queryMapped = nullptr;
        pnanovdb_uint64_t queryFrequency = 0u;
        pnanovdb_uint64_t queryReadbackFenceVal = ~0llu;

        pnanovdb_uint32_t state = 0u;
        pnanovdb_uint64_t captureID = 0llu;
        pnanovdb_uint64_t cpuFreq = 0llu;
        pnanovdb_uint64_t capacity = 0u;

        std::vector<ProfilerEntry> entries;
        std::vector<pnanovdb_compute_profiler_entry_t> deltaEntries;
    };

    void profilerCapture_init(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t capacity);
    void profilerCapture_destroy(Context* context, ProfilerCapture* ptr);
    void profilerCapture_reset(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t minCapacity, pnanovdb_uint64_t captureID);
    void profilerCapture_timestamp(Context* context, ProfilerCapture* ptr, const char* label);
    void profilerCapture_download(Context* context, ProfilerCapture* ptr);
    pnanovdb_bool_t profilerCapture_mapResults(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t* pNumEntries, pnanovdb_compute_profiler_entry_t** pEntries);
    void profilerCapture_unmapResults(Context* context, ProfilerCapture* ptr);

    struct Profiler
    {
        std::vector<ProfilerCapture> captures;
        pnanovdb_uint64_t currentCaptureIndex = 0u;
        pnanovdb_uint64_t currentCaptureID = 0llu;

        void* userdata = nullptr;
        void(PNANOVDB_ABI* reportEntries)(void* userdata, pnanovdb_uint64_t captureID, pnanovdb_uint32_t numEntries, pnanovdb_compute_profiler_entry_t* entries) = nullptr;
    };

    Profiler* profiler_create(Context* context);
    void profiler_destroy(Context* context, Profiler* ptr);
    void profiler_beginCapture(Context* context, Profiler* ptr, pnanovdb_uint64_t numEntries);
    void profiler_endCapture(Context* context, Profiler* ptr);
    void profiler_processCaptures(Context* context, Profiler* ptr);
    void profiler_timestamp(Context* context, Profiler* ptr, const char* label);

    void enableProfiler(pnanovdb_compute_context_t* context, void* userdata, void(PNANOVDB_ABI* reportEntries)(void* userdata, pnanovdb_uint64_t captureID, pnanovdb_uint32_t numEntries, pnanovdb_compute_profiler_entry_t* entries));
    void disableProfiler(pnanovdb_compute_context_t* context);

    enum ContextNodeType
    {
        eContextNodeType_unknown = 0,
        eContextNodeType_compute = 1,
        eContextNodeType_copyBuffer = 2,

        eContextNodeType_maxEnum = 0x7FFFFFFF
    };

    struct ContextNodeMemoryParams
    {
        unsigned char data[128u];
    };

    union ContextNodeParams
    {
        pnanovdb_compute_dispatch_params_t compute;
        pnanovdb_compute_copy_buffer_params_t copyBuffer;
        ContextNodeMemoryParams memory;
    };

    struct ContextNode
    {
        ContextNodeType type = eContextNodeType_unknown;
        ContextNodeParams params = {};
        const char* label = "Unknown";
        std::vector<pnanovdb_compute_descriptor_write_t> descriptorWrites;
        std::vector<pnanovdb_compute_resource_t> resources;

        std::vector<BufferTransient*> bufferTransientsCreate;
        std::vector<TextureTransient*> textureTransientsCreate;
        std::vector<BufferTransient*> bufferTransientsDestroy;
        std::vector<TextureTransient*> textureTransientsDestroy;
        std::vector<VkBufferMemoryBarrier> bufferBarriers;
        std::vector<VkImageMemoryBarrier> imageBarriers;
    };

    struct Context
    {
        DeviceQueue* deviceQueue = nullptr;

        std::vector<std::unique_ptr<Buffer>> pool_buffers;
        std::vector<std::unique_ptr<Texture>> pool_textures;
        std::vector<std::unique_ptr<Sampler>> pool_samplers;

        std::vector<std::unique_ptr<BufferTransient>> bufferTransients;
        std::vector<std::unique_ptr<TextureTransient>> textureTransients;

        std::vector<std::unique_ptr<BufferAcquire>> bufferAcquires;
        std::vector<std::unique_ptr<TextureAcquire>> textureAcquires;

        std::vector<pnanovdb_compute_buffer_t*> deferredReleaseBuffers;
        std::vector<pnanovdb_compute_texture_t*> deferredReleaseTextures;

        std::vector<ContextNode> nodes;

        std::vector<VkBufferMemoryBarrier> restore_bufferBarriers;
        std::vector<VkImageMemoryBarrier> restore_imageBarriers;

        Profiler* profiler = nullptr;

        pnanovdb_uint64_t minLifetime = 60u;

        pnanovdb_compute_log_print_t logPrint = nullptr;
    };

    void addPassCompute(pnanovdb_compute_context_t* context, const pnanovdb_compute_dispatch_params_t* params);
    void addPassCopyBuffer(pnanovdb_compute_context_t* context, const pnanovdb_compute_copy_buffer_params_t* params);

    void setResourceMinLifetime(pnanovdb_compute_context_t* context, pnanovdb_uint64_t minLifetime);

    Context* context_create(DeviceQueue* deviceQueue);
    void context_destroy(Context* context);
    void context_resetNodes(Context* context);
    void context_flushNodes(Context* context);

    /// Format conversion

    struct FormatConverter;

    FormatConverter* formatConverter_create();
    void formatConverter_destroy(FormatConverter* converter);
    VkFormat formatConverter_convertToVulkan(FormatConverter* converter, pnanovdb_compute_format_t format);
    pnanovdb_compute_format_t formatConverter_convertToPnanovdb(FormatConverter* converter, VkFormat format);
    pnanovdb_uint32_t formatConverter_getFormatSizeInBytes(FormatConverter* converter, pnanovdb_compute_format_t format);

    PNANOVDB_INLINE uint32_t context_getMemoryType(Context* context, uint32_t typeBits, VkMemoryPropertyFlags properties)
    {
        for (uint32_t i = 0u; i < context->deviceQueue->device->memoryProperties.memoryTypeCount; i++)
        {
            if ((typeBits & 1) == 1)
            {
                if ((context->deviceQueue->device->memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    return i;
                }
            }
            typeBits >>= 1u;
        }
        return ~0u;
    }

    /// Utils

    struct ExtensionRequest
    {
        const char* name;
        pnanovdb_bool_t* pEnabled;
    };

    void determineMatches(std::vector<const char*>& extensionsEnabled, std::vector<ExtensionRequest>& extensionsRequest, std::vector<VkExtensionProperties>& extensions);

    void selectInstanceExtensions(DeviceManager* ptr, std::vector<const char*>& instanceExtensionsEnabled);

    void selectDeviceExtensions(Device* ptr, std::vector<const char*>& deviceExtensionsEnabled);
}
