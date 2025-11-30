/* Copyright (c) 2019-2025, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cs_register_pressure.h"

#include "common/vk_common.h"
#include "gltf_loader.h"
#include "gui.h"
#include "filesystem/legacy.h"
#include "platform/platform.h"
#include "rendering/subpasses/forward_subpass.h"
#include "stats/stats.h"
#include <cfloat>

#ifdef TRACY_ENABLE
#    include <tracy/Tracy.hpp>
#    include <tracy/TracyVulkan.hpp>
#endif

CS_register_pressure::CS_register_pressure() {}

CS_register_pressure::~CS_register_pressure()
{
#ifdef TRACY_ENABLE
    if (tracy_context)
    {
        TracyVkDestroy(tracy_context);
    }
#endif
}

void CS_register_pressure::create_compute_resources()
{
    const VkDeviceSize size = 4ull * 1024ull * 1024ull;
    storage_buffer = std::make_unique<vkb::core::BufferC>(get_device(), size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    std::vector<float> init(size / sizeof(float));
    for (uint32_t i = 0; i < init.size(); ++i) init[i] = static_cast<float>(i) * 0.001f;
    storage_buffer->convert_and_update(init);

    auto request_module = [&](const char *path_spv, const char *path_glsl) -> vkb::ShaderModule & {
        // Prefer precompiled SPIR-V, but fall back to GLSL source if SPIR-V is not present
        std::string spv_full = vkb::fs::path::get(vkb::fs::path::Shaders, path_spv);
        if (vkb::fs::is_file(spv_full))
        {
            return get_device().get_resource_cache().request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, vkb::ShaderSource(path_spv));
        }
        // Fallback to GLSL source; framework will compile at runtime if toolchain is available
        return get_device().get_resource_cache().request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, vkb::ShaderSource(path_glsl));
    };

    auto &r8_module   = request_module("cs_register_pressure/r8.comp.spv",   "cs_register_pressure/r8.comp");
    auto &r16_module  = request_module("cs_register_pressure/r16.comp.spv",  "cs_register_pressure/r16.comp");
    auto &r32_module  = request_module("cs_register_pressure/r32.comp.spv",  "cs_register_pressure/r32.comp");
    auto &r64_module  = request_module("cs_register_pressure/r64.comp.spv",  "cs_register_pressure/r64.comp");
    auto &r128_module = request_module("cs_register_pressure/r128.comp.spv", "cs_register_pressure/r128.comp");
    auto &r256_module = request_module("cs_register_pressure/r256.comp.spv", "cs_register_pressure/r256.comp");

    layout_r8   = &get_device().get_resource_cache().request_pipeline_layout({&r8_module});
    layout_r16  = &get_device().get_resource_cache().request_pipeline_layout({&r16_module});
    layout_r32  = &get_device().get_resource_cache().request_pipeline_layout({&r32_module});
    layout_r64  = &get_device().get_resource_cache().request_pipeline_layout({&r64_module});
    layout_r128 = &get_device().get_resource_cache().request_pipeline_layout({&r128_module});
    layout_r256 = &get_device().get_resource_cache().request_pipeline_layout({&r256_module});

#ifdef TRACY_ENABLE
    // Initialize Tracy Vulkan context once
    if (!tracy_context)
    {
        auto &dev   = get_device();
        auto  phys  = dev.get_gpu().get_handle();
        auto  queue = dev.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0).get_handle();
        VkCommandBuffer cmd = dev.create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        tracy_context = TracyVkContext(phys, dev.get_handle(), queue, cmd);
        dev.flush_command_buffer(cmd, queue);
    }
#endif

    if (get_device().get_gpu().get_properties().limits.timestampComputeAndGraphics)
    {
        VkQueryPoolCreateInfo query_pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = 2;
        try
        {
            query_pool = std::make_unique<vkb::QueryPool>(get_device(), query_pool_info);
        }
        catch (const std::exception &e)
        {
            LOGW("Failed to create timestamp query pool: {}", e.what());
            query_pool.reset();
        }
    }
    else
    {
        LOGW("Timestamp queries not supported on this device");
    }
}

bool CS_register_pressure::prepare(const vkb::ApplicationOptions &options)
{
    if (!VulkanSample::prepare(options))
    {
        return false;
    }

    load_scene("scenes/sponza/Sponza01.gltf");

    auto &camera_node = vkb::add_free_camera(get_scene(), "main_camera", get_render_context().get_surface_extent());
    auto camera       = &camera_node.get_component<vkb::sg::Camera>();

    vkb::ShaderSource vert_shader("base.vert.spv");
    vkb::ShaderSource frag_shader("base.frag.spv");
    auto scene_subpass = std::make_unique<vkb::ForwardSubpass>(get_render_context(), std::move(vert_shader), std::move(frag_shader), get_scene(), *camera);
    auto render_pipeline = std::make_unique<vkb::RenderPipeline>();
    render_pipeline->add_subpass(std::move(scene_subpass));
    set_render_pipeline(std::move(render_pipeline));

    get_stats().request_stats({vkb::StatIndex::frame_times, vkb::StatIndex::gpu_cycles, vkb::StatIndex::gpu_fragment_cycles, vkb::StatIndex::gpu_vertex_cycles});
    create_gui(*window, &get_stats());

    create_compute_resources();
    return true;
}

void CS_register_pressure::draw_gui()
{
    get_gui().show_options_window([&]() {
        ImGui::Checkbox("Enable Compute", &enable_compute);
        ImGui::SliderInt("Iterations", &iterations, 128, 20000);
        int level = static_cast<int>(current_level);
        const char *labels[] = {"R8","R16","R32","R64","R128","R256"};
        if (ImGui::Combo("Pressure", &level, labels, IM_ARRAYSIZE(labels)))
        {
            current_level = static_cast<PressureLevel>(level);
        }

        int group_size_idx = static_cast<int>(current_group_size);
        const char *group_labels[] = {"32", "64", "128", "256", "512"};
        if (ImGui::Combo("Group Size", &group_size_idx, group_labels, IM_ARRAYSIZE(group_labels)))
        {
            current_group_size = static_cast<GroupSize>(group_size_idx);
        }

        // Compute and display total threads (groups * local_size)
        uint32_t local_size = 64;
        switch (current_group_size)
        {
            case GroupSize::L32: local_size = 32; break;
            case GroupSize::L64: local_size = 64; break;
            case GroupSize::L128: local_size = 128; break;
            case GroupSize::L256: local_size = 256; break;
            case GroupSize::L512: local_size = 512; break;
        }
        uint32_t groups = static_cast<uint32_t>((storage_buffer->get_size() / sizeof(float) + local_size - 1) / local_size);
        uint64_t total_threads = static_cast<uint64_t>(groups) * static_cast<uint64_t>(local_size);
        ImGui::Text("CS Threads: %llu (groups=%u, local=%u)", static_cast<unsigned long long>(total_threads), groups, local_size);

        if (query_pool)
        {
            uint64_t timestamps[2] = {0};
            // Try to read without waiting; if not ready, show the message
            VkResult result = query_pool->get_results(0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (result == VK_SUCCESS)
            {
                float timestamp_period = get_device().get_gpu().get_properties().limits.timestampPeriod;
                compute_duration_ms = (timestamps[1] - timestamps[0]) * timestamp_period / 1000000.0f;

                compute_history.push_back(compute_duration_ms);
                if (compute_history.size() > 100)
                {
                    compute_history.erase(compute_history.begin());
                }

#ifdef TRACY_ENABLE
                TracyPlot("Compute Duration", compute_duration_ms);
#endif
                ImGui::Text("Compute Time: %.3f ms", compute_duration_ms);
            }
            else if (result == VK_NOT_READY)
            {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Query Not Ready (Waiting for GPU)");
            }
            else
            {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Query error: %d", result);
            }

            if (!compute_history.empty())
            {
                ImGui::PlotLines("History", compute_history.data(), static_cast<int>(compute_history.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 80));
            }

#ifdef TRACY_ENABLE
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Tracy Enabled");
#else
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "Tracy Disabled");
#endif
        }
        else
        {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "Query Pool is NULL");
            if (!get_device().get_gpu().get_properties().limits.timestampComputeAndGraphics)
            {
                ImGui::Text("Timestamp queries not supported");
            }
        }
    }, 10); // Increase window height lines to avoid clipping
}

void CS_register_pressure::draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target)
{
#ifdef TRACY_ENABLE
    // Collect GPU data for previous frame
    if (tracy_context)
    {
        TracyVkCollect(tracy_context, command_buffer.get_handle());
    }
#endif

    // Do not reset queries here; render() controls the lifecycle
    VulkanSample::draw(command_buffer, render_target);
}

void CS_register_pressure::dispatch_compute(vkb::core::CommandBufferC &command_buffer)
{
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif
    if (!enable_compute) return;

    vkb::core::BufferC &buf = *storage_buffer;
    command_buffer.bind_buffer(buf, 0, buf.get_size(), 0, 0, 0);

    vkb::PipelineLayout *layout = nullptr;
    switch (current_level)
    {
        case PressureLevel::R8: layout = layout_r8; break;
        case PressureLevel::R16: layout = layout_r16; break;
        case PressureLevel::R32: layout = layout_r32; break;
        case PressureLevel::R64: layout = layout_r64; break;
        case PressureLevel::R128: layout = layout_r128; break;
        case PressureLevel::R256: layout = layout_r256; break;
    }
    command_buffer.bind_pipeline_layout(*layout);

    struct Push { int iters; } push{iterations};
    command_buffer.push_constants(push);

    uint32_t local_size = 64;
    switch (current_group_size)
    {
        case GroupSize::L32: local_size = 32; break;
        case GroupSize::L64: local_size = 64; break;
        case GroupSize::L128: local_size = 128; break;
        case GroupSize::L256: local_size = 256; break;
        case GroupSize::L512: local_size = 512; break;
    }

    command_buffer.set_specialization_constant(0, local_size);

    uint32_t groups = static_cast<uint32_t>((storage_buffer->get_size() / sizeof(float) + local_size - 1) / local_size);

#ifdef TRACY_ENABLE
    TracyVkZone(tracy_context, command_buffer.get_handle(), "CS Dispatch");
#endif

    if (query_pool)
    {
        // Use TOP_OF_PIPE before dispatch for reliable timestamp latching
        command_buffer.write_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *query_pool, 0);
    }

    command_buffer.dispatch(groups, 1, 1);

    if (query_pool)
    {
        // Use BOTTOM_OF_PIPE after dispatch for reliable timestamp latching
        command_buffer.write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *query_pool, 1);
    }
}

void CS_register_pressure::render(vkb::core::CommandBufferC &command_buffer)
{
    // Reset queries at the start of the frame so we write fresh timestamps
    if (query_pool)
    {
        command_buffer.reset_query_pool(*query_pool, 0, 2);
    }

    get_render_pipeline().draw(command_buffer, get_render_context().get_active_frame().get_render_target());

    // Record compute and timestamps in the same command buffer
    dispatch_compute(command_buffer);

    // Throttle to avoid compute workload explosion during measurement
    get_device().wait_idle();
}

std::unique_ptr<vkb::VulkanSampleC> create_cs_register_pressure()
{
    return std::make_unique<CS_register_pressure>();
}
