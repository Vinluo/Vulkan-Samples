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

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/command_pool.h"
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
    auto &r96_module  = request_module("cs_register_pressure/r96.comp.spv",  "cs_register_pressure/r96.comp");
    auto &r128_module = request_module("cs_register_pressure/r128.comp.spv", "cs_register_pressure/r128.comp");
    auto &r192_module = request_module("cs_register_pressure/r192.comp.spv", "cs_register_pressure/r192.comp");
    auto &r256_module = request_module("cs_register_pressure/r256.comp.spv", "cs_register_pressure/r256.comp");

    layout_r8   = &get_device().get_resource_cache().request_pipeline_layout({&r8_module});
    layout_r16  = &get_device().get_resource_cache().request_pipeline_layout({&r16_module});
    layout_r32  = &get_device().get_resource_cache().request_pipeline_layout({&r32_module});
    layout_r64  = &get_device().get_resource_cache().request_pipeline_layout({&r64_module});
    layout_r96  = &get_device().get_resource_cache().request_pipeline_layout({&r96_module});
    layout_r128 = &get_device().get_resource_cache().request_pipeline_layout({&r128_module});
    layout_r192 = &get_device().get_resource_cache().request_pipeline_layout({&r192_module});
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
        
        // Allocate queries for each frame
        uint32_t frame_count = static_cast<uint32_t>(get_render_context().get_render_frames().size());
        query_pool_info.queryCount = frame_count * 2;
        
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

    // Pre-compile all pipelines to avoid runtime stutter
    std::vector<vkb::PipelineLayout *> layouts = {layout_r8, layout_r16, layout_r32, layout_r64, layout_r96, layout_r128, layout_r192, layout_r256};
    std::vector<uint32_t>              sizes   = {32, 64, 96, 128, 192, 256, 384, 512, 1024};

    // Create a temporary command buffer to force descriptor set creation and pipeline compilation
    auto &queue = get_device().get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);
    
    // Use the first render frame for the temporary command pool to satisfy assertion
    auto &render_frame = get_render_context().get_render_frames()[0];
    vkb::core::CommandPoolC temp_pool(get_device(), queue.get_family_index(), render_frame.get());
    vkb::core::CommandBufferC temp_cmd(temp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    temp_cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Bind buffer once (it's the same for all)
    vkb::core::BufferC &buf = *storage_buffer;
    temp_cmd.bind_buffer(buf, 0, buf.get_size(), 0, 0, 0);

    for (auto *layout : layouts)
    {
        // Bind layout
        temp_cmd.bind_pipeline_layout(*layout);

        for (auto size : sizes)
        {
            vkb::PipelineState state;
            state.set_pipeline_layout(*layout);
            state.set_specialization_constant(0, vkb::to_bytes(size));
            get_device().get_resource_cache().request_compute_pipeline(state);
        }

        // Force descriptor set creation for this layout by doing a dummy dispatch
        // We use a fixed group size (e.g. 64) just to trigger the descriptor set update logic
        temp_cmd.set_specialization_constant(0, (uint32_t)64);
        temp_cmd.dispatch(1, 1, 1);
    }

    temp_cmd.end();

    // Submit the dummy work to force driver compilation/upload
    // We don't need to wait for it, but it ensures the driver sees the pipelines
    auto &fence_pool = get_device().get_fence_pool();
    auto fence = fence_pool.request_fence();
    queue.submit(temp_cmd, fence);
    fence_pool.wait();
    fence_pool.reset();
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
        ImGui::SliderInt("Iterations", &iterations, 1, 100);
        int level = static_cast<int>(current_level);
        const char *labels[] = {"R8","R16","R32","R64","R96","R128","R192","R256"};
        if (ImGui::Combo("Pressure", &level, labels, IM_ARRAYSIZE(labels)))
        {
            current_level = static_cast<PressureLevel>(level);
        }

        int group_size_idx = static_cast<int>(current_group_size);
        const char *group_labels[] = {"32", "64", "96", "128", "192", "256", "384", "512", "1024"};
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
            case GroupSize::L96: local_size = 96; break;
            case GroupSize::L128: local_size = 128; break;
            case GroupSize::L192: local_size = 192; break;
            case GroupSize::L256: local_size = 256; break;
            case GroupSize::L384: local_size = 384; break;
            case GroupSize::L512: local_size = 512; break;
            case GroupSize::L1024: local_size = 1024; break;
        }
        uint32_t groups = static_cast<uint32_t>((storage_buffer->get_size() / sizeof(float) + local_size - 1) / local_size);
        uint64_t total_threads = static_cast<uint64_t>(groups) * static_cast<uint64_t>(local_size);
        ImGui::Text("CS Threads: %llu (groups=%u, local=%u)", static_cast<unsigned long long>(total_threads), groups, local_size);

        // 768 ops per iteration (approx) based on shader logic
        uint64_t total_alu_ops = static_cast<uint64_t>(iterations) * 768;
        ImGui::Text("Per Thread Total ALU Ops: %llu", static_cast<unsigned long long>(total_alu_ops));

        if (query_pool)
        {
            // Always show the last valid duration
            ImGui::Text("Compute Time: %.3f ms", compute_duration_ms);

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
        case PressureLevel::R96: layout = layout_r96; break;
        case PressureLevel::R128: layout = layout_r128; break;
        case PressureLevel::R192: layout = layout_r192; break;
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
        case GroupSize::L96: local_size = 96; break;
        case GroupSize::L128: local_size = 128; break;
        case GroupSize::L192: local_size = 192; break;
        case GroupSize::L256: local_size = 256; break;
        case GroupSize::L384: local_size = 384; break;
        case GroupSize::L512: local_size = 512; break;
        case GroupSize::L1024: local_size = 1024; break;
    }

    command_buffer.set_specialization_constant(0, local_size);

    uint32_t groups = static_cast<uint32_t>((storage_buffer->get_size() / sizeof(float) + local_size - 1) / local_size);

#ifdef TRACY_ENABLE
    TracyVkZone(tracy_context, command_buffer.get_handle(), "CS Dispatch");
#endif

    uint32_t active_frame_index = get_render_context().get_active_frame_index();
    uint32_t query_index = active_frame_index * 2;

    if (query_pool)
    {
        // Use TOP_OF_PIPE before dispatch for reliable timestamp latching
        command_buffer.write_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *query_pool, query_index);
    }

    command_buffer.dispatch(groups, 1, 1);

    if (query_pool)
    {
        // Use BOTTOM_OF_PIPE after dispatch for reliable timestamp latching
        command_buffer.write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *query_pool, query_index + 1);
    }
}

void CS_register_pressure::render(vkb::core::CommandBufferC &command_buffer)
{
    // Reset queries at the start of the frame so we write fresh timestamps
    if (query_pool)
    {
        uint32_t active_frame_index = get_render_context().get_active_frame_index();
        uint32_t query_index = active_frame_index * 2;

        // Read results from the previous execution of this frame
        // This data is guaranteed to be ready because we waited for this frame's fence in begin_frame
        uint64_t timestamps[2] = {0};
        VkResult result = query_pool->get_results(query_index, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        
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
        }

        command_buffer.reset_query_pool(*query_pool, query_index, 2);
    }

    // Record compute and timestamps in the same command buffer
    dispatch_compute(command_buffer);

    // Synchronize Compute and Graphics
    vkCmdPipelineBarrier(command_buffer.get_handle(),
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         0, nullptr);

    get_render_pipeline().draw(command_buffer, get_render_context().get_active_frame().get_render_target());
}

std::unique_ptr<vkb::VulkanSampleC> create_cs_register_pressure()
{
    return std::make_unique<CS_register_pressure>();
}
