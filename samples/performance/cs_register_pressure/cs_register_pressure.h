/* Copyright (c) 2019-2024, Arm Limited and Contributors
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

#pragma once

#include "rendering/render_pipeline.h"
#include "scene_graph/components/camera.h"
#include "vulkan_sample.h"
#include "core/query_pool.h"
#include "core/pipeline.h"
#include <vector>

#ifdef TRACY_ENABLE
#include <tracy/TracyVulkan.hpp>
#endif

class CS_register_pressure : public vkb::VulkanSampleC
{
  public:
    CS_register_pressure();

    virtual bool prepare(const vkb::ApplicationOptions &options) override;
    virtual void draw_gui() override; // GUI controls for register pressure
    virtual void render(vkb::core::CommandBufferC &command_buffer) override; // run compute after scene
    virtual void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
    virtual ~CS_register_pressure();

  private:
    void create_compute_resources();
    void dispatch_compute(vkb::core::CommandBufferC &command_buffer);

    enum class PressureLevel { R8, R16, R32, R64, R96, R128, R192, R256 };
    PressureLevel current_level{PressureLevel::R8};

    enum class GroupSize { L32, L64, L96, L128, L192, L256, L384, L512, L1024 };
    GroupSize current_group_size{GroupSize::L64};

    std::unique_ptr<vkb::core::BufferC> storage_buffer; // workload buffer
    std::unique_ptr<vkb::QueryPool> query_pool;
    float compute_duration_ms{0.0f};
    std::vector<float> compute_history;
    bool query_ready{false};

    vkb::PipelineLayout *layout_r8{nullptr};
    vkb::PipelineLayout *layout_r16{nullptr};
    vkb::PipelineLayout *layout_r32{nullptr};
    vkb::PipelineLayout *layout_r64{nullptr};
    vkb::PipelineLayout *layout_r96{nullptr};
    vkb::PipelineLayout *layout_r128{nullptr};
    vkb::PipelineLayout *layout_r192{nullptr};
    vkb::PipelineLayout *layout_r256{nullptr};

#ifdef TRACY_ENABLE
    TracyVkCtx tracy_context{nullptr};
#endif

    bool enable_compute{true};
    int  iterations{1};
};

std::unique_ptr<vkb::VulkanSampleC> create_cs_register_pressure();
