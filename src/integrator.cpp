#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        // 为每个像素生成多条光线进行采样，实现抗锯齿
        
        // 1. 获取像素采样位置（包含亚像素偏移）
        // sampler.getPixelSample() 返回 [dx + random, dy + random]，其中 random ∈ [0,1)
        const Vec2f &pixel_sample = sampler.getPixelSample();
        
        // 2. 生成微分光线
        // 根据像素采样位置生成光线，用于光线追踪
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        
        // 3. 计算该光线的辐射度
        // Li 函数会追踪光线并计算最终的颜色值
        const Vec3f &L = Li(scene, ray, sampler);
        
        // 4. 将结果提交到胶片
        // Film 会自动对多次采样的结果进行平均
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      // 当光线击中完美折射材质时，根据折射方向继续追踪
      
      // 1. 调用 BSDF 采样函数获取折射方向
      // sample 函数会设置 interaction.wi 为折射后的入射方向
      Float pdf;
      Vec3f bsdf_value = interaction.bsdf->sample(interaction, sampler, &pdf);
      
      // 2. 更新光线为从交点沿折射方向发射的新光线
      // spawnRay 会自动处理原点偏移，避免自相交问题
      ray = interaction.spawnRay(interaction.wi);
      
      // 3. 继续循环，追踪折射后的光线
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  auto test_ray       = DifferentialRay(interaction.p, light_dir);

  // TODO(HW3): Test for occlusion
  // 测试从交点到光源的路径是否被遮挡（阴影测试）
  
  // 测试阴影光线是否与场景中的任何几何体相交
  SurfaceInteraction shadow_interaction;
  bool occluded = scene->intersect(test_ray, shadow_interaction);
  
  // 如果有交点，检查交点是否在光源之前
  // 使用 EPS 避免浮点数精度问题
  if (occluded) {
    Float dist_to_occlusion = Norm(shadow_interaction.p - interaction.p);
    if (dist_to_occlusion < dist_to_light - EPS) {
      // 被遮挡，返回黑色（无光照）
      return Vec3f(0, 0, 0);
    }
  }

  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
    // TODO(HW3): Compute the contribution
    // 计算点光源的直接光照，使用简化的 Lambert 漫反射模型
    
    // The angle between light direction and surface normal
    // Lambert 漫反射：余弦项表示光照强度随入射角度的变化
    Float cos_theta =
        std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided
    
    // 计算反照率（albedo）
    // IdealDiffusion::evaluate 返回 albedo / PI，所以需要乘以 PI 恢复
    Vec3f albedo = bsdf->evaluate(interaction) * PI;
    
    // 计算点光源的光照贡献
    // 简化的渲染方程：L = flux * albedo * cos_theta / (4 * PI * dist^2)
    // - point_light_flux: 光源的辐射通量
    // - albedo: 表面的反射率
    // - cos_theta: Lambert 余弦项
    // - 4 * PI * dist^2: 点光源的球面立体角衰减
    color = point_light_flux * albedo * cos_theta / 
            (4.0f * PI * dist_to_light * dist_to_light);
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
