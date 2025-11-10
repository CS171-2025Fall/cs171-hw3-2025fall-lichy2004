#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
              other.low_bnd[0] <= this->upper_bnd[0]) ||
             (this->low_bnd[0] >= other.low_bnd[0] &&
                 this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
              other.low_bnd[1] <= this->upper_bnd[1]) ||
             (this->low_bnd[1] >= other.low_bnd[1] &&
                 this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
              other.low_bnd[2] <= this->upper_bnd[2]) ||
             (this->low_bnd[2] >= other.low_bnd[2] &&
                 this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  // TODO(HW3): implement ray intersection with AABB.
  // 使用 Slab 方法实现光线与轴对齐包围盒的求交测试
  
  // 获取光线的安全逆方向（避免除零错误）
  const Vec3f &inv_dir = ray.safe_inverse_direction;
  
  // 对三个坐标轴分别计算进入和退出时间
  // X 轴
  Float t0_x = (low_bnd.x - ray.origin.x) * inv_dir.x;
  Float t1_x = (upper_bnd.x - ray.origin.x) * inv_dir.x;
  // 确保 t0_x <= t1_x（处理负方向的情况）
  if (t0_x > t1_x) std::swap(t0_x, t1_x);
  
  // Y 轴
  Float t0_y = (low_bnd.y - ray.origin.y) * inv_dir.y;
  Float t1_y = (upper_bnd.y - ray.origin.y) * inv_dir.y;
  if (t0_y > t1_y) std::swap(t0_y, t1_y);
  
  // Z 轴
  Float t0_z = (low_bnd.z - ray.origin.z) * inv_dir.z;
  Float t1_z = (upper_bnd.z - ray.origin.z) * inv_dir.z;
  if (t0_z > t1_z) std::swap(t0_z, t1_z);
  
  // 计算整体的进入和退出时间
  // 进入时间 = 所有进入时间的最大值
  *t_in = std::max({t0_x, t0_y, t0_z});
  // 退出时间 = 所有退出时间的最小值
  *t_out = std::min({t1_x, t1_y, t1_z});
  
  // 检查有效性
  // 如果退出时间 < 进入时间，说明光线不与 AABB 相交
  if (*t_out < *t_in) {
    return false;
  }
  
  // 如果退出时间 < 0，说明 AABB 在光线起点后面
  if (*t_out < 0) {
    return false;
  }
  
  // 有有效交点
  return true;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
    const ref<TriangleMeshResource> &mesh, SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType    = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0  = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1  = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2  = Cast<InternalScalarType>(vertices[v_idx[2]]);

  // TODO(HW3): implement ray-triangle intersection test.
  // 使用 Möller-Trumbore 算法实现光线-三角形求交
  
  // 计算三角形的两条边向量
  InternalVecType e1 = v1 - v0;
  InternalVecType e2 = v2 - v0;
  
  // 计算辅助向量 p = dir × e2
  InternalVecType p = Cross(dir, e2);
  
  // 计算行列式 det = e1 · p
  InternalScalarType det = Dot(e1, p);
  
  // 如果 det 接近0，说明光线与三角形平行，返回 false
  if (std::abs(det) < InternalScalarType(1e-8)) {
    return false;
  }
  
  // 计算 det 的倒数
  InternalScalarType inv_det = InternalScalarType(1) / det;
  
  // 计算从 v0 到光线起点的向量
  InternalVecType t_vec = Cast<InternalScalarType>(ray.origin) - v0;
  
  // 计算重心坐标 u = (t_vec · p) / det
  InternalScalarType u = Dot(t_vec, p) * inv_det;
  
  // 检查 u 的范围
  if (u < InternalScalarType(0) || u > InternalScalarType(1)) {
    return false;
  }
  
  // 计算辅助向量 q = t_vec × e1
  InternalVecType q = Cross(t_vec, e1);
  
  // 计算重心坐标 v = (dir · q) / det
  InternalScalarType v = Dot(dir, q) * inv_det;
  
  // 检查 v 的范围以及 u + v <= 1
  if (v < InternalScalarType(0) || u + v > InternalScalarType(1)) {
    return false;
  }
  
  // 计算交点距离 t = (e2 · q) / det
  InternalScalarType t = Dot(e2, q) * inv_det;
  
  // 检查 t 是否在光线的有效范围内
  if (t < InternalScalarType(ray.t_min) || t > InternalScalarType(ray.t_max)) {
    return false;
  }

  // We will reach here if there is an intersection

  CalculateTriangleDifferentials(interaction,
      {static_cast<Float>(1 - u - v), static_cast<Float>(u),
          static_cast<Float>(v)},
      mesh, triangle_index);
  AssertNear(interaction.p, ray(t));
  assert(ray.withinTimeRange(t));
  ray.setTimeMax(t);
  return true;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
      Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd   = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh  = mesh;   // set the pointer
  this->bound = bound;  // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const {
  return bound;
}

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END
