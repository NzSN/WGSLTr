struct Camera {
  mvpMatrix: mat4x4<f32>,
  mvMatrix: mat4x4<f32>,
  invMatrix: mat4x4<f32>,
  pos: vec3<f32>,
  isOrth: f32,
  vec: vec3<f32>,
  borderZ: f32,
  pxLength: f32,
  flip: f32,
  width: f32,
  height: f32,
}

struct StaticPolygonInfo {
  matrix: mat3x2<f32>,
  order: f32,
  colorIndex: f32,
}
