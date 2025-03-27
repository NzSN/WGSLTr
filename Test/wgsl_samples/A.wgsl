import * from './B';

struct Uniforms {
  matrix : mat4x4<f32>,
  alpha: f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var u_Sampler: sampler;
@group(0) @binding(3) var u_Texture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn v_main(
  @location(0) position: vec2<f32>,
  @location(1) uv: vec2f<f32>,
  @location(2) vv: Uniforms,
) -> VertexOutput {
  var output : VertexOutput  = gain;
  output.Position = camera.mvpMatrix * uniforms.matrix * vec4(position, 0.0, 1.0);
  output.uv = uv;
  return output;
}

@fragment
fn f_main(
  @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
  var color = textureSample(u_Texture, u_Sampler, uv);
  color.a *= uniforms.alpha;
  return color;
}
