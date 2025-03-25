import * from './B';

osition) Position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn v_main(
  @location(0) position: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) vv: Uniforms,
) -> VertexOutput {
  var output : VertexOutput;
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
