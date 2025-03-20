struct Uniforms {
  viewProjectionMatrix : mat4x4f
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@group(1) @binding(0) var<uniform> modelMatrix : mat4x4f;

struct VertexInput {
  @location(0) position : vec4f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) normal: vec3f,
  @location(1) uv : vec2f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.viewProjectionMatrix * modelMatrix * input.position;
  output.normal = normalize((modelMatrix * vec4(input.normal, 0)).xyz);
  output.uv = input.uv;
  return output;
}

@group(1) @binding(1) var meshSampler: sampler;
@group(1) @binding(2) var meshTexture: texture_2d<f32>;

// Static directional lighting
const lightDir = vec3f(1, 1, 1);
const dirColor = vec3(1);
const ambientColor = vec3f(0.05);

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let textureColor = textureSample(meshTexture, meshSampler, input.uv);

  // Very simplified lighting algorithm.
  let lightColor = saturate(ambientColor + max(dot(input.normal, lightDir), 0.0) * dirColor);

  return vec4f(textureColor.rgb * lightColor, textureColor.a);
}
override shadowDepthTextureSize: f32 = 1024.0;

struct Scene {
  lightViewProjMatrix : mat4x4f,
  cameraViewProjMatrix : mat4x4f,
  lightPos : vec3f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct FragmentInput {
  @location(0) shadowPos : vec3f,
  @location(1) fragPos : vec3f,
  @location(2) fragNorm : vec3f,
}

const albedo = vec3f(0.9);
const ambientFactor = 0.2;

@fragment
fn main(input : FragmentInput) -> @location(0) vec4f {
  // Percentage-closer filtering. Sample texels in the region
  // to smooth the result.
  var visibility = 0.0;
  let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2f(vec2(x, y)) * oneOverShadowDepthTextureSize;

      visibility += textureSampleCompare(
        shadowMap, shadowSampler,
        input.shadowPos.xy + offset, input.shadowPos.z - 0.007
      );
    }
  }
  visibility /= 9.0;

  let lambertFactor = max(dot(normalize(scene.lightPos - input.fragPos), normalize(input.fragNorm)), 0.0);
  let lightingFactor = min(ambientFactor + visibility * lambertFactor, 1.0);

  return vec4(lightingFactor * albedo, 1.0);
}
struct Scene {
  lightViewProjMatrix: mat4x4f,
  cameraViewProjMatrix: mat4x4f,
  lightPos: vec3f,
}

struct Model {
  modelMatrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

struct VertexOutput {
  @location(0) shadowPos: vec3f,
  @location(1) fragPos: vec3f,
  @location(2) fragNorm: vec3f,

  @builtin(position) Position: vec4f,
}

@vertex
fn main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f
) -> VertexOutput {
  var output : VertexOutput;

  // XY is in (-1, 1) space, Z is in (0, 1) space
  let posFromLight = scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);

  // Convert XY to (0, 1)
  // Y is flipped because texture coords are Y-down.
  output.shadowPos = vec3(
    posFromLight.xy * vec2(0.5, -0.5) + vec2(0.5),
    posFromLight.z
  );

  output.Position = scene.cameraViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
  output.fragPos = output.Position.xyz;
  output.fragNorm = normal;
  return output;
}
struct Scene {
  lightViewProjMatrix: mat4x4f,
  cameraViewProjMatrix: mat4x4f,
  lightPos: vec3f,
}

struct Model {
  modelMatrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn main(
  @location(0) position: vec3f
) -> @builtin(position) vec4f {
  return scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
}
struct Time {
  value : f32,
}

struct Uniforms {
  scale : f32,
  offsetX : f32,
  offsetY : f32,
  scalar : f32,
  scalarOffset : f32,
}

@binding(0) @group(0) var<uniform> time : Time;
@binding(0) @group(1) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) v_color : vec4f,
}

@vertex
fn vert_main(
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var fade = (uniforms.scalarOffset + time.value * uniforms.scalar / 10.0) % 1.0;
  if (fade < 0.5) {
    fade = fade * 2.0;
  } else {
    fade = (1.0 - fade) * 2.0;
  }
  var xpos = position.x * uniforms.scale;
  var ypos = position.y * uniforms.scale;
  var angle = 3.14159 * 2.0 * fade;
  var xrot = xpos * cos(angle) - ypos * sin(angle);
  var yrot = xpos * sin(angle) + ypos * cos(angle);
  xpos = xrot + uniforms.offsetX;
  ypos = yrot + uniforms.offsetY;

  var output : VertexOutput;
  output.v_color = vec4(fade, 1.0 - fade, 0.0, 1.0) + color;
  output.Position = vec4(xpos, ypos, 0.0, 1.0);
  return output;
}

@fragment
fn frag_main(@location(0) v_color : vec4f) -> @location(0) vec4f {
  return v_color;
}
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vertex_main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  return VertexOutput(uniforms.modelViewProjectionMatrix * position, uv);
}

@fragment
fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV);
}
@group(0) @binding(0) var tex: texture_2d<f32>;

struct Varying {
  @builtin(position) pos: vec4f,
  @location(0) texelCoord: vec2f,
  @location(1) mipLevel: f32,
}

const kMipLevels = 4;
const baseMipSize: u32 = 16;

@vertex
fn vmain(
  @builtin(instance_index) instance_index: u32, // used as mipLevel
  @builtin(vertex_index) vertex_index: u32,
) -> Varying {
  var square = array(
    vec2f(0, 0), vec2f(0, 1), vec2f(1, 0),
    vec2f(1, 0), vec2f(0, 1), vec2f(1, 1),
  );
  let uv = square[vertex_index];
  let pos = vec4(uv * 2 - vec2(1, 1), 0.0, 1.0);

  let mipLevel = instance_index;
  let mipSize = f32(1 << (kMipLevels - mipLevel));
  let texelCoord = uv * mipSize;
  return Varying(pos, texelCoord, f32(mipLevel));
}

@fragment
fn fmain(vary: Varying) -> @location(0) vec4f {
  return textureLoad(tex, vec2u(vary.texelCoord), u32(vary.mipLevel));
}
struct Config {
  viewProj: mat4x4f,
  animationOffset: vec2f,
  flangeSize: f32,
  highlightFlange: f32,
};
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> matrices: array<mat4x4f>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var tex: texture_2d<f32>;

struct Varying {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

override kTextureBaseSize: f32;
override kViewportSize: f32;

@vertex
fn vmain(
  @builtin(instance_index) instance_index: u32,
  @builtin(vertex_index) vertex_index: u32,
) -> Varying {
  let flange = config.flangeSize;
  var uvs = array(
    vec2(-flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, -flange),
    vec2(1 + flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, 1 + flange),
  );
  // Default size (if matrix is the identity) makes 1 texel = 1 pixel.
  let radius = (1 + 2 * flange) * kTextureBaseSize / kViewportSize;
  var positions = array(
    vec2(-radius, -radius), vec2(-radius, radius), vec2(radius, -radius),
    vec2(radius, -radius), vec2(-radius, radius), vec2(radius, radius),
  );

  let modelMatrix = matrices[instance_index];
  let pos = config.viewProj * modelMatrix * vec4f(positions[vertex_index] + config.animationOffset, 0, 1);
  return Varying(pos, uvs[vertex_index]);
}

@fragment
fn fmain(vary: Varying) -> @location(0) vec4f {
  let uv = vary.uv;
  var color = textureSample(tex, samp, uv);

  let outOfBounds = uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1;
  if config.highlightFlange > 0 && outOfBounds {
    color += vec4(0.7, 0, 0, 0);
  }

  return color;
}


@fragment fn fs() -> @location(0) vec4f {
  return vec4f(1, 0.5, 0.2, 1);
}
struct VSOutput {
  @location(0) texcoord: vec2f,
};

@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var t: texture_2d<f32>;

@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
  let color = textureSample(t, s, vsOut.texcoord);
  if (color.a < 0.1) {
    discard;
  }
  return color;
}
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution * clipPos.w, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Uniforms {
  color0: vec4f,
  color1: vec4f,
  size: u32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex
fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
  const pos = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vertexIndex], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let grid = vec2u(position.xy) / uni.size;
  let checker = (grid.x + grid.y) % 2 == 1;
  return select(uni.color0, uni.color1, checker);
}

struct Params {
  filterDim : i32,
  blockDim : u32,
}

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var<uniform> params : Params;
@group(1) @binding(1) var inputTex : texture_2d<f32>;
@group(1) @binding(2) var outputTex : texture_storage_2d<rgba8unorm, write>;

struct Flip {
  value : u32,
}
@group(1) @binding(3) var<uniform> flip : Flip;

// This shader blurs the input texture in one direction, depending on whether
// |flip.value| is 0 or 1.
// It does so by running (128 / 4) threads per workgroup to load 128
// texels into 4 rows of shared memory. Each thread loads a
// 4 x 4 block of texels to take advantage of the texture sampling
// hardware.
// Then, each thread computes the blur result by averaging the adjacent texel values
// in shared memory.
// Because we're operating on a subset of the texture, we cannot compute all of the
// results since not all of the neighbors are available in shared memory.
// Specifically, with 128 x 128 tiles, we can only compute and write out
// square blocks of size 128 - (filterSize - 1). We compute the number of blocks
// needed in Javascript and dispatch that amount.

var<workgroup> tile : array<array<vec3f, 128>, 4>;

@compute @workgroup_size(32, 1, 1)
fn main(
  @builtin(workgroup_id) WorkGroupID : vec3u,
  @builtin(local_invocation_id) LocalInvocationID : vec3u
) {
  let filterOffset = (params.filterDim - 1) / 2;
  let dims = vec2i(textureDimensions(inputTex, 0));
  let baseIndex = vec2i(WorkGroupID.xy * vec2(params.blockDim, 4) +
                            LocalInvocationID.xy * vec2(4, 1))
                  - vec2(filterOffset, 0);

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var loadIndex = baseIndex + vec2(c, r);
      if (flip.value != 0u) {
        loadIndex = loadIndex.yx;
      }

      tile[r][4 * LocalInvocationID.x + u32(c)] = textureSampleLevel(
        inputTex,
        samp,
        (vec2f(loadIndex) + vec2f(0.25, 0.25)) / vec2f(dims),
        0.0
      ).rgb;
    }
  }

  workgroupBarrier();

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var writeIndex = baseIndex + vec2(c, r);
      if (flip.value != 0) {
        writeIndex = writeIndex.yx;
      }

      let center = i32(4 * LocalInvocationID.x) + c;
      if (center >= filterOffset &&
          center < 128 - filterOffset &&
          all(writeIndex < dims)) {
        var acc = vec3(0.0, 0.0, 0.0);
        for (var f = 0; f < params.filterDim; f++) {
          var i = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
        }
        textureStore(outputTex, writeIndex, vec4(acc, 1.0));
      }
    }
  }
}
struct ComputeUniforms {
  width: f32,
  height: f32,
  algo: u32,
  blockHeight: u32,
}

struct FragmentUniforms {
  // boolean, either 0 or 1
  highlight: u32,
}

struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) fragUV: vec2f
}

// Uniforms from compute shader
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: ComputeUniforms;
// Fragment shader uniforms
@group(1) @binding(0) var<uniform> fragment_uniforms: FragmentUniforms;

@fragment
fn frag_main(input: VertexOutput) -> @location(0) vec4f {
  var uv: vec2f = vec2f(
    input.fragUV.x * uniforms.width,
    input.fragUV.y * uniforms.height
  );

  var pixel: vec2u = vec2u(
    u32(floor(uv.x)),
    u32(floor(uv.y)),
  );
  
  var elementIndex = u32(uniforms.width) * pixel.y + pixel.x;
  var colorChanger = data[elementIndex];

  var subtracter = f32(colorChanger) / (uniforms.width * uniforms.height);

  if (fragment_uniforms.highlight == 1) {
    return select(
      //If element is above halfHeight, highlight green
      vec4f(vec3f(0.0, 1.0 - subtracter, 0.0).rgb, 1.0),
      //If element is below halfheight, highlight red
      vec4f(vec3f(1.0 - subtracter, 0.0, 0.0).rgb, 1.0),
      elementIndex % uniforms.blockHeight < uniforms.blockHeight / 2
    );
  }

  var color: vec3f = vec3f(
    1.0 - subtracter
  );

  return vec4f(color.rgb, 1.0);
}
@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn atomicToZero() {
  let counterValue = atomicLoad(&counter);
  atomicSub(&counter, counterValue);
}
struct OurVertexShaderOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

struct Uniforms {
  matrix: mat4x4f,
};

@group(0) @binding(2) var<uniform> uni: Uniforms;

@vertex fn vs(
  @builtin(vertex_index) vertexIndex : u32
) -> OurVertexShaderOutput {
  let pos = array(

    vec2f( 0.0,  0.0),  // center
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 0.0,  1.0),  // center, top

    // 2st triangle
    vec2f( 0.0,  1.0),  // center, top
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 1.0,  1.0),  // right, top
  );

  var vsOutput: OurVertexShaderOutput;
  let xy = pos[vertexIndex];
  vsOutput.position = uni.matrix * vec4f(xy, 0.0, 1.0);
  vsOutput.texcoord = xy;
  return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(fsInput: OurVertexShaderOutput) -> @location(0) vec4f {
  return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}// Whale.glb Vertex attributes
// Read in VertexInput from attributes
// f32x3    f32x3   f32x2       u8x4       f32x4
struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) normal: vec3f,
  @location(1) joints: vec4f,
  @location(2) weights: vec4f,
}

struct CameraUniforms {
  proj_matrix: mat4x4f,
  view_matrix: mat4x4f,
  model_matrix: mat4x4f,
}

struct GeneralUniforms {
  render_mode: u32,
  skin_mode: u32,
}

struct NodeUniforms {
  world_matrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
@group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
@group(2) @binding(0) var<uniform> node_uniforms: NodeUniforms;
@group(3) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
@group(3) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  // Compute joint_matrices * inverse_bind_matrices
  let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
  let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
  let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
  let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];
  // Compute influence of joint based on weight
  let skin_matrix = 
    joint0 * input.weights[0] +
    joint1 * input.weights[1] +
    joint2 * input.weights[2] +
    joint3 * input.weights[3];
  // Position of the vertex relative to our world
  let world_position = vec4f(input.position.x, input.position.y, input.position.z, 1.0);
  // Vertex position with model rotation, skinning, and the mesh's node transformation applied.
  let skinned_position = camera_uniforms.model_matrix * skin_matrix * node_uniforms.world_matrix * world_position;
  // Vertex position with only the model rotation applied.
  let rotated_position = camera_uniforms.model_matrix * world_position;
  // Determine which position to used based on whether skinMode is turnd on or off.
  let transformed_position = select(
    rotated_position,
    skinned_position,
    general_uniforms.skin_mode == 0
  );
  // Apply the camera and projection matrix transformations to our transformed position;
  output.Position = camera_uniforms.proj_matrix * camera_uniforms.view_matrix * transformed_position;
  output.normal = input.normal;
  // Convert u32 joint data to f32s to prevent flat interpolation error.
  output.joints = vec4f(f32(input.joints[0]), f32(input.joints[1]), f32(input.joints[2]), f32(input.joints[3]));
  output.weights = input.weights;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  switch general_uniforms.render_mode {
    case 1: {
      return input.joints;
    } 
    case 2: {
      return input.weights;
    }
    default: {
      return vec4f(input.normal, 1.0);
    }
  }
}struct VertexInput {
  @location(0) vert_pos: vec2f,
  @location(1) joints: vec4u,
  @location(2) weights: vec4f
}

struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) joints: vec4f,
  @location(2) weights: vec4f,
}

struct CameraUniforms {
  projMatrix: mat4x4f,
  viewMatrix: mat4x4f,
  modelMatrix: mat4x4f,
}

struct GeneralUniforms {
  render_mode: u32,
  skin_mode: u32,
}

@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
@group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
@group(2) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  var bones = vec4f(0.0, 0.0, 0.0, 0.0);
  let position = vec4f(input.vert_pos.x, input.vert_pos.y, 0.0, 1.0);
  // Get relevant 4 bone matrices
  let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
  let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
  let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
  let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];
  // Compute influence of joint based on weight
  let skin_matrix = 
    joint0 * input.weights[0] +
    joint1 * input.weights[1] +
    joint2 * input.weights[2] +
    joint3 * input.weights[3];
  // Bone transformed mesh
  output.Position = select(
    camera_uniforms.projMatrix * camera_uniforms.viewMatrix * camera_uniforms.modelMatrix * position,
    camera_uniforms.projMatrix * camera_uniforms.viewMatrix * camera_uniforms.modelMatrix * skin_matrix * position,
    general_uniforms.skin_mode == 0
  );

  //Get unadjusted world coordinates
  output.world_pos = position.xyz;
  output.joints = vec4f(f32(input.joints.x), f32(input.joints.y), f32(input.joints.z), f32(input.joints.w));
  output.weights = input.weights;
  return output;
}


@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  switch general_uniforms.render_mode {
    case 1: {
      return input.joints;
    }
    case 2: {
      return input.weights;
    }
    default: {
      return vec4f(255.0, 0.0, 1.0, 1.0); 
    }
  }
}@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_2d<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV) * fragPosition;
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Particle {
  pos : vec2f,
  vel : vec2f,
}
struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
}
struct Particles {
  particles : array<Particle>,
}
@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> particlesA : Particles;
@binding(2) @group(0) var<storage, read_write> particlesB : Particles;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
  var index = GlobalInvocationID.x;

  var vPos = particlesA.particles[index].pos;
  var vVel = particlesA.particles[index].vel;
  var cMass = vec2(0.0);
  var cVel = vec2(0.0);
  var colVel = vec2(0.0);
  var cMassCount = 0u;
  var cVelCount = 0u;
  var pos : vec2f;
  var vel : vec2f;

  for (var i = 0u; i < arrayLength(&particlesA.particles); i++) {
    if (i == index) {
      continue;
    }

    pos = particlesA.particles[i].pos.xy;
    vel = particlesA.particles[i].vel.xy;
    if (distance(pos, vPos) < params.rule1Distance) {
      cMass += pos;
      cMassCount++;
    }
    if (distance(pos, vPos) < params.rule2Distance) {
      colVel -= pos - vPos;
    }
    if (distance(pos, vPos) < params.rule3Distance) {
      cVel += vel;
      cVelCount++;
    }
  }
  if (cMassCount > 0) {
    cMass = (cMass / vec2(f32(cMassCount))) - vPos;
  }
  if (cVelCount > 0) {
    cVel /= f32(cVelCount);
  }
  vVel += (cMass * params.rule1Scale) + (colVel * params.rule2Scale) + (cVel * params.rule3Scale);

  // clamp velocity for a more pleasing simulation
  vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
  // kinematic update
  vPos = vPos + (vVel * params.deltaT);
  // Wrap around boundary
  if (vPos.x < -1.0) {
    vPos.x = 1.0;
  }
  if (vPos.x > 1.0) {
    vPos.x = -1.0;
  }
  if (vPos.y < -1.0) {
    vPos.y = 1.0;
  }
  if (vPos.y > 1.0) {
    vPos.y = -1.0;
  }
  // Write back
  particlesB.particles[index].pos = vPos;
  particlesB.particles[index].vel = vVel;
}
struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(4) color : vec4f,
}

@vertex
fn vert_main(
  @location(0) a_particlePos : vec2f,
  @location(1) a_particleVel : vec2f,
  @location(2) a_pos : vec2f
) -> VertexOutput {
  let angle = -atan2(a_particleVel.x, a_particleVel.y);
  let pos = vec2(
    (a_pos.x * cos(angle)) - (a_pos.y * sin(angle)),
    (a_pos.x * sin(angle)) + (a_pos.y * cos(angle))
  );
  
  var output : VertexOutput;
  output.position = vec4(pos + a_particlePos, 0.0, 1.0);
  output.color = vec4(
    1.0 - sin(angle + 1.0) - a_particleVel.y,
    pos.x * 100.0 - a_particleVel.y + 0.1,
    a_particleVel.x + cos(angle + 0.5),
    1.0);
  return output;
}

@fragment
fn frag_main(@location(4) color : vec4f) -> @location(0) vec4f {
  return color;
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) instance: u32
};

@vertex
fn main_vs(@location(0) position: vec4f, @builtin(instance_index) instance: u32) -> VertexOutput {
  var output: VertexOutput;

  // distribute instances into a staggered 4x4 grid
  const gridWidth = 125.0;
  const cellSize = gridWidth / 4.0;
  let row = instance / 2u;
  let col = instance % 2u;

  let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u != 0u) * cellSize;
  let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

  let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

  output.position = uniforms.modelViewProjectionMatrix * offsetPos;
  output.instance = instance;
  return output;
}

@fragment
fn main_fs(@location(0) @interpolate(flat) instance: u32) -> @location(0) vec4f {
  const colors = array<vec3f,6>(
      vec3(1.0, 0.0, 0.0),
      vec3(0.0, 1.0, 0.0),
      vec3(0.0, 0.0, 1.0),
      vec3(1.0, 0.0, 1.0),
      vec3(1.0, 1.0, 0.0),
      vec3(0.0, 1.0, 1.0),
  );

  return vec4(colors[instance % 6u], 1.0);
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
  maxStorableFragments: u32,
  targetWidth: u32,
};

struct SliceInfo {
  sliceStartY: i32
};

struct Heads {
  numFragments: atomic<u32>,
  data: array<atomic<u32>>
};

struct LinkedListElement {
  color: vec4f,
  depth: f32,
  next: u32
};

struct LinkedList {
  data: array<LinkedListElement>
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(0) var<storage, read_write> heads: Heads;
@binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
@binding(3) @group(0) var opaqueDepthTexture: texture_depth_2d;
@binding(4) @group(0) var<uniform> sliceInfo: SliceInfo;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) instance: u32
};

@vertex
fn main_vs(@location(0) position: vec4f, @builtin(instance_index) instance: u32) -> VertexOutput {
  var output: VertexOutput;

  // distribute instances into a staggered 4x4 grid
  const gridWidth = 125.0;
  const cellSize = gridWidth / 4.0;
  let row = instance / 2u;
  let col = instance % 2u;

  let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u == 0u) * cellSize;
  let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

  let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

  output.position = uniforms.modelViewProjectionMatrix * offsetPos;
  output.instance = instance;

  return output;
}

@fragment
fn main_fs(@builtin(position) position: vec4f, @location(0) @interpolate(flat) instance: u32) {
  const colors = array<vec3f,6>(
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 1.0, 1.0),
  );

  let fragCoords = vec2i(position.xy);
  let opaqueDepth = textureLoad(opaqueDepthTexture, fragCoords, 0);

  // reject fragments behind opaque objects
  if position.z >= opaqueDepth {
    discard;
  }

  // The index in the heads buffer corresponding to the head data for the fragment at
  // the current location.
  let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

  // The index in the linkedList buffer at which to store the new fragment
  let fragIndex = atomicAdd(&heads.numFragments, 1u);

  // If we run out of space to store the fragments, we just lose them
  if fragIndex < uniforms.maxStorableFragments {
    let lastHead = atomicExchange(&heads.data[headsIndex], fragIndex);
    linkedList.data[fragIndex].depth = position.z;
    linkedList.data[fragIndex].next = lastHead;
    linkedList.data[fragIndex].color = vec4(colors[(instance + 3u) % 6u], 0.3);
  }
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
  maxStorableFragments: u32,
  targetWidth: u32,
};

struct SliceInfo {
  sliceStartY: i32
};

struct Heads {
  numFragments: u32,
  data: array<u32>
};

struct LinkedListElement {
  color: vec4f,
  depth: f32,
  next: u32
};

struct LinkedList {
  data: array<LinkedListElement>
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(0) var<storage, read_write> heads: Heads;
@binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
@binding(3) @group(0) var<uniform> sliceInfo: SliceInfo;

// Output a full screen quad
@vertex
fn main_vs(@builtin(vertex_index) vertIndex: u32) -> @builtin(position) vec4f {
  const position = array<vec2f, 6>(
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
  );
  
  return vec4(position[vertIndex], 0.0, 1.0);
}

@fragment
fn main_fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let fragCoords = vec2i(position.xy);
  let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

  // The maximum layers we can process for any pixel
  const maxLayers = 12u;

  var layers: array<LinkedListElement, maxLayers>;

  var numLayers = 0u;
  var elementIndex = heads.data[headsIndex];

  // copy the list elements into an array up to the maximum amount of layers
  while elementIndex != 0xFFFFFFFFu && numLayers < maxLayers {
    layers[numLayers] = linkedList.data[elementIndex];
    numLayers++;
    elementIndex = linkedList.data[elementIndex].next;
  }

  if numLayers == 0u {
    discard;
  }
  
  // sort the fragments by depth
  for (var i = 1u; i < numLayers; i++) {
    let toInsert = layers[i];
    var j = i;

    while j > 0u && toInsert.depth > layers[j - 1u].depth {
      layers[j] = layers[j - 1u];
      j--;
    }

    layers[j] = toInsert;
  }

  // pre-multiply alpha for the first layer
  var color = vec4(layers[0].color.a * layers[0].color.rgb, layers[0].color.a);

  // blend the remaining layers
  for (var i = 1u; i < numLayers; i++) {
    let mixed = mix(color.rgb, layers[i].color.rgb, layers[i].color.aaa);
    color = vec4(mixed, color.a);
  }

  return color;
}
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_depth_2d;

override canvasSizeWidth: f32;
override canvasSizeHeight: f32;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  var result : vec4f;
  let c = coord.xy / vec2f(canvasSizeWidth, canvasSizeHeight);
  if (c.x < 0.33333) {
    let rawDepth = textureLoad(
      gBufferDepth,
      vec2i(floor(coord.xy)),
      0
    );
    // remap depth into something a bit more visible
    let depth = (1.0 - rawDepth) * 50.0;
    result = vec4(depth);
  } else if (c.x < 0.66667) {
    result = textureLoad(
      gBufferNormal,
      vec2i(floor(coord.xy)),
      0
    );
    result.x = (result.x + 1.0) * 0.5;
    result.y = (result.y + 1.0) * 0.5;
    result.z = (result.z + 1.0) * 0.5;
  } else {
    result = textureLoad(
      gBufferAlbedo,
      vec2i(floor(coord.xy)),
      0
    );
  }
  return result;
}
struct GBufferOutput {
  @location(0) normal : vec4f,

  // Textures: diffuse color, specular color, smoothness, emissive etc. could go here
  @location(1) albedo : vec4f,
}

@fragment
fn main(
  @location(0) fragNormal: vec3f,
  @location(1) fragUV : vec2f
) -> GBufferOutput {
  // faking some kind of checkerboard texture
  let uv = floor(30.0 * fragUV);
  let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));

  var output : GBufferOutput;
  output.normal = vec4(normalize(fragNormal), 1.0);
  output.albedo = vec4(c, c, c, 1.0);

  return output;
}
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_depth_2d;

struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}
@group(1) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;

struct Config {
  numLights : u32,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}
@group(1) @binding(1) var<uniform> config: Config;
@group(1) @binding(2) var<uniform> camera: Camera;

fn world_from_screen_coord(coord : vec2f, depth_sample: f32) -> vec3f {
  // reconstruct world-space position from the screen coordinate.
  let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
  let posWorldW = camera.invViewProjectionMatrix * posClip;
  let posWorld = posWorldW.xyz / posWorldW.www;
  return posWorld;
}

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  var result : vec3f;

  let depth = textureLoad(
    gBufferDepth,
    vec2i(floor(coord.xy)),
    0
  );

  // Don't light the sky.
  if (depth >= 1.0) {
    discard;
  }

  let bufferSize = textureDimensions(gBufferDepth);
  let coordUV = coord.xy / vec2f(bufferSize);
  let position = world_from_screen_coord(coordUV, depth);

  let normal = textureLoad(
    gBufferNormal,
    vec2i(floor(coord.xy)),
    0
  ).xyz;

  let albedo = textureLoad(
    gBufferAlbedo,
    vec2i(floor(coord.xy)),
    0
  ).rgb;

  for (var i = 0u; i < config.numLights; i++) {
    let L = lightsBuffer.lights[i].position.xyz - position;
    let distance = length(L);
    if (distance > lightsBuffer.lights[i].radius) {
      continue;
    }
    let lambert = max(dot(normal, normalize(L)), 0.0);
    result += vec3f(
      lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo
    );
  }

  // some manual ambient
  result += vec3(0.2);

  return vec4(result, 1.0);
}
struct Uniforms {
  modelMatrix : mat4x4f,
  normalModelMatrix : mat4x4f,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragNormal: vec3f,    // normal in world space
  @location(1) fragUV: vec2f,
}

@vertex
fn main(
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  let worldPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
  output.Position = camera.viewProjectionMatrix * vec4(worldPosition, 1.0);
  output.fragNormal = normalize((uniforms.normalModelMatrix * vec4(normal, 1.0)).xyz);
  output.fragUV = uv;
  return output;
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4f(pos[VertexIndex], 0.0, 1.0);
}
struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}
@group(0) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;

struct Config {
  numLights : u32,
}
@group(0) @binding(1) var<uniform> config: Config;

struct LightExtent {
  min : vec4f,
  max : vec4f,
}
@group(0) @binding(2) var<uniform> lightExtent: LightExtent;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
  var index = GlobalInvocationID.x;
  if (index >= config.numLights) {
    return;
  }

  lightsBuffer.lights[index].position.y = lightsBuffer.lights[index].position.y - 0.5 - 0.003 * (f32(index) - 64.0 * floor(f32(index) / 64.0));

  if (lightsBuffer.lights[index].position.y < lightExtent.min.y) {
    lightsBuffer.lights[index].position.y = lightExtent.max.y;
  }
}
// Positions for simple quad geometry
const pos = array(vec2f(0, -1), vec2f(1, -1), vec2f(0, 0), vec2f(1, 0));

struct VertexInput {
  @builtin(vertex_index) vertex : u32,
  @builtin(instance_index) instance : u32,
};

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) texcoord : vec2f,
};

struct Char {
  texOffset: vec2f,
  texExtent: vec2f,
  size: vec2f,
  offset: vec2f,
};

struct FormattedText {
  transform: mat4x4f,
  color: vec4f,
  scale: f32,
  chars: array<vec3f>,
};

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
};

// Font bindings
@group(0) @binding(0) var fontTexture: texture_2d<f32>;
@group(0) @binding(1) var fontSampler: sampler;
@group(0) @binding(2) var<storage> chars: array<Char>;

// Text bindings
@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var<storage> text: FormattedText;

@vertex
fn vertexMain(input : VertexInput) -> VertexOutput {
  let textElement = text.chars[input.instance];
  let char = chars[u32(textElement.z)];
  let charPos = (pos[input.vertex] * char.size + textElement.xy + char.offset) * text.scale;

  var output : VertexOutput;
  output.position = camera.projection * camera.view * text.transform * vec4f(charPos, 0, 1);

  output.texcoord = pos[input.vertex] * vec2f(1, -1);
  output.texcoord *= char.texExtent;
  output.texcoord += char.texOffset;
  return output;
}

fn sampleMsdf(texcoord: vec2f) -> f32 {
  let c = textureSample(fontTexture, fontSampler, texcoord);
  return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

// Antialiasing technique from Paul Houx 
// https://github.com/Chlumsky/msdfgen/issues/22#issuecomment-234958005
@fragment
fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {
  // pxRange (AKA distanceRange) comes from the msdfgen tool. Don McCurdy's tool
  // uses the default which is 4.
  let pxRange = 4.0;
  let sz = vec2f(textureDimensions(fontTexture, 0));
  let dx = sz.x*length(vec2f(dpdxFine(input.texcoord.x), dpdyFine(input.texcoord.x)));
  let dy = sz.y*length(vec2f(dpdxFine(input.texcoord.y), dpdyFine(input.texcoord.y)));
  let toPixels = pxRange * inverseSqrt(dx * dx + dy * dy);
  let sigDist = sampleMsdf(input.texcoord) - 0.5;
  let pxDist = sigDist * toPixels;

  let edgeWidth = 0.5;

  let alpha = smoothstep(-edgeWidth, edgeWidth, pxDist);

  if (alpha < 0.001) {
    discard;
  }

  return vec4f(text.color.rgb, text.color.a * alpha);
}////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////
var<private> rand_seed : vec2f;

fn init_rand(invocation_id : u32, seed : vec4f) {
  rand_seed = seed.xz;
  rand_seed = fract(rand_seed * cos(35.456+f32(invocation_id) * seed.yw));
  rand_seed = fract(rand_seed * cos(41.235+f32(invocation_id) * seed.xw));
}

fn rand() -> f32 {
  rand_seed.x = fract(cos(dot(rand_seed, vec2f(23.14077926, 232.61690225))) * 136.8168);
  rand_seed.y = fract(cos(dot(rand_seed, vec2f(54.47856553, 345.84153136))) * 534.7645);
  return rand_seed.y;
}

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
struct RenderParams {
  modelViewProjectionMatrix : mat4x4f,
  right : vec3f,
  up : vec3f
}
@binding(0) @group(0) var<uniform> render_params : RenderParams;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) color : vec4f,
  @location(2) quad_pos : vec2f, // -1..+1
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) quad_pos : vec2f, // -1..+1
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  var quad_pos = mat2x3f(render_params.right, render_params.up) * in.quad_pos;
  var position = in.position + quad_pos * 0.01;
  var out : VertexOutput;
  out.position = render_params.modelViewProjectionMatrix * vec4f(position, 1.0);
  out.color = in.color;
  out.quad_pos = in.quad_pos;
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  var color = in.color;
  // Apply a circular particle alpha mask
  color.a = color.a * max(1.0 - length(in.quad_pos), 0.0);
  return color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
struct SimulationParams {
  deltaTime : f32,
  brightnessFactor : f32,
  seed : vec4f,
}

struct Particle {
  position : vec3f,
  lifetime : f32,
  color    : vec4f,
  velocity : vec3f,
}

struct Particles {
  particles : array<Particle>,
}

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> data : Particles;
@binding(2) @group(0) var texture : texture_2d<f32>;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) global_invocation_id : vec3u) {
  let idx = global_invocation_id.x;

  init_rand(idx, sim_params.seed);

  var particle = data.particles[idx];

  // Apply gravity
  particle.velocity.z = particle.velocity.z - sim_params.deltaTime * 0.5;

  // Basic velocity integration
  particle.position = particle.position + sim_params.deltaTime * particle.velocity;

  // Age each particle. Fade out before vanishing.
  particle.lifetime = particle.lifetime - sim_params.deltaTime;
  particle.color.a = smoothstep(0.0, 0.5, particle.lifetime);

  // If the lifetime has gone negative, then the particle is dead and should be
  // respawned.
  if (particle.lifetime < 0.0) {
    // Use the probability map to find where the particle should be spawned.
    // Starting with the 1x1 mip level.
    var coord : vec2i;
    for (var level = u32(textureNumLevels(texture) - 1); level > 0; level--) {
      // Load the probability value from the mip-level
      // Generate a random number and using the probabilty values, pick the
      // next texel in the next largest mip level:
      //
      // 0.0    probabilites.r    probabilites.g    probabilites.b   1.0
      //  |              |              |              |              |
      //  |   TOP-LEFT   |  TOP-RIGHT   | BOTTOM-LEFT  | BOTTOM_RIGHT |
      //
      let probabilites = textureLoad(texture, coord, level);
      let value = vec4f(rand());
      let mask = (value >= vec4f(0.0, probabilites.xyz)) & (value < probabilites);
      coord = coord * 2;
      coord.x = coord.x + select(0, 1, any(mask.yw)); // x  y
      coord.y = coord.y + select(0, 1, any(mask.zw)); // z  w
    }
    let uv = vec2f(coord) / vec2f(textureDimensions(texture));
    particle.position = vec3f((uv - 0.5) * 3.0 * vec2f(1.0, -1.0), 0.0);
    particle.color = textureLoad(texture, coord, 0);
    particle.color.r *= sim_params.brightnessFactor;
    particle.color.g *= sim_params.brightnessFactor;
    particle.color.b *= sim_params.brightnessFactor;
    particle.velocity.x = (rand() - 0.5) * 0.1;
    particle.velocity.y = (rand() - 0.5) * 0.1;
    particle.velocity.z = rand() * 0.3;
    particle.lifetime = 0.5 + rand() * 3.0;
  }

  // Store the new particle value
  data.particles[idx] = particle;
}
struct UBO {
  width : u32,
}

@binding(0) @group(0) var<uniform> ubo : UBO;
@binding(1) @group(0) var<storage, read> buf_in : array<f32>;
@binding(2) @group(0) var<storage, read_write> buf_out : array<f32>;
@binding(3) @group(0) var tex_in : texture_2d<f32>;
@binding(3) @group(0) var tex_out : texture_storage_2d<rgba8unorm, write>;

////////////////////////////////////////////////////////////////////////////////
// import_level
//
// Loads the alpha channel from a texel of the source image, and writes it to
// the buf_out.weights.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn import_level(@builtin(global_invocation_id) coord : vec3u) {
  _ = &buf_in; // so the bindGroups are similar.
  if (!all(coord.xy < vec2u(textureDimensions(tex_in)))) {
    return;
  }

  let offset = coord.x + coord.y * ubo.width;
  buf_out[offset] = textureLoad(tex_in, vec2i(coord.xy), 0).w;
}

////////////////////////////////////////////////////////////////////////////////
// export_level
//
// Loads 4 f32 weight values from buf_in.weights, and stores summed value into
// buf_out.weights, along with the calculated 'probabilty' vec4 values into the
// mip level of tex_out. See simulate() in particle.wgsl to understand the
// probability logic.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn export_level(@builtin(global_invocation_id) coord : vec3u) {
  if (!all(coord.xy < vec2u(textureDimensions(tex_out)))) {
    return;
  }

  let dst_offset = coord.x    + coord.y    * ubo.width;
  let src_offset = coord.x*2u + coord.y*2u * ubo.width;

  let a = buf_in[src_offset + 0u];
  let b = buf_in[src_offset + 1u];
  let c = buf_in[src_offset + 0u + ubo.width];
  let d = buf_in[src_offset + 1u + ubo.width];
  let sum = a + b + c + d;

  buf_out[dst_offset] = sum / 4.0;

  let probabilities = vec4f(a, a+b, a+b+c, sum) / max(sum, 0.0001);
  textureStore(tex_out, vec2i(coord.xy), probabilities);
}
const modeAlbedoTexture = 0;
const modeNormalTexture = 1;
const modeDepthTexture = 2;
const modeNormalMap = 3;
const modeParallaxScale = 4;
const modeSteepParallax = 5;

struct SpaceTransforms {
  worldViewProjMatrix: mat4x4f,
  worldViewMatrix: mat4x4f,
}

struct MapInfo {
  lightPosVS: vec3f, // Light position in view space
  mode: u32,
  lightIntensity: f32,
  depthScale: f32,
  depthLayers: f32,
}

struct VertexInput {
  // Shader assumes the missing 4th float is 1.0
  @location(0) position : vec4f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f,
  @location(3) vert_tan: vec3f,
  @location(4) vert_bitan: vec3f,
}

struct VertexOutput {
  @builtin(position) posCS : vec4f,    // vertex position in clip space
  @location(0) posVS : vec3f,          // vertex position in view space
  @location(1) tangentVS: vec3f,       // vertex tangent in view space
  @location(2) bitangentVS: vec3f,     // vertex tangent in view space
  @location(3) normalVS: vec3f,        // vertex normal in view space
  @location(5) uv : vec2f,             // vertex texture coordinate
}

// Uniforms
@group(0) @binding(0) var<uniform> spaceTransform : SpaceTransforms;
@group(0) @binding(1) var<uniform> mapInfo: MapInfo;

// Texture info
@group(1) @binding(0) var textureSampler: sampler;
@group(1) @binding(1) var albedoTexture: texture_2d<f32>;
@group(1) @binding(2) var normalTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_2d<f32>;


@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output : VertexOutput;

  output.posCS = spaceTransform.worldViewProjMatrix * input.position;
  output.posVS = (spaceTransform.worldViewMatrix * input.position).xyz;
  output.tangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_tan, 0)).xyz;
  output.bitangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_bitan, 0)).xyz;
  output.normalVS = (spaceTransform.worldViewMatrix * vec4(input.normal, 0)).xyz;
  output.uv = input.uv;

  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Build the matrix to convert from tangent space to view space
  let tangentToView = mat3x3f(
    input.tangentVS,
    input.bitangentVS,
    input.normalVS,
  );

  // The inverse of a non-scaling affine 3x3 matrix is it's transpose
  let viewToTangent = transpose(tangentToView);

  // Calculate the normalized vector in tangent space from the camera to the fragment
  let viewDirTS = normalize(viewToTangent * input.posVS);

  // Apply parallax to the texture coordinate, if parallax is enabled
  var uv : vec2f;
  switch (mapInfo.mode) {
    case modeParallaxScale: {
      uv = parallaxScale(input.uv, viewDirTS);
      break;
    }
    case modeSteepParallax: {
      uv = parallaxSteep(input.uv, viewDirTS);
      break;
    }
    default: {
      uv = input.uv;
      break;
    }
  }

  // Sample the albedo texture
  let albedoSample = textureSample(albedoTexture, textureSampler, uv);

  // Sample the normal texture
  let normalSample = textureSample(normalTexture, textureSampler, uv);

  switch (mapInfo.mode) {
    case modeAlbedoTexture: { // Output the albedo sample
      return albedoSample;
    }
    case modeNormalTexture: { // Output the normal sample
      return normalSample;
    }
    case modeDepthTexture: { // Output the depth map
      return textureSample(depthTexture, textureSampler, input.uv);
    }
    default: {
      // Transform the normal sample to a tangent space normal
      let normalTS = normalSample.xyz * 2 - 1;

      // Convert normal from tangent space to view space, and normalize
      let normalVS = normalize(tangentToView * normalTS);

      // Calculate the vector in view space from the light position to the fragment
      let fragToLightVS = mapInfo.lightPosVS - input.posVS;

      // Calculate the square distance from the light to the fragment
      let lightSqrDist = dot(fragToLightVS, fragToLightVS);

      // Calculate the normalized vector in view space from the fragment to the light
      let lightDirVS = fragToLightVS * inverseSqrt(lightSqrDist);

      // Light strength is inversely proportional to square of distance from light
      let diffuseLight = mapInfo.lightIntensity * max(dot(lightDirVS, normalVS), 0) / lightSqrDist;

      // The diffuse is the albedo color multiplied by the diffuseLight
      let diffuse = albedoSample.rgb * diffuseLight;

      return vec4f(diffuse, 1.0);
    }
  }
}


// Returns the uv coordinate displaced in the view direction by a magnitude calculated by the depth
// sampled from the depthTexture and the angle between the surface normal and view direction.
fn parallaxScale(uv: vec2f, viewDirTS: vec3f) -> vec2f {
  let depthSample = textureSample(depthTexture, textureSampler, uv).r;
  return uv + viewDirTS.xy * (depthSample * mapInfo.depthScale) / -viewDirTS.z;
}

// Returns the uv coordinates displaced in the view direction by ray-tracing the depth map.
fn parallaxSteep(startUV: vec2f, viewDirTS: vec3f) -> vec2f {
  // Calculate derivatives of the texture coordinate, so we can sample the texture with non-uniform
  // control flow.
  let ddx = dpdx(startUV);
  let ddy = dpdy(startUV);

  // Calculate the delta step in UV and depth per iteration
  let uvDelta = viewDirTS.xy * mapInfo.depthScale / (-viewDirTS.z * mapInfo.depthLayers);
  let depthDelta = 1.0 / f32(mapInfo.depthLayers);
  let posDelta = vec3(uvDelta, depthDelta);

  // Walk the depth texture, and stop when the ray intersects the depth map
  var pos = vec3(startUV, 0);
  for (var i = 0; i < 32; i++) {
    if (pos.z >= textureSampleGrad(depthTexture, textureSampler, pos.xy, ddx, ddy).r) {
      break; // Hit the surface
    }
    pos += posDelta;
  }

  return pos.xy;
}
@fragment
fn main(@location(0) cell: f32) -> @location(0) vec4f {
  return vec4f(cell, cell, cell, 1.);
}
@binding(0) @group(0) var<storage, read> size: vec2u;
@binding(1) @group(0) var<storage, read> current: array<u32>;
@binding(2) @group(0) var<storage, read_write> next: array<u32>;

override blockSize = 8;

fn getIndex(x: u32, y: u32) -> u32 {
  let h = size.y;
  let w = size.x;

  return (y % h) * w + (x % w);
}

fn getCell(x: u32, y: u32) -> u32 {
  return current[getIndex(x, y)];
}

fn countNeighbors(x: u32, y: u32) -> u32 {
  return getCell(x - 1, y - 1) + getCell(x, y - 1) + getCell(x + 1, y - 1) + 
         getCell(x - 1, y) +                         getCell(x + 1, y) + 
         getCell(x - 1, y + 1) + getCell(x, y + 1) + getCell(x + 1, y + 1);
}

@compute @workgroup_size(blockSize, blockSize)
fn main(@builtin(global_invocation_id) grid: vec3u) {
  let x = grid.x;
  let y = grid.y;
  let n = countNeighbors(x, y);
  next[getIndex(x, y)] = select(u32(n == 3u), u32(n == 2u || n == 3u), getCell(x, y) == 1u); 
} 
struct Out {
  @builtin(position) pos: vec4f,
  @location(0) cell: f32,
}

@binding(0) @group(0) var<uniform> size: vec2u;

@vertex
fn main(@builtin(instance_index) i: u32, @location(0) cell: u32, @location(1) pos: vec2u) -> Out {
  let w = size.x;
  let h = size.y;
  let x = (f32(i % w + pos.x) / f32(w) - 0.5) * 2. * f32(w) / f32(max(w, h));
  let y = (f32((i - (i % w)) / w + pos.y) / f32(h) - 0.5) * 2. * f32(h) / f32(max(w, h));

  return Out(vec4f(x, y, 0., 1.), f32(cell));
}
// The lightmap data
@group(1) @binding(0) var lightmap : texture_2d_array<f32>;

// The sampler used to sample the lightmap
@group(1) @binding(1) var smpl : sampler;

// The output framebuffer
@group(1) @binding(2) var framebuffer : texture_storage_2d<rgba16float, write>;

override WorkgroupSizeX : u32;
override WorkgroupSizeY : u32;

const NumReflectionRays = 5;

@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
  if (all(invocation_id.xy < textureDimensions(framebuffer))) {
    init_rand(invocation_id);

    // Calculate the fragment's NDC coordinates for the intersection of the near
    // clip plane and far clip plane
    let uv = vec2f(invocation_id.xy) / vec2f(textureDimensions(framebuffer).xy);
    let ndcXY = (uv - 0.5) * vec2(2, -2);

    // Transform the coordinates back into world space
    var near = common_uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
    var far = common_uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
    near /= near.w;
    far /= far.w;

    // Create a ray that starts at the near clip plane, heading in the fragment's
    // z-direction, and raytrace to find the nearest quad that the ray intersects.
    let ray = Ray(near.xyz, normalize(far.xyz - near.xyz));
    let hit = raytrace(ray);

    let hit_color = sample_hit(hit);
    var normal = quads[hit.quad].plane.xyz;

    // Fire a few rays off the surface to collect some reflections
    let bounce = reflect(ray.dir, normal);
    var reflection : vec3f;
    for (var i = 0; i < NumReflectionRays; i++) {
      let reflection_dir = normalize(bounce + rand_unit_sphere()*0.1);
      let reflection_ray = Ray(hit.pos + bounce * 1e-5, reflection_dir);
      let reflection_hit = raytrace(reflection_ray);
      reflection += sample_hit(reflection_hit);
    }
    let color = mix(reflection / NumReflectionRays, hit_color, 0.95);

    textureStore(framebuffer, invocation_id.xy, vec4(color, 1));
  }
}


// Returns the sampled hit quad's lightmap at 'hit.uv', and adds the quad's
// emissive value.
fn sample_hit(hit : HitInfo) -> vec3f {
  let quad = quads[hit.quad];
  // Sample the quad's lightmap, and add emissive.
  return textureSampleLevel(lightmap, smpl, hit.uv, hit.quad, 0).rgb +
         quad.emissive * quad.color;
}
const pi = 3.14159265359;

// Quad describes 2D rectangle on a plane
struct Quad {
  // The surface plane
  plane    : vec4f,
  // A plane with a normal in the 'u' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'right' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  right    : vec4f,
  // A plane with a normal in the 'v' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'up' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  up       : vec4f,
  // The diffuse color of the quad
  color    : vec3f,
  // Emissive value. 0=no emissive, 1=full emissive.
  emissive : f32,
};

// Ray is a start point and direction.
struct Ray {
  start : vec3f,
  dir   : vec3f,
}

// Value for HitInfo.quad if no intersection occured.
const kNoHit = 0xffffffff;

// HitInfo describes the hit location of a ray-quad intersection
struct HitInfo {
  // Distance along the ray to the intersection
  dist : f32,
  // The quad index that was hit
  quad : u32,
  // The position of the intersection
  pos : vec3f,
  // The UVs of the quad at the point of intersection
  uv : vec2f,
}

// CommonUniforms uniform buffer data
struct CommonUniforms {
  // Model View Projection matrix
  mvp : mat4x4f,
  // Inverse of mvp
  inv_mvp : mat4x4f,
  // Random seed for the workgroup
  seed : vec3u,
}

// The common uniform buffer binding.
@group(0) @binding(0) var<uniform> common_uniforms : CommonUniforms;

// The quad buffer binding.
@group(0) @binding(1) var<storage> quads : array<Quad>;

// intersect_ray_quad will check to see if the ray 'r' intersects the quad 'q'.
// If an intersection occurs, and the intersection is closer than 'closest' then
// the intersection information is returned, otherwise 'closest' is returned.
fn intersect_ray_quad(r : Ray, quad : u32, closest : HitInfo) -> HitInfo {
  let q = quads[quad];
  let plane_dist = dot(q.plane, vec4(r.start, 1));
  let ray_dist = plane_dist / -dot(q.plane.xyz, r.dir);
  let pos = r.start + r.dir * ray_dist;
  let uv = vec2(dot(vec4f(pos, 1), q.right),
                dot(vec4f(pos, 1), q.up)) * 0.5 + 0.5;
  let hit = plane_dist > 0 &&
            ray_dist > 0 &&
            ray_dist < closest.dist &&
            all((uv > vec2f()) & (uv < vec2f(1)));
  return HitInfo(
    select(closest.dist, ray_dist, hit),
    select(closest.quad, quad,     hit),
    select(closest.pos,  pos,      hit),
    select(closest.uv,   uv,       hit),
  );
}

// raytrace finds the closest intersecting quad for the given ray
fn raytrace(ray : Ray) -> HitInfo {
  var hit = HitInfo();
  hit.dist = 1e20;
  hit.quad = kNoHit;
  for (var quad = 0u; quad < arrayLength(&quads); quad++) {
    hit = intersect_ray_quad(ray, quad, hit);
  }
  return hit;
}

// A psuedo random number. Initialized with init_rand(), updated with rand().
var<private> rnd : vec3u;

// Initializes the random number generator.
fn init_rand(invocation_id : vec3u) {
  const A = vec3(1741651 * 1009,
                 140893  * 1609 * 13,
                 6521    * 983  * 7 * 2);
  rnd = (invocation_id * A) ^ common_uniforms.seed;
}

// Returns a random number between 0 and 1.
fn rand() -> f32 {
  const C = vec3(60493  * 9377,
                 11279  * 2539 * 23,
                 7919   * 631  * 5 * 3);

  rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
  return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
}

// Returns a random point within a unit sphere centered at (0,0,0).
fn rand_unit_sphere() -> vec3f {
    var u = rand();
    var v = rand();
    var theta = u * 2.0 * pi;
    var phi = acos(2.0 * v - 1.0);
    var r = pow(rand(), 1.0/3.0);
    var sin_theta = sin(theta);
    var cos_theta = cos(theta);
    var sin_phi = sin(phi);
    var cos_phi = cos(phi);
    var x = r * sin_phi * sin_theta;
    var y = r * sin_phi * cos_theta;
    var z = r * cos_phi;
    return vec3f(x, y, z);
}

fn rand_concentric_disk() -> vec2f {
    let u = vec2f(rand(), rand());
    let uOffset = 2.f * u - vec2f(1, 1);

    if (uOffset.x == 0 && uOffset.y == 0){
        return vec2f(0, 0);
    }

    var theta = 0.0;
    var r = 0.0;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (pi / 4) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (pi / 2) - (pi / 4) * (uOffset.x / uOffset.y);
    }
    return r * vec2f(cos(theta), sin(theta));
}

fn rand_cosine_weighted_hemisphere() -> vec3f {
    let d = rand_concentric_disk();
    let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return vec3f(d.x, d.y, z);
}
// The lightmap data
@group(1) @binding(0) var lightmap : texture_2d_array<f32>;

// The sampler used to sample the lightmap
@group(1) @binding(1) var smpl : sampler;

// Vertex shader input data
struct VertexIn {
  @location(0) position : vec4f,
  @location(1) uv : vec3f,
  @location(2) emissive : vec3f,
}

// Vertex shader output data
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) emissive : vec3f,
  @interpolate(flat)
  @location(2) quad : u32,
}

@vertex
fn vs_main(input : VertexIn) -> VertexOut {
  var output : VertexOut;
  output.position = common_uniforms.mvp * input.position;
  output.uv = input.uv.xy;
  output.quad = u32(input.uv.z + 0.5);
  output.emissive = input.emissive;
  return output;
}

@fragment
fn fs_main(vertex_out : VertexOut) -> @location(0) vec4f {
  return textureSample(lightmap, smpl, vertex_out.uv, vertex_out.quad) + vec4f(vertex_out.emissive, 1);
}
// A storage buffer holding an array of atomic<u32>.
// The array elements are a sequence of red, green, blue components, for each
// lightmap texel, for each quad surface.
@group(1) @binding(0)
var<storage, read_write> accumulation : array<atomic<u32>>;

// The output lightmap texture.
@group(1) @binding(1)
var lightmap : texture_storage_2d_array<rgba16float, write>;

// Uniform data used by the accumulation_to_lightmap entry point
struct Uniforms {
  // Scalar for converting accumulation values to output lightmap values
  accumulation_to_lightmap_scale : f32,
  // Accumulation buffer rescaling value
  accumulation_buffer_scale : f32,
  // The width of the light
  light_width : f32,
  // The height of the light
  light_height : f32,
  // The center of the light
  light_center : vec3f,
}

// accumulation_to_lightmap uniforms binding point
@group(1) @binding(2) var<uniform> uniforms : Uniforms;

// Number of photons emitted per workgroup
override PhotonsPerWorkgroup : u32;

// Maximum value that can be added to the accumulation buffer from a single photon
override PhotonEnergy : f32;

// Number of bounces of each photon
const PhotonBounces = 4;

// Amount of light absorbed with each photon bounce (0: 0%, 1: 100%)
const LightAbsorbtion = 0.5;

// Radiosity compute shader.
// Each invocation creates a photon from the light source, and accumulates
// bounce lighting into the 'accumulation' buffer.
@compute @workgroup_size(PhotonsPerWorkgroup)
fn radiosity(@builtin(global_invocation_id) invocation_id : vec3u) {
  init_rand(invocation_id);
  photon();
}

// Spawns a photon at the light source, performs ray tracing in the scene,
// accumulating light values into 'accumulation' for each quad surface hit.
fn photon() {
  // Create a random ray from the light.
  var ray = new_light_ray();
  // Give the photon an initial energy value.
  var color = PhotonEnergy * vec3f(1, 0.8, 0.6);

  // Start bouncing.
  for (var i = 0; i < (PhotonBounces+1); i++) {
    // Find the closest hit of the ray with the scene's quads.
    let hit = raytrace(ray);
    let quad = quads[hit.quad];

    // Bounce the ray.
    ray.start = hit.pos + quad.plane.xyz * 1e-5;
    ray.dir = normalize(reflect(ray.dir, quad.plane.xyz) + rand_unit_sphere() * 0.75);

    // Photon color is multiplied by the quad's color.
    color *= quad.color;

    // Accumulate the aborbed light into the 'accumulation' buffer.
    accumulate(hit.uv, hit.quad, color * LightAbsorbtion);

    // What wasn't absorbed is reflected.
    color *= 1 - LightAbsorbtion;
  }
}

// Performs an atomicAdd() with 'color' into the 'accumulation' buffer at 'uv'
// and 'quad'.
fn accumulate(uv : vec2f, quad : u32, color : vec3f) {
  let dims = textureDimensions(lightmap);
  let base_idx = accumulation_base_index(vec2u(uv * vec2f(dims)), quad);
  atomicAdd(&accumulation[base_idx + 0], u32(color.r + 0.5));
  atomicAdd(&accumulation[base_idx + 1], u32(color.g + 0.5));
  atomicAdd(&accumulation[base_idx + 2], u32(color.b + 0.5));
}

// Returns the base element index for the texel at 'coord' for 'quad'
fn accumulation_base_index(coord : vec2u, quad : u32) -> u32 {
  let dims = textureDimensions(lightmap);
  let c = min(vec2u(dims) - 1, coord);
  return 3 * (c.x + dims.x * c.y + dims.x * dims.y * quad);
}

// Returns a new Ray at a random point on the light, in a random downwards
// direction.
fn new_light_ray() -> Ray {
  let center = uniforms.light_center;
  let pos = center + vec3f(uniforms.light_width * (rand() - 0.5),
                           0,
                           uniforms.light_height * (rand() - 0.5));
  var dir = rand_cosine_weighted_hemisphere().xzy;
  dir.y = -dir.y;
  return Ray(pos, dir);
}

override AccumulationToLightmapWorkgroupSizeX : u32;
override AccumulationToLightmapWorkgroupSizeY : u32;

// Compute shader used to copy the atomic<u32> data in 'accumulation' to
// 'lightmap'. 'accumulation' might also be scaled to reduce integer overflow.
@compute @workgroup_size(AccumulationToLightmapWorkgroupSizeX, AccumulationToLightmapWorkgroupSizeY)
fn accumulation_to_lightmap(@builtin(global_invocation_id) invocation_id : vec3u,
                            @builtin(workgroup_id)         workgroup_id  : vec3u) {
  let dims = textureDimensions(lightmap);
  let quad = workgroup_id.z; // The workgroup 'z' value holds the quad index.
  let coord = invocation_id.xy;
  if (all(coord < dims)) {
    // Load the color value out of 'accumulation'
    let base_idx = accumulation_base_index(coord, quad);
    let color = vec3(f32(atomicLoad(&accumulation[base_idx + 0])),
                     f32(atomicLoad(&accumulation[base_idx + 1])),
                     f32(atomicLoad(&accumulation[base_idx + 2])));

    // Multiply the color by 'uniforms.accumulation_to_lightmap_scale' and write it to
    // the lightmap.
    textureStore(lightmap, coord, quad, vec4(color * uniforms.accumulation_to_lightmap_scale, 1));

    // If the 'accumulation' buffer is nearing saturation, then
    // 'uniforms.accumulation_buffer_scale' will be less than 1, scaling the values
    // to something less likely to overflow the u32.
    if (uniforms.accumulation_buffer_scale != 1.0) {
      let scaled = color * uniforms.accumulation_buffer_scale + 0.5;
      atomicStore(&accumulation[base_idx + 0], u32(scaled.r));
      atomicStore(&accumulation[base_idx + 1], u32(scaled.g));
      atomicStore(&accumulation[base_idx + 2], u32(scaled.b));
    }
  }
}
// The linear-light input framebuffer
@group(0) @binding(0) var input  : texture_2d<f32>;

// The tonemapped, gamma-corrected output framebuffer
@group(0) @binding(1) var output : texture_storage_2d<{OUTPUT_FORMAT}, write>;

const TonemapExposure = 0.5;

const Gamma = 2.2;

override WorkgroupSizeX : u32;
override WorkgroupSizeY : u32;

@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
  let color = textureLoad(input, invocation_id.xy, 0).rgb;
  let tonemapped = reinhard_tonemap(color);
  textureStore(output, invocation_id.xy, vec4f(tonemapped, 1));
}

fn reinhard_tonemap(linearColor: vec3f) -> vec3f {
  let color = linearColor * TonemapExposure;
  let mapped = color / (1+color);
  return pow(mapped, vec3f(1 / Gamma));
}
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_cube<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  // Our camera and the skybox cube are both centered at (0, 0, 0)
  // so we can use the cube geometry position to get viewing vector to sample
  // the cube texture. The magnitude of the vector doesn't matter.
  var cubemapVec = fragPosition.xyz - vec3(0.5);
  // When viewed from the inside, cubemaps are left-handed (z away from viewer),
  // but common camera matrix convention results in a right-handed world space
  // (z toward viewer), so we have to flip it.
  cubemapVec.z *= -1;
  return textureSample(myTexture, mySampler, cubemapVec);
}
@binding(1) @group(0) var mySampler: sampler;
@binding(2) @group(0) var myTexture: texture_2d<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  let texColor = textureSample(myTexture, mySampler, fragUV * 0.8 + vec2(0.1));
  let f = select(1.0, 0.0, length(texColor.rgb - vec3(0.5)) < 0.01);
  return f * texColor + (1.0 - f) * fragPosition;
}
struct Uniforms {
  inverseModelViewProjectionMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_3d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) near : vec3f,
  @location(1) step : vec3f,
}

const NumSteps = 64u;

@vertex
fn vertex_main(
  @builtin(vertex_index) VertexIndex : u32
) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2(-1.0, 3.0),
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0)
  );
  var xy = pos[VertexIndex];
  var near = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 0.0, 1);
  var far = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 1, 1);
  near /= near.w;
  far /= far.w;
  return VertexOutput(
    vec4f(xy, 0.0, 1.0),
    near.xyz,
    (far.xyz - near.xyz) / f32(NumSteps)
  );
}

@fragment
fn fragment_main(
  @location(0) near: vec3f,
  @location(1) step: vec3f
) -> @location(0) vec4f {
  var rayPos = near;
  var result = 0.0;
  for (var i = 0u; i < NumSteps; i++) {
    let texCoord = (rayPos.xyz + 1.0) * 0.5;
    let sample =
      textureSample(myTexture, mySampler, texCoord).r * 4.0 / f32(NumSteps);
    let intersects =
      all(rayPos.xyz < vec3f(1.0)) && all(rayPos.xyz > vec3f(-1.0));
    result += select(0.0, (1.0 - result) * sample, intersects && result < 1.0);
    rayPos += step;
  }
  return vec4f(vec3f(result), 1.0);
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct LineUniforms {
  stride: u32,
  thickness: f32,
  alphaThreshold: f32,
};

struct VSOut {
  @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<uniform> line: LineUniforms;

@vertex fn vsIndexedU32(@builtin(vertex_index) vNdx: u32) -> VSOut {
  // indices make a triangle so for every 3 indices we need to output
  // 6 values
  let triNdx = vNdx / 6;
  // 0 1 0 1 0 1  0 1 0 1 0 1  vNdx % 2
  // 0 0 1 1 2 2  3 3 4 4 5 5  vNdx / 2
  // 0 1 1 2 2 3  3 4 4 5 5 6  vNdx % 2 + vNdx / 2
  // 0 1 1 2 2 0  0 1 1 2 2 0  (vNdx % 2 + vNdx / 2) % 3
  let vertNdx = (vNdx % 2 + vNdx / 2) % 3;
  let index = indices[triNdx * 3 + vertNdx];

  // note:
  //
  // * if your indices are U16 you could use this
  //
  //    let indexNdx = triNdx * 3 + vertNdx;
  //    let twoIndices = indices[indexNdx / 2];  // indices is u32 but we want u16
  //    let index = (twoIndices >> ((indexNdx & 1) * 16)) & 0xFFFF;
  //
  // * if you're not using indices you could use this
  //
  //    let index = triNdx * 3 + vertNdx;

  let pNdx = index * line.stride;
  let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * position;
  return vOut;
}

@fragment fn fs() -> @location(0) vec4f {
  return uni.color + vec4f(0.5);
}

struct BarycentricCoordinateBasedVSOutput {
  @builtin(position) position: vec4f,
  @location(0) barycenticCoord: vec3f,
};

@vertex fn vsIndexedU32BarycentricCoordinateBasedLines(
  @builtin(vertex_index) vNdx: u32
) -> BarycentricCoordinateBasedVSOutput {
  let vertNdx = vNdx % 3;
  let index = indices[vNdx];

  // note:
  //
  // * if your indices are U16 you could use this
  //
  //    let twoIndices = indices[vNdx / 2];  // indices is u32 but we want u16
  //    let index = (twoIndices >> ((vNdx & 1) * 16)) & 0xFFFF;
  //
  // * if you're not using indices you could use this
  //
  //    let index = vNdx;

  let pNdx = index * line.stride;
  let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

  var vsOut: BarycentricCoordinateBasedVSOutput;
  vsOut.position = uni.worldViewProjectionMatrix * position;

  // emit a barycentric coordinate
  vsOut.barycenticCoord = vec3f(0);
  vsOut.barycenticCoord[vertNdx] = 1.0;
  return vsOut;
}

fn edgeFactor(bary: vec3f) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * line.thickness, bary);
  return min(min(a3.x, a3.y), a3.z);
}

@fragment fn fsBarycentricCoordinateBasedLines(
  v: BarycentricCoordinateBasedVSOutput
) -> @location(0) vec4f {
  let a = 1.0 - edgeFactor(v.barycenticCoord);
  if (a < line.alphaThreshold) {
    discard;
  }

  return vec4((uni.color.rgb + 0.5) * a, a);
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Uniforms {
  modelViewProjectionMatrix : array<mat4x4f, 16>,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix[instanceIdx] * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0));
  return output;
}
@fragment
fn main(
  @location(0) fragColor: vec4f
) -> @location(0) vec4f {
  return fragColor;
}
@group(0) @binding(0) var depthTexture: texture_depth_2d;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0);
  return vec4f(depthValue, depthValue, depthValue, 1.0);
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4(pos[VertexIndex], 0.0, 1.0);
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragColor : vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  output.fragColor = color;
  return output;
}@group(1) @binding(0) var depthTexture: texture_depth_2d;

@fragment
fn main(
  @builtin(position) coord: vec4f,
  @location(0) clipPos: vec4f
) -> @location(0) vec4f {
  let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0);
  let v : f32 = abs(clipPos.z / clipPos.w - depthValue) * 2000000.0;
  return vec4f(v, v, v, 1.0);
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) clipPos : vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  output.clipPos = output.Position;
  return output;
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f
) -> @builtin(position) vec4f {
  return camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
}
@fragment
fn main() -> @location(0) vec4f {
  return vec4(0.0, 0.0, 0.0, 1.0);
}struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}
@group(0) @binding(0) var mySampler : sampler;
@group(0) @binding(1) var myTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

@fragment
fn frag_main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV);
}
@fragment
fn main() -> @location(0) vec4f {
  return vec4(1.0, 0.0, 0.0, 1.0);
}@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  return fragPosition;
}
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_external;

@fragment
fn main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  return textureSampleBaseClampToEdge(myTexture, mySampler, fragUV);
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2(0.0, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5)
  );

  return vec4f(pos[VertexIndex], 0.0, 1.0);
}
struct Uniforms {
  viewProjectionMatrix : mat4x4f
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@group(1) @binding(0) var<uniform> modelMatrix : mat4x4f;

struct VertexInput {
  @location(0) position : vec4f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) normal: vec3f,
  @location(1) uv : vec2f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.viewProjectionMatrix * modelMatrix * input.position;
  output.normal = normalize((modelMatrix * vec4(input.normal, 0)).xyz);
  output.uv = input.uv;
  return output;
}

@group(1) @binding(1) var meshSampler: sampler;
@group(1) @binding(2) var meshTexture: texture_2d<f32>;

// Static directional lighting
const lightDir = vec3f(1, 1, 1);
const dirColor = vec3(1);
const ambientColor = vec3f(0.05);

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let textureColor = textureSample(meshTexture, meshSampler, input.uv);

  // Very simplified lighting algorithm.
  let lightColor = saturate(ambientColor + max(dot(input.normal, lightDir), 0.0) * dirColor);

  return vec4f(textureColor.rgb * lightColor, textureColor.a);
}
override shadowDepthTextureSize: f32 = 1024.0;

struct Scene {
  lightViewProjMatrix : mat4x4f,
  cameraViewProjMatrix : mat4x4f,
  lightPos : vec3f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct FragmentInput {
  @location(0) shadowPos : vec3f,
  @location(1) fragPos : vec3f,
  @location(2) fragNorm : vec3f,
}

const albedo = vec3f(0.9);
const ambientFactor = 0.2;

@fragment
fn main(input : FragmentInput) -> @location(0) vec4f {
  // Percentage-closer filtering. Sample texels in the region
  // to smooth the result.
  var visibility = 0.0;
  let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2f(vec2(x, y)) * oneOverShadowDepthTextureSize;

      visibility += textureSampleCompare(
        shadowMap, shadowSampler,
        input.shadowPos.xy + offset, input.shadowPos.z - 0.007
      );
    }
  }
  visibility /= 9.0;

  let lambertFactor = max(dot(normalize(scene.lightPos - input.fragPos), normalize(input.fragNorm)), 0.0);
  let lightingFactor = min(ambientFactor + visibility * lambertFactor, 1.0);

  return vec4(lightingFactor * albedo, 1.0);
}
struct Scene {
  lightViewProjMatrix: mat4x4f,
  cameraViewProjMatrix: mat4x4f,
  lightPos: vec3f,
}

struct Model {
  modelMatrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

struct VertexOutput {
  @location(0) shadowPos: vec3f,
  @location(1) fragPos: vec3f,
  @location(2) fragNorm: vec3f,

  @builtin(position) Position: vec4f,
}

@vertex
fn main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f
) -> VertexOutput {
  var output : VertexOutput;

  // XY is in (-1, 1) space, Z is in (0, 1) space
  let posFromLight = scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);

  // Convert XY to (0, 1)
  // Y is flipped because texture coords are Y-down.
  output.shadowPos = vec3(
    posFromLight.xy * vec2(0.5, -0.5) + vec2(0.5),
    posFromLight.z
  );

  output.Position = scene.cameraViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
  output.fragPos = output.Position.xyz;
  output.fragNorm = normal;
  return output;
}
struct Scene {
  lightViewProjMatrix: mat4x4f,
  cameraViewProjMatrix: mat4x4f,
  lightPos: vec3f,
}

struct Model {
  modelMatrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn main(
  @location(0) position: vec3f
) -> @builtin(position) vec4f {
  return scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
}
struct Time {
  value : f32,
}

struct Uniforms {
  scale : f32,
  offsetX : f32,
  offsetY : f32,
  scalar : f32,
  scalarOffset : f32,
}

@binding(0) @group(0) var<uniform> time : Time;
@binding(0) @group(1) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) v_color : vec4f,
}

@vertex
fn vert_main(
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var fade = (uniforms.scalarOffset + time.value * uniforms.scalar / 10.0) % 1.0;
  if (fade < 0.5) {
    fade = fade * 2.0;
  } else {
    fade = (1.0 - fade) * 2.0;
  }
  var xpos = position.x * uniforms.scale;
  var ypos = position.y * uniforms.scale;
  var angle = 3.14159 * 2.0 * fade;
  var xrot = xpos * cos(angle) - ypos * sin(angle);
  var yrot = xpos * sin(angle) + ypos * cos(angle);
  xpos = xrot + uniforms.offsetX;
  ypos = yrot + uniforms.offsetY;

  var output : VertexOutput;
  output.v_color = vec4(fade, 1.0 - fade, 0.0, 1.0) + color;
  output.Position = vec4(xpos, ypos, 0.0, 1.0);
  return output;
}

@fragment
fn frag_main(@location(0) v_color : vec4f) -> @location(0) vec4f {
  return v_color;
}
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vertex_main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  return VertexOutput(uniforms.modelViewProjectionMatrix * position, uv);
}

@fragment
fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV);
}
@group(0) @binding(0) var tex: texture_2d<f32>;

struct Varying {
  @builtin(position) pos: vec4f,
  @location(0) texelCoord: vec2f,
  @location(1) mipLevel: f32,
}

const kMipLevels = 4;
const baseMipSize: u32 = 16;

@vertex
fn vmain(
  @builtin(instance_index) instance_index: u32, // used as mipLevel
  @builtin(vertex_index) vertex_index: u32,
) -> Varying {
  var square = array(
    vec2f(0, 0), vec2f(0, 1), vec2f(1, 0),
    vec2f(1, 0), vec2f(0, 1), vec2f(1, 1),
  );
  let uv = square[vertex_index];
  let pos = vec4(uv * 2 - vec2(1, 1), 0.0, 1.0);

  let mipLevel = instance_index;
  let mipSize = f32(1 << (kMipLevels - mipLevel));
  let texelCoord = uv * mipSize;
  return Varying(pos, texelCoord, f32(mipLevel));
}

@fragment
fn fmain(vary: Varying) -> @location(0) vec4f {
  return textureLoad(tex, vec2u(vary.texelCoord), u32(vary.mipLevel));
}
struct Config {
  viewProj: mat4x4f,
  animationOffset: vec2f,
  flangeSize: f32,
  highlightFlange: f32,
};
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> matrices: array<mat4x4f>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var tex: texture_2d<f32>;

struct Varying {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

override kTextureBaseSize: f32;
override kViewportSize: f32;

@vertex
fn vmain(
  @builtin(instance_index) instance_index: u32,
  @builtin(vertex_index) vertex_index: u32,
) -> Varying {
  let flange = config.flangeSize;
  var uvs = array(
    vec2(-flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, -flange),
    vec2(1 + flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, 1 + flange),
  );
  // Default size (if matrix is the identity) makes 1 texel = 1 pixel.
  let radius = (1 + 2 * flange) * kTextureBaseSize / kViewportSize;
  var positions = array(
    vec2(-radius, -radius), vec2(-radius, radius), vec2(radius, -radius),
    vec2(radius, -radius), vec2(-radius, radius), vec2(radius, radius),
  );

  let modelMatrix = matrices[instance_index];
  let pos = config.viewProj * modelMatrix * vec4f(positions[vertex_index] + config.animationOffset, 0, 1);
  return Varying(pos, uvs[vertex_index]);
}

@fragment
fn fmain(vary: Varying) -> @location(0) vec4f {
  let uv = vary.uv;
  var color = textureSample(tex, samp, uv);

  let outOfBounds = uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1;
  if config.highlightFlange > 0 && outOfBounds {
    color += vec4(0.7, 0, 0, 0);
  }

  return color;
}


@fragment fn fs() -> @location(0) vec4f {
  return vec4f(1, 0.5, 0.2, 1);
}
struct VSOutput {
  @location(0) texcoord: vec2f,
};

@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var t: texture_2d<f32>;

@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
  let color = textureSample(t, s, vsOut.texcoord);
  if (color.a < 0.1) {
    discard;
  }
  return color;
}
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution * clipPos.w, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Uniforms {
  color0: vec4f,
  color1: vec4f,
  size: u32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex
fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
  const pos = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vertexIndex], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let grid = vec2u(position.xy) / uni.size;
  let checker = (grid.x + grid.y) % 2 == 1;
  return select(uni.color0, uni.color1, checker);
}

struct Params {
  filterDim : i32,
  blockDim : u32,
}

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var<uniform> params : Params;
@group(1) @binding(1) var inputTex : texture_2d<f32>;
@group(1) @binding(2) var outputTex : texture_storage_2d<rgba8unorm, write>;

struct Flip {
  value : u32,
}
@group(1) @binding(3) var<uniform> flip : Flip;

// This shader blurs the input texture in one direction, depending on whether
// |flip.value| is 0 or 1.
// It does so by running (128 / 4) threads per workgroup to load 128
// texels into 4 rows of shared memory. Each thread loads a
// 4 x 4 block of texels to take advantage of the texture sampling
// hardware.
// Then, each thread computes the blur result by averaging the adjacent texel values
// in shared memory.
// Because we're operating on a subset of the texture, we cannot compute all of the
// results since not all of the neighbors are available in shared memory.
// Specifically, with 128 x 128 tiles, we can only compute and write out
// square blocks of size 128 - (filterSize - 1). We compute the number of blocks
// needed in Javascript and dispatch that amount.

var<workgroup> tile : array<array<vec3f, 128>, 4>;

@compute @workgroup_size(32, 1, 1)
fn main(
  @builtin(workgroup_id) WorkGroupID : vec3u,
  @builtin(local_invocation_id) LocalInvocationID : vec3u
) {
  let filterOffset = (params.filterDim - 1) / 2;
  let dims = vec2i(textureDimensions(inputTex, 0));
  let baseIndex = vec2i(WorkGroupID.xy * vec2(params.blockDim, 4) +
                            LocalInvocationID.xy * vec2(4, 1))
                  - vec2(filterOffset, 0);

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var loadIndex = baseIndex + vec2(c, r);
      if (flip.value != 0u) {
        loadIndex = loadIndex.yx;
      }

      tile[r][4 * LocalInvocationID.x + u32(c)] = textureSampleLevel(
        inputTex,
        samp,
        (vec2f(loadIndex) + vec2f(0.25, 0.25)) / vec2f(dims),
        0.0
      ).rgb;
    }
  }

  workgroupBarrier();

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var writeIndex = baseIndex + vec2(c, r);
      if (flip.value != 0) {
        writeIndex = writeIndex.yx;
      }

      let center = i32(4 * LocalInvocationID.x) + c;
      if (center >= filterOffset &&
          center < 128 - filterOffset &&
          all(writeIndex < dims)) {
        var acc = vec3(0.0, 0.0, 0.0);
        for (var f = 0; f < params.filterDim; f++) {
          var i = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
        }
        textureStore(outputTex, writeIndex, vec4(acc, 1.0));
      }
    }
  }
}
struct ComputeUniforms {
  width: f32,
  height: f32,
  algo: u32,
  blockHeight: u32,
}

struct FragmentUniforms {
  // boolean, either 0 or 1
  highlight: u32,
}

struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) fragUV: vec2f
}

// Uniforms from compute shader
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: ComputeUniforms;
// Fragment shader uniforms
@group(1) @binding(0) var<uniform> fragment_uniforms: FragmentUniforms;

@fragment
fn frag_main(input: VertexOutput) -> @location(0) vec4f {
  var uv: vec2f = vec2f(
    input.fragUV.x * uniforms.width,
    input.fragUV.y * uniforms.height
  );

  var pixel: vec2u = vec2u(
    u32(floor(uv.x)),
    u32(floor(uv.y)),
  );
  
  var elementIndex = u32(uniforms.width) * pixel.y + pixel.x;
  var colorChanger = data[elementIndex];

  var subtracter = f32(colorChanger) / (uniforms.width * uniforms.height);

  if (fragment_uniforms.highlight == 1) {
    return select(
      //If element is above halfHeight, highlight green
      vec4f(vec3f(0.0, 1.0 - subtracter, 0.0).rgb, 1.0),
      //If element is below halfheight, highlight red
      vec4f(vec3f(1.0 - subtracter, 0.0, 0.0).rgb, 1.0),
      elementIndex % uniforms.blockHeight < uniforms.blockHeight / 2
    );
  }

  var color: vec3f = vec3f(
    1.0 - subtracter
  );

  return vec4f(color.rgb, 1.0);
}
@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn atomicToZero() {
  let counterValue = atomicLoad(&counter);
  atomicSub(&counter, counterValue);
}
struct OurVertexShaderOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

struct Uniforms {
  matrix: mat4x4f,
};

@group(0) @binding(2) var<uniform> uni: Uniforms;

@vertex fn vs(
  @builtin(vertex_index) vertexIndex : u32
) -> OurVertexShaderOutput {
  let pos = array(

    vec2f( 0.0,  0.0),  // center
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 0.0,  1.0),  // center, top

    // 2st triangle
    vec2f( 0.0,  1.0),  // center, top
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 1.0,  1.0),  // right, top
  );

  var vsOutput: OurVertexShaderOutput;
  let xy = pos[vertexIndex];
  vsOutput.position = uni.matrix * vec4f(xy, 0.0, 1.0);
  vsOutput.texcoord = xy;
  return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(fsInput: OurVertexShaderOutput) -> @location(0) vec4f {
  return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}// Whale.glb Vertex attributes
// Read in VertexInput from attributes
// f32x3    f32x3   f32x2       u8x4       f32x4
struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) normal: vec3f,
  @location(1) joints: vec4f,
  @location(2) weights: vec4f,
}

struct CameraUniforms {
  proj_matrix: mat4x4f,
  view_matrix: mat4x4f,
  model_matrix: mat4x4f,
}

struct GeneralUniforms {
  render_mode: u32,
  skin_mode: u32,
}

struct NodeUniforms {
  world_matrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
@group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
@group(2) @binding(0) var<uniform> node_uniforms: NodeUniforms;
@group(3) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
@group(3) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  // Compute joint_matrices * inverse_bind_matrices
  let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
  let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
  let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
  let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];
  // Compute influence of joint based on weight
  let skin_matrix = 
    joint0 * input.weights[0] +
    joint1 * input.weights[1] +
    joint2 * input.weights[2] +
    joint3 * input.weights[3];
  // Position of the vertex relative to our world
  let world_position = vec4f(input.position.x, input.position.y, input.position.z, 1.0);
  // Vertex position with model rotation, skinning, and the mesh's node transformation applied.
  let skinned_position = camera_uniforms.model_matrix * skin_matrix * node_uniforms.world_matrix * world_position;
  // Vertex position with only the model rotation applied.
  let rotated_position = camera_uniforms.model_matrix * world_position;
  // Determine which position to used based on whether skinMode is turnd on or off.
  let transformed_position = select(
    rotated_position,
    skinned_position,
    general_uniforms.skin_mode == 0
  );
  // Apply the camera and projection matrix transformations to our transformed position;
  output.Position = camera_uniforms.proj_matrix * camera_uniforms.view_matrix * transformed_position;
  output.normal = input.normal;
  // Convert u32 joint data to f32s to prevent flat interpolation error.
  output.joints = vec4f(f32(input.joints[0]), f32(input.joints[1]), f32(input.joints[2]), f32(input.joints[3]));
  output.weights = input.weights;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  switch general_uniforms.render_mode {
    case 1: {
      return input.joints;
    } 
    case 2: {
      return input.weights;
    }
    default: {
      return vec4f(input.normal, 1.0);
    }
  }
}struct VertexInput {
  @location(0) vert_pos: vec2f,
  @location(1) joints: vec4u,
  @location(2) weights: vec4f
}

struct VertexOutput {
  @builtin(position) Position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) joints: vec4f,
  @location(2) weights: vec4f,
}

struct CameraUniforms {
  projMatrix: mat4x4f,
  viewMatrix: mat4x4f,
  modelMatrix: mat4x4f,
}

struct GeneralUniforms {
  render_mode: u32,
  skin_mode: u32,
}

@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
@group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
@group(2) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  var bones = vec4f(0.0, 0.0, 0.0, 0.0);
  let position = vec4f(input.vert_pos.x, input.vert_pos.y, 0.0, 1.0);
  // Get relevant 4 bone matrices
  let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
  let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
  let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
  let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];
  // Compute influence of joint based on weight
  let skin_matrix = 
    joint0 * input.weights[0] +
    joint1 * input.weights[1] +
    joint2 * input.weights[2] +
    joint3 * input.weights[3];
  // Bone transformed mesh
  output.Position = select(
    camera_uniforms.projMatrix * camera_uniforms.viewMatrix * camera_uniforms.modelMatrix * position,
    camera_uniforms.projMatrix * camera_uniforms.viewMatrix * camera_uniforms.modelMatrix * skin_matrix * position,
    general_uniforms.skin_mode == 0
  );

  //Get unadjusted world coordinates
  output.world_pos = position.xyz;
  output.joints = vec4f(f32(input.joints.x), f32(input.joints.y), f32(input.joints.z), f32(input.joints.w));
  output.weights = input.weights;
  return output;
}


@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  switch general_uniforms.render_mode {
    case 1: {
      return input.joints;
    }
    case 2: {
      return input.weights;
    }
    default: {
      return vec4f(255.0, 0.0, 1.0, 1.0); 
    }
  }
}@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_2d<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV) * fragPosition;
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Particle {
  pos : vec2f,
  vel : vec2f,
}
struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
}
struct Particles {
  particles : array<Particle>,
}
@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> particlesA : Particles;
@binding(2) @group(0) var<storage, read_write> particlesB : Particles;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
  var index = GlobalInvocationID.x;

  var vPos = particlesA.particles[index].pos;
  var vVel = particlesA.particles[index].vel;
  var cMass = vec2(0.0);
  var cVel = vec2(0.0);
  var colVel = vec2(0.0);
  var cMassCount = 0u;
  var cVelCount = 0u;
  var pos : vec2f;
  var vel : vec2f;

  for (var i = 0u; i < arrayLength(&particlesA.particles); i++) {
    if (i == index) {
      continue;
    }

    pos = particlesA.particles[i].pos.xy;
    vel = particlesA.particles[i].vel.xy;
    if (distance(pos, vPos) < params.rule1Distance) {
      cMass += pos;
      cMassCount++;
    }
    if (distance(pos, vPos) < params.rule2Distance) {
      colVel -= pos - vPos;
    }
    if (distance(pos, vPos) < params.rule3Distance) {
      cVel += vel;
      cVelCount++;
    }
  }
  if (cMassCount > 0) {
    cMass = (cMass / vec2(f32(cMassCount))) - vPos;
  }
  if (cVelCount > 0) {
    cVel /= f32(cVelCount);
  }
  vVel += (cMass * params.rule1Scale) + (colVel * params.rule2Scale) + (cVel * params.rule3Scale);

  // clamp velocity for a more pleasing simulation
  vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
  // kinematic update
  vPos = vPos + (vVel * params.deltaT);
  // Wrap around boundary
  if (vPos.x < -1.0) {
    vPos.x = 1.0;
  }
  if (vPos.x > 1.0) {
    vPos.x = -1.0;
  }
  if (vPos.y < -1.0) {
    vPos.y = 1.0;
  }
  if (vPos.y > 1.0) {
    vPos.y = -1.0;
  }
  // Write back
  particlesB.particles[index].pos = vPos;
  particlesB.particles[index].vel = vVel;
}
struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(4) color : vec4f,
}

@vertex
fn vert_main(
  @location(0) a_particlePos : vec2f,
  @location(1) a_particleVel : vec2f,
  @location(2) a_pos : vec2f
) -> VertexOutput {
  let angle = -atan2(a_particleVel.x, a_particleVel.y);
  let pos = vec2(
    (a_pos.x * cos(angle)) - (a_pos.y * sin(angle)),
    (a_pos.x * sin(angle)) + (a_pos.y * cos(angle))
  );
  
  var output : VertexOutput;
  output.position = vec4(pos + a_particlePos, 0.0, 1.0);
  output.color = vec4(
    1.0 - sin(angle + 1.0) - a_particleVel.y,
    pos.x * 100.0 - a_particleVel.y + 0.1,
    a_particleVel.x + cos(angle + 0.5),
    1.0);
  return output;
}

@fragment
fn frag_main(@location(4) color : vec4f) -> @location(0) vec4f {
  return color;
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) instance: u32
};

@vertex
fn main_vs(@location(0) position: vec4f, @builtin(instance_index) instance: u32) -> VertexOutput {
  var output: VertexOutput;

  // distribute instances into a staggered 4x4 grid
  const gridWidth = 125.0;
  const cellSize = gridWidth / 4.0;
  let row = instance / 2u;
  let col = instance % 2u;

  let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u != 0u) * cellSize;
  let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

  let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

  output.position = uniforms.modelViewProjectionMatrix * offsetPos;
  output.instance = instance;
  return output;
}

@fragment
fn main_fs(@location(0) @interpolate(flat) instance: u32) -> @location(0) vec4f {
  const colors = array<vec3f,6>(
      vec3(1.0, 0.0, 0.0),
      vec3(0.0, 1.0, 0.0),
      vec3(0.0, 0.0, 1.0),
      vec3(1.0, 0.0, 1.0),
      vec3(1.0, 1.0, 0.0),
      vec3(0.0, 1.0, 1.0),
  );

  return vec4(colors[instance % 6u], 1.0);
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
  maxStorableFragments: u32,
  targetWidth: u32,
};

struct SliceInfo {
  sliceStartY: i32
};

struct Heads {
  numFragments: atomic<u32>,
  data: array<atomic<u32>>
};

struct LinkedListElement {
  color: vec4f,
  depth: f32,
  next: u32
};

struct LinkedList {
  data: array<LinkedListElement>
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(0) var<storage, read_write> heads: Heads;
@binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
@binding(3) @group(0) var opaqueDepthTexture: texture_depth_2d;
@binding(4) @group(0) var<uniform> sliceInfo: SliceInfo;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) instance: u32
};

@vertex
fn main_vs(@location(0) position: vec4f, @builtin(instance_index) instance: u32) -> VertexOutput {
  var output: VertexOutput;

  // distribute instances into a staggered 4x4 grid
  const gridWidth = 125.0;
  const cellSize = gridWidth / 4.0;
  let row = instance / 2u;
  let col = instance % 2u;

  let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u == 0u) * cellSize;
  let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

  let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

  output.position = uniforms.modelViewProjectionMatrix * offsetPos;
  output.instance = instance;

  return output;
}

@fragment
fn main_fs(@builtin(position) position: vec4f, @location(0) @interpolate(flat) instance: u32) {
  const colors = array<vec3f,6>(
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 1.0, 1.0),
  );

  let fragCoords = vec2i(position.xy);
  let opaqueDepth = textureLoad(opaqueDepthTexture, fragCoords, 0);

  // reject fragments behind opaque objects
  if position.z >= opaqueDepth {
    discard;
  }

  // The index in the heads buffer corresponding to the head data for the fragment at
  // the current location.
  let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

  // The index in the linkedList buffer at which to store the new fragment
  let fragIndex = atomicAdd(&heads.numFragments, 1u);

  // If we run out of space to store the fragments, we just lose them
  if fragIndex < uniforms.maxStorableFragments {
    let lastHead = atomicExchange(&heads.data[headsIndex], fragIndex);
    linkedList.data[fragIndex].depth = position.z;
    linkedList.data[fragIndex].next = lastHead;
    linkedList.data[fragIndex].color = vec4(colors[(instance + 3u) % 6u], 0.3);
  }
}
struct Uniforms {
  modelViewProjectionMatrix: mat4x4f,
  maxStorableFragments: u32,
  targetWidth: u32,
};

struct SliceInfo {
  sliceStartY: i32
};

struct Heads {
  numFragments: u32,
  data: array<u32>
};

struct LinkedListElement {
  color: vec4f,
  depth: f32,
  next: u32
};

struct LinkedList {
  data: array<LinkedListElement>
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(0) var<storage, read_write> heads: Heads;
@binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
@binding(3) @group(0) var<uniform> sliceInfo: SliceInfo;

// Output a full screen quad
@vertex
fn main_vs(@builtin(vertex_index) vertIndex: u32) -> @builtin(position) vec4f {
  const position = array<vec2f, 6>(
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
  );
  
  return vec4(position[vertIndex], 0.0, 1.0);
}

@fragment
fn main_fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let fragCoords = vec2i(position.xy);
  let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

  // The maximum layers we can process for any pixel
  const maxLayers = 12u;

  var layers: array<LinkedListElement, maxLayers>;

  var numLayers = 0u;
  var elementIndex = heads.data[headsIndex];

  // copy the list elements into an array up to the maximum amount of layers
  while elementIndex != 0xFFFFFFFFu && numLayers < maxLayers {
    layers[numLayers] = linkedList.data[elementIndex];
    numLayers++;
    elementIndex = linkedList.data[elementIndex].next;
  }

  if numLayers == 0u {
    discard;
  }
  
  // sort the fragments by depth
  for (var i = 1u; i < numLayers; i++) {
    let toInsert = layers[i];
    var j = i;

    while j > 0u && toInsert.depth > layers[j - 1u].depth {
      layers[j] = layers[j - 1u];
      j--;
    }

    layers[j] = toInsert;
  }

  // pre-multiply alpha for the first layer
  var color = vec4(layers[0].color.a * layers[0].color.rgb, layers[0].color.a);

  // blend the remaining layers
  for (var i = 1u; i < numLayers; i++) {
    let mixed = mix(color.rgb, layers[i].color.rgb, layers[i].color.aaa);
    color = vec4(mixed, color.a);
  }

  return color;
}
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_depth_2d;

override canvasSizeWidth: f32;
override canvasSizeHeight: f32;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  var result : vec4f;
  let c = coord.xy / vec2f(canvasSizeWidth, canvasSizeHeight);
  if (c.x < 0.33333) {
    let rawDepth = textureLoad(
      gBufferDepth,
      vec2i(floor(coord.xy)),
      0
    );
    // remap depth into something a bit more visible
    let depth = (1.0 - rawDepth) * 50.0;
    result = vec4(depth);
  } else if (c.x < 0.66667) {
    result = textureLoad(
      gBufferNormal,
      vec2i(floor(coord.xy)),
      0
    );
    result.x = (result.x + 1.0) * 0.5;
    result.y = (result.y + 1.0) * 0.5;
    result.z = (result.z + 1.0) * 0.5;
  } else {
    result = textureLoad(
      gBufferAlbedo,
      vec2i(floor(coord.xy)),
      0
    );
  }
  return result;
}
struct GBufferOutput {
  @location(0) normal : vec4f,

  // Textures: diffuse color, specular color, smoothness, emissive etc. could go here
  @location(1) albedo : vec4f,
}

@fragment
fn main(
  @location(0) fragNormal: vec3f,
  @location(1) fragUV : vec2f
) -> GBufferOutput {
  // faking some kind of checkerboard texture
  let uv = floor(30.0 * fragUV);
  let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));

  var output : GBufferOutput;
  output.normal = vec4(normalize(fragNormal), 1.0);
  output.albedo = vec4(c, c, c, 1.0);

  return output;
}
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_depth_2d;

struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}
@group(1) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;

struct Config {
  numLights : u32,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}
@group(1) @binding(1) var<uniform> config: Config;
@group(1) @binding(2) var<uniform> camera: Camera;

fn world_from_screen_coord(coord : vec2f, depth_sample: f32) -> vec3f {
  // reconstruct world-space position from the screen coordinate.
  let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
  let posWorldW = camera.invViewProjectionMatrix * posClip;
  let posWorld = posWorldW.xyz / posWorldW.www;
  return posWorld;
}

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  var result : vec3f;

  let depth = textureLoad(
    gBufferDepth,
    vec2i(floor(coord.xy)),
    0
  );

  // Don't light the sky.
  if (depth >= 1.0) {
    discard;
  }

  let bufferSize = textureDimensions(gBufferDepth);
  let coordUV = coord.xy / vec2f(bufferSize);
  let position = world_from_screen_coord(coordUV, depth);

  let normal = textureLoad(
    gBufferNormal,
    vec2i(floor(coord.xy)),
    0
  ).xyz;

  let albedo = textureLoad(
    gBufferAlbedo,
    vec2i(floor(coord.xy)),
    0
  ).rgb;

  for (var i = 0u; i < config.numLights; i++) {
    let L = lightsBuffer.lights[i].position.xyz - position;
    let distance = length(L);
    if (distance > lightsBuffer.lights[i].radius) {
      continue;
    }
    let lambert = max(dot(normal, normalize(L)), 0.0);
    result += vec3f(
      lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo
    );
  }

  // some manual ambient
  result += vec3(0.2);

  return vec4(result, 1.0);
}
struct Uniforms {
  modelMatrix : mat4x4f,
  normalModelMatrix : mat4x4f,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragNormal: vec3f,    // normal in world space
  @location(1) fragUV: vec2f,
}

@vertex
fn main(
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  let worldPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
  output.Position = camera.viewProjectionMatrix * vec4(worldPosition, 1.0);
  output.fragNormal = normalize((uniforms.normalModelMatrix * vec4(normal, 1.0)).xyz);
  output.fragUV = uv;
  return output;
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4f(pos[VertexIndex], 0.0, 1.0);
}
struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}
@group(0) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;

struct Config {
  numLights : u32,
}
@group(0) @binding(1) var<uniform> config: Config;

struct LightExtent {
  min : vec4f,
  max : vec4f,
}
@group(0) @binding(2) var<uniform> lightExtent: LightExtent;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
  var index = GlobalInvocationID.x;
  if (index >= config.numLights) {
    return;
  }

  lightsBuffer.lights[index].position.y = lightsBuffer.lights[index].position.y - 0.5 - 0.003 * (f32(index) - 64.0 * floor(f32(index) / 64.0));

  if (lightsBuffer.lights[index].position.y < lightExtent.min.y) {
    lightsBuffer.lights[index].position.y = lightExtent.max.y;
  }
}
// Positions for simple quad geometry
const pos = array(vec2f(0, -1), vec2f(1, -1), vec2f(0, 0), vec2f(1, 0));

struct VertexInput {
  @builtin(vertex_index) vertex : u32,
  @builtin(instance_index) instance : u32,
};

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) texcoord : vec2f,
};

struct Char {
  texOffset: vec2f,
  texExtent: vec2f,
  size: vec2f,
  offset: vec2f,
};

struct FormattedText {
  transform: mat4x4f,
  color: vec4f,
  scale: f32,
  chars: array<vec3f>,
};

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
};

// Font bindings
@group(0) @binding(0) var fontTexture: texture_2d<f32>;
@group(0) @binding(1) var fontSampler: sampler;
@group(0) @binding(2) var<storage> chars: array<Char>;

// Text bindings
@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var<storage> text: FormattedText;

@vertex
fn vertexMain(input : VertexInput) -> VertexOutput {
  let textElement = text.chars[input.instance];
  let char = chars[u32(textElement.z)];
  let charPos = (pos[input.vertex] * char.size + textElement.xy + char.offset) * text.scale;

  var output : VertexOutput;
  output.position = camera.projection * camera.view * text.transform * vec4f(charPos, 0, 1);

  output.texcoord = pos[input.vertex] * vec2f(1, -1);
  output.texcoord *= char.texExtent;
  output.texcoord += char.texOffset;
  return output;
}

fn sampleMsdf(texcoord: vec2f) -> f32 {
  let c = textureSample(fontTexture, fontSampler, texcoord);
  return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

// Antialiasing technique from Paul Houx 
// https://github.com/Chlumsky/msdfgen/issues/22#issuecomment-234958005
@fragment
fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {
  // pxRange (AKA distanceRange) comes from the msdfgen tool. Don McCurdy's tool
  // uses the default which is 4.
  let pxRange = 4.0;
  let sz = vec2f(textureDimensions(fontTexture, 0));
  let dx = sz.x*length(vec2f(dpdxFine(input.texcoord.x), dpdyFine(input.texcoord.x)));
  let dy = sz.y*length(vec2f(dpdxFine(input.texcoord.y), dpdyFine(input.texcoord.y)));
  let toPixels = pxRange * inverseSqrt(dx * dx + dy * dy);
  let sigDist = sampleMsdf(input.texcoord) - 0.5;
  let pxDist = sigDist * toPixels;

  let edgeWidth = 0.5;

  let alpha = smoothstep(-edgeWidth, edgeWidth, pxDist);

  if (alpha < 0.001) {
    discard;
  }

  return vec4f(text.color.rgb, text.color.a * alpha);
}////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////
var<private> rand_seed : vec2f;

fn init_rand(invocation_id : u32, seed : vec4f) {
  rand_seed = seed.xz;
  rand_seed = fract(rand_seed * cos(35.456+f32(invocation_id) * seed.yw));
  rand_seed = fract(rand_seed * cos(41.235+f32(invocation_id) * seed.xw));
}

fn rand() -> f32 {
  rand_seed.x = fract(cos(dot(rand_seed, vec2f(23.14077926, 232.61690225))) * 136.8168);
  rand_seed.y = fract(cos(dot(rand_seed, vec2f(54.47856553, 345.84153136))) * 534.7645);
  return rand_seed.y;
}

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
struct RenderParams {
  modelViewProjectionMatrix : mat4x4f,
  right : vec3f,
  up : vec3f
}
@binding(0) @group(0) var<uniform> render_params : RenderParams;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) color : vec4f,
  @location(2) quad_pos : vec2f, // -1..+1
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) quad_pos : vec2f, // -1..+1
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  var quad_pos = mat2x3f(render_params.right, render_params.up) * in.quad_pos;
  var position = in.position + quad_pos * 0.01;
  var out : VertexOutput;
  out.position = render_params.modelViewProjectionMatrix * vec4f(position, 1.0);
  out.color = in.color;
  out.quad_pos = in.quad_pos;
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  var color = in.color;
  // Apply a circular particle alpha mask
  color.a = color.a * max(1.0 - length(in.quad_pos), 0.0);
  return color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
struct SimulationParams {
  deltaTime : f32,
  brightnessFactor : f32,
  seed : vec4f,
}

struct Particle {
  position : vec3f,
  lifetime : f32,
  color    : vec4f,
  velocity : vec3f,
}

struct Particles {
  particles : array<Particle>,
}

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> data : Particles;
@binding(2) @group(0) var texture : texture_2d<f32>;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) global_invocation_id : vec3u) {
  let idx = global_invocation_id.x;

  init_rand(idx, sim_params.seed);

  var particle = data.particles[idx];

  // Apply gravity
  particle.velocity.z = particle.velocity.z - sim_params.deltaTime * 0.5;

  // Basic velocity integration
  particle.position = particle.position + sim_params.deltaTime * particle.velocity;

  // Age each particle. Fade out before vanishing.
  particle.lifetime = particle.lifetime - sim_params.deltaTime;
  particle.color.a = smoothstep(0.0, 0.5, particle.lifetime);

  // If the lifetime has gone negative, then the particle is dead and should be
  // respawned.
  if (particle.lifetime < 0.0) {
    // Use the probability map to find where the particle should be spawned.
    // Starting with the 1x1 mip level.
    var coord : vec2i;
    for (var level = u32(textureNumLevels(texture) - 1); level > 0; level--) {
      // Load the probability value from the mip-level
      // Generate a random number and using the probabilty values, pick the
      // next texel in the next largest mip level:
      //
      // 0.0    probabilites.r    probabilites.g    probabilites.b   1.0
      //  |              |              |              |              |
      //  |   TOP-LEFT   |  TOP-RIGHT   | BOTTOM-LEFT  | BOTTOM_RIGHT |
      //
      let probabilites = textureLoad(texture, coord, level);
      let value = vec4f(rand());
      let mask = (value >= vec4f(0.0, probabilites.xyz)) & (value < probabilites);
      coord = coord * 2;
      coord.x = coord.x + select(0, 1, any(mask.yw)); // x  y
      coord.y = coord.y + select(0, 1, any(mask.zw)); // z  w
    }
    let uv = vec2f(coord) / vec2f(textureDimensions(texture));
    particle.position = vec3f((uv - 0.5) * 3.0 * vec2f(1.0, -1.0), 0.0);
    particle.color = textureLoad(texture, coord, 0);
    particle.color.r *= sim_params.brightnessFactor;
    particle.color.g *= sim_params.brightnessFactor;
    particle.color.b *= sim_params.brightnessFactor;
    particle.velocity.x = (rand() - 0.5) * 0.1;
    particle.velocity.y = (rand() - 0.5) * 0.1;
    particle.velocity.z = rand() * 0.3;
    particle.lifetime = 0.5 + rand() * 3.0;
  }

  // Store the new particle value
  data.particles[idx] = particle;
}
struct UBO {
  width : u32,
}

@binding(0) @group(0) var<uniform> ubo : UBO;
@binding(1) @group(0) var<storage, read> buf_in : array<f32>;
@binding(2) @group(0) var<storage, read_write> buf_out : array<f32>;
@binding(3) @group(0) var tex_in : texture_2d<f32>;
@binding(3) @group(0) var tex_out : texture_storage_2d<rgba8unorm, write>;

////////////////////////////////////////////////////////////////////////////////
// import_level
//
// Loads the alpha channel from a texel of the source image, and writes it to
// the buf_out.weights.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn import_level(@builtin(global_invocation_id) coord : vec3u) {
  _ = &buf_in; // so the bindGroups are similar.
  if (!all(coord.xy < vec2u(textureDimensions(tex_in)))) {
    return;
  }

  let offset = coord.x + coord.y * ubo.width;
  buf_out[offset] = textureLoad(tex_in, vec2i(coord.xy), 0).w;
}

////////////////////////////////////////////////////////////////////////////////
// export_level
//
// Loads 4 f32 weight values from buf_in.weights, and stores summed value into
// buf_out.weights, along with the calculated 'probabilty' vec4 values into the
// mip level of tex_out. See simulate() in particle.wgsl to understand the
// probability logic.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn export_level(@builtin(global_invocation_id) coord : vec3u) {
  if (!all(coord.xy < vec2u(textureDimensions(tex_out)))) {
    return;
  }

  let dst_offset = coord.x    + coord.y    * ubo.width;
  let src_offset = coord.x*2u + coord.y*2u * ubo.width;

  let a = buf_in[src_offset + 0u];
  let b = buf_in[src_offset + 1u];
  let c = buf_in[src_offset + 0u + ubo.width];
  let d = buf_in[src_offset + 1u + ubo.width];
  let sum = a + b + c + d;

  buf_out[dst_offset] = sum / 4.0;

  let probabilities = vec4f(a, a+b, a+b+c, sum) / max(sum, 0.0001);
  textureStore(tex_out, vec2i(coord.xy), probabilities);
}
const modeAlbedoTexture = 0;
const modeNormalTexture = 1;
const modeDepthTexture = 2;
const modeNormalMap = 3;
const modeParallaxScale = 4;
const modeSteepParallax = 5;

struct SpaceTransforms {
  worldViewProjMatrix: mat4x4f,
  worldViewMatrix: mat4x4f,
}

struct MapInfo {
  lightPosVS: vec3f, // Light position in view space
  mode: u32,
  lightIntensity: f32,
  depthScale: f32,
  depthLayers: f32,
}

struct VertexInput {
  // Shader assumes the missing 4th float is 1.0
  @location(0) position : vec4f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f,
  @location(3) vert_tan: vec3f,
  @location(4) vert_bitan: vec3f,
}

struct VertexOutput {
  @builtin(position) posCS : vec4f,    // vertex position in clip space
  @location(0) posVS : vec3f,          // vertex position in view space
  @location(1) tangentVS: vec3f,       // vertex tangent in view space
  @location(2) bitangentVS: vec3f,     // vertex tangent in view space
  @location(3) normalVS: vec3f,        // vertex normal in view space
  @location(5) uv : vec2f,             // vertex texture coordinate
}

// Uniforms
@group(0) @binding(0) var<uniform> spaceTransform : SpaceTransforms;
@group(0) @binding(1) var<uniform> mapInfo: MapInfo;

// Texture info
@group(1) @binding(0) var textureSampler: sampler;
@group(1) @binding(1) var albedoTexture: texture_2d<f32>;
@group(1) @binding(2) var normalTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_2d<f32>;


@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output : VertexOutput;

  output.posCS = spaceTransform.worldViewProjMatrix * input.position;
  output.posVS = (spaceTransform.worldViewMatrix * input.position).xyz;
  output.tangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_tan, 0)).xyz;
  output.bitangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_bitan, 0)).xyz;
  output.normalVS = (spaceTransform.worldViewMatrix * vec4(input.normal, 0)).xyz;
  output.uv = input.uv;

  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Build the matrix to convert from tangent space to view space
  let tangentToView = mat3x3f(
    input.tangentVS,
    input.bitangentVS,
    input.normalVS,
  );

  // The inverse of a non-scaling affine 3x3 matrix is it's transpose
  let viewToTangent = transpose(tangentToView);

  // Calculate the normalized vector in tangent space from the camera to the fragment
  let viewDirTS = normalize(viewToTangent * input.posVS);

  // Apply parallax to the texture coordinate, if parallax is enabled
  var uv : vec2f;
  switch (mapInfo.mode) {
    case modeParallaxScale: {
      uv = parallaxScale(input.uv, viewDirTS);
      break;
    }
    case modeSteepParallax: {
      uv = parallaxSteep(input.uv, viewDirTS);
      break;
    }
    default: {
      uv = input.uv;
      break;
    }
  }

  // Sample the albedo texture
  let albedoSample = textureSample(albedoTexture, textureSampler, uv);

  // Sample the normal texture
  let normalSample = textureSample(normalTexture, textureSampler, uv);

  switch (mapInfo.mode) {
    case modeAlbedoTexture: { // Output the albedo sample
      return albedoSample;
    }
    case modeNormalTexture: { // Output the normal sample
      return normalSample;
    }
    case modeDepthTexture: { // Output the depth map
      return textureSample(depthTexture, textureSampler, input.uv);
    }
    default: {
      // Transform the normal sample to a tangent space normal
      let normalTS = normalSample.xyz * 2 - 1;

      // Convert normal from tangent space to view space, and normalize
      let normalVS = normalize(tangentToView * normalTS);

      // Calculate the vector in view space from the light position to the fragment
      let fragToLightVS = mapInfo.lightPosVS - input.posVS;

      // Calculate the square distance from the light to the fragment
      let lightSqrDist = dot(fragToLightVS, fragToLightVS);

      // Calculate the normalized vector in view space from the fragment to the light
      let lightDirVS = fragToLightVS * inverseSqrt(lightSqrDist);

      // Light strength is inversely proportional to square of distance from light
      let diffuseLight = mapInfo.lightIntensity * max(dot(lightDirVS, normalVS), 0) / lightSqrDist;

      // The diffuse is the albedo color multiplied by the diffuseLight
      let diffuse = albedoSample.rgb * diffuseLight;

      return vec4f(diffuse, 1.0);
    }
  }
}


// Returns the uv coordinate displaced in the view direction by a magnitude calculated by the depth
// sampled from the depthTexture and the angle between the surface normal and view direction.
fn parallaxScale(uv: vec2f, viewDirTS: vec3f) -> vec2f {
  let depthSample = textureSample(depthTexture, textureSampler, uv).r;
  return uv + viewDirTS.xy * (depthSample * mapInfo.depthScale) / -viewDirTS.z;
}

// Returns the uv coordinates displaced in the view direction by ray-tracing the depth map.
fn parallaxSteep(startUV: vec2f, viewDirTS: vec3f) -> vec2f {
  // Calculate derivatives of the texture coordinate, so we can sample the texture with non-uniform
  // control flow.
  let ddx = dpdx(startUV);
  let ddy = dpdy(startUV);

  // Calculate the delta step in UV and depth per iteration
  let uvDelta = viewDirTS.xy * mapInfo.depthScale / (-viewDirTS.z * mapInfo.depthLayers);
  let depthDelta = 1.0 / f32(mapInfo.depthLayers);
  let posDelta = vec3(uvDelta, depthDelta);

  // Walk the depth texture, and stop when the ray intersects the depth map
  var pos = vec3(startUV, 0);
  for (var i = 0; i < 32; i++) {
    if (pos.z >= textureSampleGrad(depthTexture, textureSampler, pos.xy, ddx, ddy).r) {
      break; // Hit the surface
    }
    pos += posDelta;
  }

  return pos.xy;
}
@fragment
fn main(@location(0) cell: f32) -> @location(0) vec4f {
  return vec4f(cell, cell, cell, 1.);
}
@binding(0) @group(0) var<storage, read> size: vec2u;
@binding(1) @group(0) var<storage, read> current: array<u32>;
@binding(2) @group(0) var<storage, read_write> next: array<u32>;

override blockSize = 8;

fn getIndex(x: u32, y: u32) -> u32 {
  let h = size.y;
  let w = size.x;

  return (y % h) * w + (x % w);
}

fn getCell(x: u32, y: u32) -> u32 {
  return current[getIndex(x, y)];
}

fn countNeighbors(x: u32, y: u32) -> u32 {
  return getCell(x - 1, y - 1) + getCell(x, y - 1) + getCell(x + 1, y - 1) + 
         getCell(x - 1, y) +                         getCell(x + 1, y) + 
         getCell(x - 1, y + 1) + getCell(x, y + 1) + getCell(x + 1, y + 1);
}

@compute @workgroup_size(blockSize, blockSize)
fn main(@builtin(global_invocation_id) grid: vec3u) {
  let x = grid.x;
  let y = grid.y;
  let n = countNeighbors(x, y);
  next[getIndex(x, y)] = select(u32(n == 3u), u32(n == 2u || n == 3u), getCell(x, y) == 1u); 
} 
struct Out {
  @builtin(position) pos: vec4f,
  @location(0) cell: f32,
}

@binding(0) @group(0) var<uniform> size: vec2u;

@vertex
fn main(@builtin(instance_index) i: u32, @location(0) cell: u32, @location(1) pos: vec2u) -> Out {
  let w = size.x;
  let h = size.y;
  let x = (f32(i % w + pos.x) / f32(w) - 0.5) * 2. * f32(w) / f32(max(w, h));
  let y = (f32((i - (i % w)) / w + pos.y) / f32(h) - 0.5) * 2. * f32(h) / f32(max(w, h));

  return Out(vec4f(x, y, 0., 1.), f32(cell));
}
// The lightmap data
@group(1) @binding(0) var lightmap : texture_2d_array<f32>;

// The sampler used to sample the lightmap
@group(1) @binding(1) var smpl : sampler;

// The output framebuffer
@group(1) @binding(2) var framebuffer : texture_storage_2d<rgba16float, write>;

override WorkgroupSizeX : u32;
override WorkgroupSizeY : u32;

const NumReflectionRays = 5;

@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
  if (all(invocation_id.xy < textureDimensions(framebuffer))) {
    init_rand(invocation_id);

    // Calculate the fragment's NDC coordinates for the intersection of the near
    // clip plane and far clip plane
    let uv = vec2f(invocation_id.xy) / vec2f(textureDimensions(framebuffer).xy);
    let ndcXY = (uv - 0.5) * vec2(2, -2);

    // Transform the coordinates back into world space
    var near = common_uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
    var far = common_uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
    near /= near.w;
    far /= far.w;

    // Create a ray that starts at the near clip plane, heading in the fragment's
    // z-direction, and raytrace to find the nearest quad that the ray intersects.
    let ray = Ray(near.xyz, normalize(far.xyz - near.xyz));
    let hit = raytrace(ray);

    let hit_color = sample_hit(hit);
    var normal = quads[hit.quad].plane.xyz;

    // Fire a few rays off the surface to collect some reflections
    let bounce = reflect(ray.dir, normal);
    var reflection : vec3f;
    for (var i = 0; i < NumReflectionRays; i++) {
      let reflection_dir = normalize(bounce + rand_unit_sphere()*0.1);
      let reflection_ray = Ray(hit.pos + bounce * 1e-5, reflection_dir);
      let reflection_hit = raytrace(reflection_ray);
      reflection += sample_hit(reflection_hit);
    }
    let color = mix(reflection / NumReflectionRays, hit_color, 0.95);

    textureStore(framebuffer, invocation_id.xy, vec4(color, 1));
  }
}


// Returns the sampled hit quad's lightmap at 'hit.uv', and adds the quad's
// emissive value.
fn sample_hit(hit : HitInfo) -> vec3f {
  let quad = quads[hit.quad];
  // Sample the quad's lightmap, and add emissive.
  return textureSampleLevel(lightmap, smpl, hit.uv, hit.quad, 0).rgb +
         quad.emissive * quad.color;
}
const pi = 3.14159265359;

// Quad describes 2D rectangle on a plane
struct Quad {
  // The surface plane
  plane    : vec4f,
  // A plane with a normal in the 'u' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'right' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  right    : vec4f,
  // A plane with a normal in the 'v' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'up' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  up       : vec4f,
  // The diffuse color of the quad
  color    : vec3f,
  // Emissive value. 0=no emissive, 1=full emissive.
  emissive : f32,
};

// Ray is a start point and direction.
struct Ray {
  start : vec3f,
  dir   : vec3f,
}

// Value for HitInfo.quad if no intersection occured.
const kNoHit = 0xffffffff;

// HitInfo describes the hit location of a ray-quad intersection
struct HitInfo {
  // Distance along the ray to the intersection
  dist : f32,
  // The quad index that was hit
  quad : u32,
  // The position of the intersection
  pos : vec3f,
  // The UVs of the quad at the point of intersection
  uv : vec2f,
}

// CommonUniforms uniform buffer data
struct CommonUniforms {
  // Model View Projection matrix
  mvp : mat4x4f,
  // Inverse of mvp
  inv_mvp : mat4x4f,
  // Random seed for the workgroup
  seed : vec3u,
}

// The common uniform buffer binding.
@group(0) @binding(0) var<uniform> common_uniforms : CommonUniforms;

// The quad buffer binding.
@group(0) @binding(1) var<storage> quads : array<Quad>;

// intersect_ray_quad will check to see if the ray 'r' intersects the quad 'q'.
// If an intersection occurs, and the intersection is closer than 'closest' then
// the intersection information is returned, otherwise 'closest' is returned.
fn intersect_ray_quad(r : Ray, quad : u32, closest : HitInfo) -> HitInfo {
  let q = quads[quad];
  let plane_dist = dot(q.plane, vec4(r.start, 1));
  let ray_dist = plane_dist / -dot(q.plane.xyz, r.dir);
  let pos = r.start + r.dir * ray_dist;
  let uv = vec2(dot(vec4f(pos, 1), q.right),
                dot(vec4f(pos, 1), q.up)) * 0.5 + 0.5;
  let hit = plane_dist > 0 &&
            ray_dist > 0 &&
            ray_dist < closest.dist &&
            all((uv > vec2f()) & (uv < vec2f(1)));
  return HitInfo(
    select(closest.dist, ray_dist, hit),
    select(closest.quad, quad,     hit),
    select(closest.pos,  pos,      hit),
    select(closest.uv,   uv,       hit),
  );
}

// raytrace finds the closest intersecting quad for the given ray
fn raytrace(ray : Ray) -> HitInfo {
  var hit = HitInfo();
  hit.dist = 1e20;
  hit.quad = kNoHit;
  for (var quad = 0u; quad < arrayLength(&quads); quad++) {
    hit = intersect_ray_quad(ray, quad, hit);
  }
  return hit;
}

// A psuedo random number. Initialized with init_rand(), updated with rand().
var<private> rnd : vec3u;

// Initializes the random number generator.
fn init_rand(invocation_id : vec3u) {
  const A = vec3(1741651 * 1009,
                 140893  * 1609 * 13,
                 6521    * 983  * 7 * 2);
  rnd = (invocation_id * A) ^ common_uniforms.seed;
}

// Returns a random number between 0 and 1.
fn rand() -> f32 {
  const C = vec3(60493  * 9377,
                 11279  * 2539 * 23,
                 7919   * 631  * 5 * 3);

  rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
  return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
}

// Returns a random point within a unit sphere centered at (0,0,0).
fn rand_unit_sphere() -> vec3f {
    var u = rand();
    var v = rand();
    var theta = u * 2.0 * pi;
    var phi = acos(2.0 * v - 1.0);
    var r = pow(rand(), 1.0/3.0);
    var sin_theta = sin(theta);
    var cos_theta = cos(theta);
    var sin_phi = sin(phi);
    var cos_phi = cos(phi);
    var x = r * sin_phi * sin_theta;
    var y = r * sin_phi * cos_theta;
    var z = r * cos_phi;
    return vec3f(x, y, z);
}

fn rand_concentric_disk() -> vec2f {
    let u = vec2f(rand(), rand());
    let uOffset = 2.f * u - vec2f(1, 1);

    if (uOffset.x == 0 && uOffset.y == 0){
        return vec2f(0, 0);
    }

    var theta = 0.0;
    var r = 0.0;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (pi / 4) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (pi / 2) - (pi / 4) * (uOffset.x / uOffset.y);
    }
    return r * vec2f(cos(theta), sin(theta));
}

fn rand_cosine_weighted_hemisphere() -> vec3f {
    let d = rand_concentric_disk();
    let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return vec3f(d.x, d.y, z);
}
// The lightmap data
@group(1) @binding(0) var lightmap : texture_2d_array<f32>;

// The sampler used to sample the lightmap
@group(1) @binding(1) var smpl : sampler;

// Vertex shader input data
struct VertexIn {
  @location(0) position : vec4f,
  @location(1) uv : vec3f,
  @location(2) emissive : vec3f,
}

// Vertex shader output data
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) emissive : vec3f,
  @interpolate(flat)
  @location(2) quad : u32,
}

@vertex
fn vs_main(input : VertexIn) -> VertexOut {
  var output : VertexOut;
  output.position = common_uniforms.mvp * input.position;
  output.uv = input.uv.xy;
  output.quad = u32(input.uv.z + 0.5);
  output.emissive = input.emissive;
  return output;
}

@fragment
fn fs_main(vertex_out : VertexOut) -> @location(0) vec4f {
  return textureSample(lightmap, smpl, vertex_out.uv, vertex_out.quad) + vec4f(vertex_out.emissive, 1);
}
// A storage buffer holding an array of atomic<u32>.
// The array elements are a sequence of red, green, blue components, for each
// lightmap texel, for each quad surface.
@group(1) @binding(0)
var<storage, read_write> accumulation : array<atomic<u32>>;

// The output lightmap texture.
@group(1) @binding(1)
var lightmap : texture_storage_2d_array<rgba16float, write>;

// Uniform data used by the accumulation_to_lightmap entry point
struct Uniforms {
  // Scalar for converting accumulation values to output lightmap values
  accumulation_to_lightmap_scale : f32,
  // Accumulation buffer rescaling value
  accumulation_buffer_scale : f32,
  // The width of the light
  light_width : f32,
  // The height of the light
  light_height : f32,
  // The center of the light
  light_center : vec3f,
}

// accumulation_to_lightmap uniforms binding point
@group(1) @binding(2) var<uniform> uniforms : Uniforms;

// Number of photons emitted per workgroup
override PhotonsPerWorkgroup : u32;

// Maximum value that can be added to the accumulation buffer from a single photon
override PhotonEnergy : f32;

// Number of bounces of each photon
const PhotonBounces = 4;

// Amount of light absorbed with each photon bounce (0: 0%, 1: 100%)
const LightAbsorbtion = 0.5;

// Radiosity compute shader.
// Each invocation creates a photon from the light source, and accumulates
// bounce lighting into the 'accumulation' buffer.
@compute @workgroup_size(PhotonsPerWorkgroup)
fn radiosity(@builtin(global_invocation_id) invocation_id : vec3u) {
  init_rand(invocation_id);
  photon();
}

// Spawns a photon at the light source, performs ray tracing in the scene,
// accumulating light values into 'accumulation' for each quad surface hit.
fn photon() {
  // Create a random ray from the light.
  var ray = new_light_ray();
  // Give the photon an initial energy value.
  var color = PhotonEnergy * vec3f(1, 0.8, 0.6);

  // Start bouncing.
  for (var i = 0; i < (PhotonBounces+1); i++) {
    // Find the closest hit of the ray with the scene's quads.
    let hit = raytrace(ray);
    let quad = quads[hit.quad];

    // Bounce the ray.
    ray.start = hit.pos + quad.plane.xyz * 1e-5;
    ray.dir = normalize(reflect(ray.dir, quad.plane.xyz) + rand_unit_sphere() * 0.75);

    // Photon color is multiplied by the quad's color.
    color *= quad.color;

    // Accumulate the aborbed light into the 'accumulation' buffer.
    accumulate(hit.uv, hit.quad, color * LightAbsorbtion);

    // What wasn't absorbed is reflected.
    color *= 1 - LightAbsorbtion;
  }
}

// Performs an atomicAdd() with 'color' into the 'accumulation' buffer at 'uv'
// and 'quad'.
fn accumulate(uv : vec2f, quad : u32, color : vec3f) {
  let dims = textureDimensions(lightmap);
  let base_idx = accumulation_base_index(vec2u(uv * vec2f(dims)), quad);
  atomicAdd(&accumulation[base_idx + 0], u32(color.r + 0.5));
  atomicAdd(&accumulation[base_idx + 1], u32(color.g + 0.5));
  atomicAdd(&accumulation[base_idx + 2], u32(color.b + 0.5));
}

// Returns the base element index for the texel at 'coord' for 'quad'
fn accumulation_base_index(coord : vec2u, quad : u32) -> u32 {
  let dims = textureDimensions(lightmap);
  let c = min(vec2u(dims) - 1, coord);
  return 3 * (c.x + dims.x * c.y + dims.x * dims.y * quad);
}

// Returns a new Ray at a random point on the light, in a random downwards
// direction.
fn new_light_ray() -> Ray {
  let center = uniforms.light_center;
  let pos = center + vec3f(uniforms.light_width * (rand() - 0.5),
                           0,
                           uniforms.light_height * (rand() - 0.5));
  var dir = rand_cosine_weighted_hemisphere().xzy;
  dir.y = -dir.y;
  return Ray(pos, dir);
}

override AccumulationToLightmapWorkgroupSizeX : u32;
override AccumulationToLightmapWorkgroupSizeY : u32;

// Compute shader used to copy the atomic<u32> data in 'accumulation' to
// 'lightmap'. 'accumulation' might also be scaled to reduce integer overflow.
@compute @workgroup_size(AccumulationToLightmapWorkgroupSizeX, AccumulationToLightmapWorkgroupSizeY)
fn accumulation_to_lightmap(@builtin(global_invocation_id) invocation_id : vec3u,
                            @builtin(workgroup_id)         workgroup_id  : vec3u) {
  let dims = textureDimensions(lightmap);
  let quad = workgroup_id.z; // The workgroup 'z' value holds the quad index.
  let coord = invocation_id.xy;
  if (all(coord < dims)) {
    // Load the color value out of 'accumulation'
    let base_idx = accumulation_base_index(coord, quad);
    let color = vec3(f32(atomicLoad(&accumulation[base_idx + 0])),
                     f32(atomicLoad(&accumulation[base_idx + 1])),
                     f32(atomicLoad(&accumulation[base_idx + 2])));

    // Multiply the color by 'uniforms.accumulation_to_lightmap_scale' and write it to
    // the lightmap.
    textureStore(lightmap, coord, quad, vec4(color * uniforms.accumulation_to_lightmap_scale, 1));

    // If the 'accumulation' buffer is nearing saturation, then
    // 'uniforms.accumulation_buffer_scale' will be less than 1, scaling the values
    // to something less likely to overflow the u32.
    if (uniforms.accumulation_buffer_scale != 1.0) {
      let scaled = color * uniforms.accumulation_buffer_scale + 0.5;
      atomicStore(&accumulation[base_idx + 0], u32(scaled.r));
      atomicStore(&accumulation[base_idx + 1], u32(scaled.g));
      atomicStore(&accumulation[base_idx + 2], u32(scaled.b));
    }
  }
}
// The linear-light input framebuffer
@group(0) @binding(0) var input  : texture_2d<f32>;

// The tonemapped, gamma-corrected output framebuffer
@group(0) @binding(1) var output : texture_storage_2d<{OUTPUT_FORMAT}, write>;

const TonemapExposure = 0.5;

const Gamma = 2.2;

override WorkgroupSizeX : u32;
override WorkgroupSizeY : u32;

@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
  let color = textureLoad(input, invocation_id.xy, 0).rgb;
  let tonemapped = reinhard_tonemap(color);
  textureStore(output, invocation_id.xy, vec4f(tonemapped, 1));
}

fn reinhard_tonemap(linearColor: vec3f) -> vec3f {
  let color = linearColor * TonemapExposure;
  let mapped = color / (1+color);
  return pow(mapped, vec3f(1 / Gamma));
}
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_cube<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  // Our camera and the skybox cube are both centered at (0, 0, 0)
  // so we can use the cube geometry position to get viewing vector to sample
  // the cube texture. The magnitude of the vector doesn't matter.
  var cubemapVec = fragPosition.xyz - vec3(0.5);
  // When viewed from the inside, cubemaps are left-handed (z away from viewer),
  // but common camera matrix convention results in a right-handed world space
  // (z toward viewer), so we have to flip it.
  cubemapVec.z *= -1;
  return textureSample(myTexture, mySampler, cubemapVec);
}
@binding(1) @group(0) var mySampler: sampler;
@binding(2) @group(0) var myTexture: texture_2d<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  let texColor = textureSample(myTexture, mySampler, fragUV * 0.8 + vec2(0.1));
  let f = select(1.0, 0.0, length(texColor.rgb - vec3(0.5)) < 0.01);
  return f * texColor + (1.0 - f) * fragPosition;
}
struct Uniforms {
  inverseModelViewProjectionMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_3d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) near : vec3f,
  @location(1) step : vec3f,
}

const NumSteps = 64u;

@vertex
fn vertex_main(
  @builtin(vertex_index) VertexIndex : u32
) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2(-1.0, 3.0),
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0)
  );
  var xy = pos[VertexIndex];
  var near = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 0.0, 1);
  var far = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 1, 1);
  near /= near.w;
  far /= far.w;
  return VertexOutput(
    vec4f(xy, 0.0, 1.0),
    near.xyz,
    (far.xyz - near.xyz) / f32(NumSteps)
  );
}

@fragment
fn fragment_main(
  @location(0) near: vec3f,
  @location(1) step: vec3f
) -> @location(0) vec4f {
  var rayPos = near;
  var result = 0.0;
  for (var i = 0u; i < NumSteps; i++) {
    let texCoord = (rayPos.xyz + 1.0) * 0.5;
    let sample =
      textureSample(myTexture, mySampler, texCoord).r * 4.0 / f32(NumSteps);
    let intersects =
      all(rayPos.xyz < vec3f(1.0)) && all(rayPos.xyz > vec3f(-1.0));
    result += select(0.0, (1.0 - result) * sample, intersects && result < 1.0);
    rayPos += step;
  }
  return vec4f(vec3f(result), 1.0);
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct LineUniforms {
  stride: u32,
  thickness: f32,
  alphaThreshold: f32,
};

struct VSOut {
  @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<uniform> line: LineUniforms;

@vertex fn vsIndexedU32(@builtin(vertex_index) vNdx: u32) -> VSOut {
  // indices make a triangle so for every 3 indices we need to output
  // 6 values
  let triNdx = vNdx / 6;
  // 0 1 0 1 0 1  0 1 0 1 0 1  vNdx % 2
  // 0 0 1 1 2 2  3 3 4 4 5 5  vNdx / 2
  // 0 1 1 2 2 3  3 4 4 5 5 6  vNdx % 2 + vNdx / 2
  // 0 1 1 2 2 0  0 1 1 2 2 0  (vNdx % 2 + vNdx / 2) % 3
  let vertNdx = (vNdx % 2 + vNdx / 2) % 3;
  let index = indices[triNdx * 3 + vertNdx];

  // note:
  //
  // * if your indices are U16 you could use this
  //
  //    let indexNdx = triNdx * 3 + vertNdx;
  //    let twoIndices = indices[indexNdx / 2];  // indices is u32 but we want u16
  //    let index = (twoIndices >> ((indexNdx & 1) * 16)) & 0xFFFF;
  //
  // * if you're not using indices you could use this
  //
  //    let index = triNdx * 3 + vertNdx;

  let pNdx = index * line.stride;
  let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * position;
  return vOut;
}

@fragment fn fs() -> @location(0) vec4f {
  return uni.color + vec4f(0.5);
}

struct BarycentricCoordinateBasedVSOutput {
  @builtin(position) position: vec4f,
  @location(0) barycenticCoord: vec3f,
};

@vertex fn vsIndexedU32BarycentricCoordinateBasedLines(
  @builtin(vertex_index) vNdx: u32
) -> BarycentricCoordinateBasedVSOutput {
  let vertNdx = vNdx % 3;
  let index = indices[vNdx];

  // note:
  //
  // * if your indices are U16 you could use this
  //
  //    let twoIndices = indices[vNdx / 2];  // indices is u32 but we want u16
  //    let index = (twoIndices >> ((vNdx & 1) * 16)) & 0xFFFF;
  //
  // * if you're not using indices you could use this
  //
  //    let index = vNdx;

  let pNdx = index * line.stride;
  let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

  var vsOut: BarycentricCoordinateBasedVSOutput;
  vsOut.position = uni.worldViewProjectionMatrix * position;

  // emit a barycentric coordinate
  vsOut.barycenticCoord = vec3f(0);
  vsOut.barycenticCoord[vertNdx] = 1.0;
  return vsOut;
}

fn edgeFactor(bary: vec3f) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * line.thickness, bary);
  return min(min(a3.x, a3.y), a3.z);
}

@fragment fn fsBarycentricCoordinateBasedLines(
  v: BarycentricCoordinateBasedVSOutput
) -> @location(0) vec4f {
  let a = 1.0 - edgeFactor(v.barycenticCoord);
  if (a < line.alphaThreshold) {
    discard;
  }

  return vec4((uni.color.rgb + 0.5) * a, a);
}
struct Uniforms {
  worldViewProjectionMatrix: mat4x4f,
  worldMatrix: mat4x4f,
  color: vec4f,
};

struct Vertex {
  @location(0) position: vec4f,
  @location(1) normal: vec3f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(vin: Vertex) -> VSOut {
  var vOut: VSOut;
  vOut.position = uni.worldViewProjectionMatrix * vin.position;
  vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
  return vOut;
}

@fragment fn fs(vin: VSOut) -> @location(0) vec4f {
  let lightDirection = normalize(vec3f(4, 10, 6));
  let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
  return vec4f(uni.color.rgb * light, uni.color.a);
}
struct Uniforms {
  modelViewProjectionMatrix : array<mat4x4f, 16>,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix[instanceIdx] * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0));
  return output;
}
@fragment
fn main(
  @location(0) fragColor: vec4f
) -> @location(0) vec4f {
  return fragColor;
}
@group(0) @binding(0) var depthTexture: texture_depth_2d;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0);
  return vec4f(depthValue, depthValue, depthValue, 1.0);
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4(pos[VertexIndex], 0.0, 1.0);
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragColor : vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  output.fragColor = color;
  return output;
}@group(1) @binding(0) var depthTexture: texture_depth_2d;

@fragment
fn main(
  @builtin(position) coord: vec4f,
  @location(0) clipPos: vec4f
) -> @location(0) vec4f {
  let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0);
  let v : f32 = abs(clipPos.z / clipPos.w - depthValue) * 2000000.0;
  return vec4f(v, v, v, 1.0);
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) clipPos : vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  output.clipPos = output.Position;
  return output;
}
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f
) -> @builtin(position) vec4f {
  return camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
}
@fragment
fn main() -> @location(0) vec4f {
  return vec4(0.0, 0.0, 0.0, 1.0);
}struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}
@group(0) @binding(0) var mySampler : sampler;
@group(0) @binding(1) var myTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

@fragment
fn frag_main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV);
}
@fragment
fn main() -> @location(0) vec4f {
  return vec4(1.0, 0.0, 0.0, 1.0);
}@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  return fragPosition;
}
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_external;

@fragment
fn main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  return textureSampleBaseClampToEdge(myTexture, mySampler, fragUV);
}
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2(0.0, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5)
  );

  return vec4f(pos[VertexIndex], 0.0, 1.0);
}
