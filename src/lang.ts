let builtinTypes: string[] = [
    // Boolean Types
    'bool',
    'true',
    'false',
    // Integer Types
    'i32',
    'u32',
    // Floating Point Types
    'f32',
    'f16',
    // Vector Types
    'vec2', 'vec3', 'vec4',
    'vec2i', 'vec3i', 'vec4i',
    'vec2u', 'vec3u', 'vec4u',
    'vec2f', 'vec3f', 'vec4f',
    'vec2h', 'vec3h', 'vec4h',
    // Matrix Types
    'mat2x2', 'mat2x3', 'mat2x4',
    'mat3x2', 'mat3x3', 'mat3x4',
    'mat4x2', 'mat4x3', 'mat4x4',
    'mat2x2f', 'mat2x3f', 'mat2x4f',
    'mat3x2f', 'mat3x3f', 'mat3x4f',
    'mat4x2f', 'mat4x3f', 'mat4x4f',
    'mat2x2h', 'mat2x3h', 'mat2x4h',
    'mat3x2h', 'mat3x3h', 'mat3x4h',
    'mat4x2h', 'mat4x3h', 'mat4x4h',
    // Atomic Types
    'atomic',
    // Array Types
    'array',
    // Texture Types
    'texture_1d', 'texture_2d',
    'texture_3d', 'texture_cube',
    'texture_cube_array',
    'texture_multisampled_2d',
    'texture_depth_multisampled_2d',
    'texture_external',
    'texture_storage_1d',
    'texture_storage_2d',
    'texture_storage_2d_array',
    'texture_storage_3d',
    'texture_depth_2d',
    'texture_depth_2d_array',
    'texture_depth_cube',
    'texture_depth_cube_array',
    // Sampler Types
    'sampler', 'sampler_comparison',
];

let builtinFunctions: string[] = [
    // Bit
    'bitcast',

    'all', 'any', 'select',

    'arrayLength',

    'abs', 'acos', 'acosh', 'asin', 'asinh',
    'atan', 'atan2', 'cell', 'clamp', 'cos', 'cosh',
    'countLeadingZeros', 'countOneBits',
    'countTrailingZeros',
    'countOneBits',
    'countTrailingZeros',
    'cross',
    'degress',
    'determinant',
    'distance',
    'dot',
    'dot4U8Packed',
    'dot4I8Packed',
    'exp',
    'exp2',
    'extractBits',
    'faceForward',
    'firstLeadingBit',
    'firstTrailingBit',
    'floor',
    'fma',
    'fract',
    'frexp',
    'insertBits',
    'inverseSqrt',
    'ldexp',
    'length',
    'log',
    'log2',
    'max',
    'min',
    'mix',
    'modf',
    'normalize',
    'pow',
    'quantizeToF16',
    'radians',
    'reflext',
    'refract',
    'reverseBits',
    'round',
    'saturate',
    'sign',
    'sin',
    'sinh',
    'smoothstep',
    'sqrt',
    'step',
    'tan',
    'tanh',
    'transpose',
    'trunc',
    // Derivative
    'dpdx',
    'dpdxCoarse',
    'dpdxFine',
    'dpdy',
    'dpdyCoarse',
    'dpdyFine',
    'fwidth',
    'fwidthCoarse',
    'fwidthFine',
    // Texture
    'textureDimensions',
    'textureGather',
    'textureGatherCompare',
    'textureLoad',
    'textureNumLayers',
    'textureNumLevels',
    'textureNumSamples',
    'textureSample',
    'textureSampleBias',
    'textureSampleCompare',
    'textureSampleCompareLevel',
    'textureSampleGrad',
    'textureSampleGrad',
    'textureSampleLevel',
    'textureSampleBaseClampToEdge',
    'textureStore',
    // Atomic
    'atomicLoad',
    'atomicStore',
    'atomicAdd',
    'atomicSub',
    'atomicMax',
    'atomicMin',
    'atomicAnd',
    'atomicOr',
    'atomicXor',
    'atomicExchange',
    'atomicCompareExchangeWeak',
    // Data Packing
    'pack4x8snorm',
    'pack4x8unorm',
    'pack4xI8',
    'pack4xU8',
    'pack4xI8Clamp',
    'pack4xU8Clamp',
    'pack2x16snorm',
    'pack2x16unorm',
    'pack2x16float',
    // Data Unpacking
    'unpack4x8snorm',
    'unpack4x8unorm',
    'unpack4xI8',
    'unpack4xU8',
    'unpack4xI8Clamp',
    'unpack4xU8Clamp',
    'unpack2x16snorm',
    'unpack2x16unorm',
    'unpack2x16float',
    // Synchronization
    'storageBarrier',
    'textureBarrier',
    'workgroupBarrier',
    'workgroupUniformLoad',
];

let builtinIOs: string[] = [
    'frag_depth',
    'front_facing',
    'global_invocation_id',
    'instance_index',
    'local_invocation_id',
    'local_invocation_index',
    'num_workgroups',
    'position',
    'sample_index',
    'sample_mask',
    'vertex_index',
    'workgroup_id',
];


export function isBuiltinType(s: string): boolean {
    return builtinTypes.find((t: string) => {
        return t == s;
    }) != undefined;
}

export function isBuiltinFunction(s: string): boolean {
    return builtinFunctions.find((f: string) => {
        return f == s;
    }) != undefined;
}

export function isBuiltinIO(s: string): boolean {
    return builtinIOs.find((io: string) => {
        return io == s;
    }) != undefined;
}

export function isAddressSpace(s: string): boolean {
    let addressSpaces: string[] = [
        'function',
        'private',
        'workgroup',
        'uniform',
        'storage',
        'handle',
    ]

    return addressSpaces.find((as: string) => {
        return as == s;
    }) != undefined;
}

export function isMemoryAccess(s: string): boolean {
    let acceses: string[] = [
        'read',
        'write',
        'read_write',
    ]

    return acceses.find((as: string) => {
        return as == s;
    }) != undefined;
}

export function isReserved(s: string): boolean {
    let reserveds: string[] = [
        'NULL',
        'Self',
        'abstract',
        'active',
        'alignas',
        'alignof',
        'as',
        'asm',
        'asm_fragment',
        'async',
        'attribute',
        'auto',
        'await',
        'become',
        'binding_array',
        'cast',
        'catch',
        'class',
        'co_await',
        'co_return',
        'co_yield',
        'coherent',
        'column_major',
        'common',
        'compile',
        'compile_fragment',
        'concept',
        'const_cast',
        'consteval',
        'constexpr',
        'constinit',
        'crate',
        'debugger',
        'decltype',
        'delete',
        'demote',
        'demote_to_helper',
        'do',
        'dynamic_cast',
        'enum',
        'explicit',
        'export',
        'extends',
        'extern',
        'external',
        'fallthrough',
        'filter',
        'final',
        'finally',
        'friend',
        'from',
        'fxgroup',
        'get',
        'goto',
        'groupshared',
        'highp',
        'impl',
        'implements',
        'import',
        'inline',
        'instanceof',
        'interface',
        'layout',
        'lowp',
        'macro',
        'macro_rules',
        'match',
        'mediump',
        'meta',
        'mod',
        'module',
        'move',
        'mut',
        'mutable',
        'namespace',
        'new',
        'nil',
        'noexcept',
        'noinline',
        'nointerpolation',
        'noperspective',
        'null',
        'nullptr',
        'of',
        'operator',
        'package',
        'packoffset',
        'partition',
        'pass',
        'patch',
        'pixelfragment',
        'precise',
        'precision',
        'premerge',
        'priv',
        'protected',
        'pub',
        'public',
        'readonly',
        'ref',
        'regardless',
        'register',
        'reinterpret_cast',
        'require',
        'resource',
        'restrict',
        'self',
        'set',
        'shared',
        'sizeof',
        'smooth',
        'snorm',
        'static',
        'static_assert',
        'static_cast',
        'std',
        'subroutine',
        'super',
        'target',
        'template',
        'this',
        'thread_local',
        'throw',
        'trait',
        'try',
        'type',
        'typedef',
        'typeid',
        'typename',
        'typeof',
        'union',
        'unless',
        'unorm',
        'unsafe',
        'unsized',
        'use',
        'using',
        'varying',
        'virtual',
        'volatile',
        'wgsl',
        'where',
        'with',
        'writeonly',
        'yield',
    ];

    return reserveds.find((w: string) => {
        return s == w;
    }) != undefined;
}

export function isTexelFormats(s: string): boolean {
    let texelFormats = [
        'rgba8unorm',
        'rgba8snorm',
        'rgba8uint',
        'rgba8sin',
        'rgba16uint',
        'rgba16sint',
        'rgba16float',
        'r32uint',
        'r32sint',
        'r32float',
        'rg32uint',
        'rg32sint',
        'rg32float',
        'rgba32uint',
        'rgba32sint',
        'rgba32float',
        'bgra8unorm',
    ]

    return texelFormats.find((tf: string) => {
        return tf == s;
    }) != undefined;
}

export function isInterpolationType(s: string): boolean {
    let types = [
        'perspective',
        'linear',
        'flat'
    ];

    return types.find((t: string) => {
        return t == s;
    }) != undefined;
}

export function isShaderStageAttribute(s: string) {
    return s == 'vertex' ||
           s == 'compute' ||
           s == 'fragment';
}

export function isBuiltinSymbol(s: string): boolean {
    return isBuiltinFunction(s) ||
        isBuiltinIO(s) ||
        isAddressSpace(s) ||
        isMemoryAccess(s) ||
        isBuiltinType(s) ||
        isReserved(s);
}
