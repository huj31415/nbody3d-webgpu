// point renderer from webgpufundamentals, added camera controls

let adapter, device;

const canvas = document.getElementById("canvas");

function createPoints({
  numSamples,
  radius,
}) {
  const vertices = [];
  const increment = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < numSamples; ++i) {
    const offset = 2 / numSamples;
    const y = ((i * offset) - 1) + (offset / 2);
    const r = Math.sqrt(1 - Math.pow(y, 2));
    const phi = (i % numSamples) * increment;
    const x = Math.cos(phi) * r;// * Math.random();
    const z = Math.sin(phi) * r;// * Math.random();
    vertices.push(x * radius, y * radius, z * radius, Math.random() * 2000);

  }
  return new Float32Array(vertices);
}

/**
 * Generates N random points uniformly distributed inside
 * the disk of given radius, centered at `center`, whose
 * plane is oriented by `normal`.
 *
 * @param {vec3} center  vec3 representing the center [cx,cy,cz]
 * @param {vec3} normal  vec3 representing the normal [nx,ny,nz] (need not be unit)
 * @param {number} radius  Disk radius
 * @param {number} count  Number of points
 * @returns {Array}  Packed coords and masses [x,y,z,m x,y,z,m ...]
 */
function randomDiskPoints(center, normal, radius, count) {
  const out = [];

  // 1) normalize the normal
  const n = vec3.normalize(normal);

  // 2) build an orthonormal basis {u,v} for the disk plane
  //    choose any vector not parallel to n:
  const tmp = Math.abs(n[0]) > 0.9
    ? vec3.fromValues(0, 1, 0)
    : vec3.fromValues(1, 0, 0);

  const u = vec3.normalize(vec3.cross(tmp, n));
  const v = vec3.cross(n, u);

  // 3) sample points
  for (let i = 0; i < count; i++) {
    // random radius is proportional sqrt(U) for uniform distribution
    const t = Math.random();
    const r = t * Math.sqrt(t) * radius;
    // random angle
    const theta = Math.random() * 2 * Math.PI;

    // local offset = u*(r*cos theta) + v*(r*sin theta)
    const ux = u[0] * (r * Math.cos(theta));
    const uy = u[1] * (r * Math.cos(theta));
    const uz = u[2] * (r * Math.cos(theta));

    const vx = v[0] * (r * Math.sin(theta));
    const vy = v[1] * (r * Math.sin(theta));
    const vz = v[2] * (r * Math.sin(theta));

    // world position = center + offset_u + offset_v
    // const idx = i * 4;
    out.push(center[0] + ux + vx);
    out.push(center[1] + uy + vy);
    out.push(center[2] + uz + vz);
    out.push(Math.random() * 2000); // mass
  }

  return out;
}


async function main() {

  // WebGPU Setup
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  if (!adapter) {
    adapter = await navigator.gpu?.requestAdapter();
    device = await adapter?.requestDevice();
  }
  if (!device) {
    alert("Browser does not support WebGPU");
    document.body.textContent = "WebGPU is not supported in this browser.";
    return;
  }
  const context = canvas.getContext("webgpu");
  const swapChainFormat = "bgra8unorm";
  context.configure({
    device: device,
    format: swapChainFormat,
  });

  const computeModule = device.createShaderModule({
    code: `
      struct Body {
        pos: vec3f,
        mass: f32
      }

      struct Uniforms {
        matrix: mat4x4f,
        resolution: vec2f,
        size: f32,
      };

      @group(0) @binding(0) var<storage, read> bodiesIn: array<Body>;
      @group(0) @binding(1) var<storage, read_write> bodiesOut: array<Body>;
      @group(0) @binding(2) var<uniform> uni: Uniforms;

      // calculate acceleration between bodies b1 and b2
      // returns total acceleration
      fn bodyAccel(b1: Body, b2: Body, a1: vec3f) -> vec3f {
        let r = b2.pos - b1.pos;
        let distSqr = dot(r, r) + 1e-4; // eps = 1e-2
        let invDist3 = inverseSqrt(distSqr * distSqr * distSqr);
        return a1 + (b2.mass * invDist3 * r);
      }

      // each thread accumulates accelerations for one body
      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x = i32(global_id.x);
        let y = i32(global_id.y);
      }
    `
  })

  const renderModule = device.createShaderModule({
    code: `
      struct Body {
        pos: vec3f,
        mass: f32
      }

      struct Uniforms {
        matrix: mat4x4f,
        resolution: vec2f,
        f: f32,
      };

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
      };

      @group(0) @binding(0) var<storage, read> bodies: array<Body>;
      @group(0) @binding(1) var<uniform> uni: Uniforms;

      @vertex fn vs(
        @builtin(instance_index) instNdx: u32,
        @builtin(vertex_index) vNdx: u32,
      ) -> VSOutput {
        var vsOut: VSOutput;
        let body = bodies[instNdx];

        let quad = array(
          vec2f(-1, -1),
          vec2f( 1, -1),
          vec2f(-1,  1),
          vec2f(-1,  1),
          vec2f( 1, -1),
          vec2f( 1,  1),
        );
        let uv = quad[vNdx];

        //get radius from mass (r = cbrt(mass * 3/4 / pi))
        let radius = pow(body.mass * 0.239, 1.0/3.0);
        
        let clipPos = uni.matrix * vec4f(body.pos, 1);

        // ensure points are at least 2px wide
        let clipOffset = vec4f(uv / uni.resolution * 2 * max(clipPos.w, radius * uni.f), 0, 0);

        vsOut.position = clipPos + clipOffset;
        vsOut.uv = uv;
        return vsOut;
      }

      fn colorMap(value: f32) -> vec3f {
        return vec3<f32>(value, 1.0 - abs(value - 0.5), 1.0 - value); // rgb
      }

      @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        // circle SDF with anti-aliasing
        // const innerRadius = 0.9;
        // const outerRadius = 1;
        // let distance = length(vsOut.uv);
        // let alpha = 1.0 - smoothstep(innerRadius, outerRadius, distance);
        // if (alpha < 0.9) { discard; }
        // return vec4f(colorMap(vsOut.position.w) * alpha, alpha);

        // circle SDF
        let dist = length(vsOut.uv) - 1;
        if (dist > 0.0) { discard; }
        if (dist > -0.1) { return vec4f(0); }
        return vec4f(colorMap(vsOut.position.w), 1);
      }
    `,
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    label: '3d point renderer',
    layout: renderPipelineLayout,// 'auto',
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [
        {
          format: swapChainFormat,
        },
      ],
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });

  const pointData = new Float32Array(randomDiskPoints(
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(1, 1, 1), // vec3.fromValues(Math.random(), Math.random(), Math.random()),
    2 * Math.random() + 2,
    10000
  ).concat(randomDiskPoints(
    vec3.scale(vec3.fromValues(Math.random(), Math.random(), Math.random()), 5),
    vec3.fromValues(Math.random(), Math.random(), Math.random()),
    2 * Math.random() + 2,
    10000
  )));
  // createPoints({
  //   radius: 1,
  //   numSamples: 1000,
  // });
  const kNumPoints = pointData.length / 4;

  const pointBuffer = device.createBuffer({
    label: 'point buffer',
    size: pointData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(pointBuffer, 0, pointData);

  const uniformValues = new Float32Array(16 + 2 + 1 + 1);
  const uniformBuffer = device.createBuffer({
    size: uniformValues.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const kMatrixOffset = 0;
  const kResolutionOffset = 16;
  const kFOffset = 18;
  const matrixValue = uniformValues.subarray(
    kMatrixOffset, kMatrixOffset + 16);
  const resolutionValue = uniformValues.subarray(
    kResolutionOffset, kResolutionOffset + 2);
  const fValue = uniformValues.subarray(
    kFOffset, kFOffset + 1);


  const bindGroup = device.createBindGroup({
    layout: renderBindGroupLayout, //pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pointBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ],
  });

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [
      {
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
    depthStencilAttachment: {
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  };

  let depthTexture;

  function render(time) {
    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    // If we don't have a depth texture OR if its size is different
    // from the canvasTexture when make a new depth texture
    if (!depthTexture ||
      depthTexture.width !== canvasTexture.width ||
      depthTexture.height !== canvasTexture.height) {
      if (depthTexture) {
        depthTexture.destroy();
      }
      depthTexture = device.createTexture({
        size: [canvasTexture.width, canvasTexture.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();


    // Set the matrix in the uniform values
    matrixValue.set(matrix);

    // Set the f value in the uniform values
    fValue[0] = fVal;
    // Update the resolution in the uniform values - only when resizing?
    resolutionValue.set([canvasTexture.width, canvasTexture.height]);


    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(renderPipeline);
    pass.setVertexBuffer(0, pointBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, kNumPoints);
    pass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    requestAnimationFrame(render);
  }

  updateMatrix();
  requestAnimationFrame(render);
}
window.onresize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  updateMatrix();
};


function fail(msg) {
  alert(msg);
}

main();