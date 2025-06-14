// point renderer from webgpufundamentals, added camera controls

let adapter, device;

const TILE_SIZE = 256;

let sizeFactor = window.outerHeight;

const canvas = document.getElementById("canvas");

const getRadius = (mass) => Math.cbrt(mass / (4/3 * Math.PI)) / sizeFactor;

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

  const cMass = 1e6;
  const maxOuterMass = 100;
  const cRadius = getRadius(cMass) + getRadius(maxOuterMass);
  out.push(center[0]);
  out.push(center[1]);
  out.push(center[2]);
  out.push(cMass); // mass

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
    const r = t * Math.sqrt(t) * radius + cRadius;
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
    out.push(Math.random() * maxOuterMass); // mass
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
      struct Uniforms {
        matrix: mat4x4f,
        sizeRatio: vec2f,
        f: f32,
        dt: f32,
        cameraPos: vec3f,
        G: f32
      };

      // position and mass buffer
      @group(0) @binding(0) var<storage, read_write> bodies: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> vel: array<vec3f>;
      @group(0) @binding(2) var<storage, read_write> accel: array<vec3f>;
      @group(0) @binding(3) var<uniform> uni: Uniforms;

      const TILE_SIZE = ${TILE_SIZE};
      var<workgroup> tile: array<vec4f, TILE_SIZE>;

      // calculate gravitational acceleration between bodies b1 and b2
      fn bodyAccel(b1: vec4f, b2: vec4f) -> vec3f {
        let r = b2.xyz - b1.xyz;
        let distSqr = dot(r, r) + 1e-6; // eps = 1e-3
        let invDistCubed = inverseSqrt(distSqr * distSqr * distSqr);
        return b2.w * invDistCubed * r;
      }

      // each thread accumulates accelerations for one body
      @compute @workgroup_size(TILE_SIZE)
      fn main(
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u
      ) {
        let bodyIndex = gid.x;
        let body = bodies[bodyIndex];

        let nBodies = arrayLength(&bodies);

        // number of tiles rounded up
        let nTiles = (nBodies + TILE_SIZE - 1) / TILE_SIZE;

        var newAccel = vec3f(0);

        // iterate across tiles and accumulate acceleration
        for (var t = 0u; t < nTiles; t++) {
          let index = t * TILE_SIZE + lid.x;
          if (index < nBodies) { tile[lid.x] = bodies[index]; }

          // sync tile loading
          workgroupBarrier();

          // iterate within tile, unroll
          for (var i = 0u; i < TILE_SIZE; i++) {
            let body2 = tile[i];
            if (bodyIndex == index) { continue; }
            newAccel += bodyAccel(body, body2);
          }

          // sync before loading next tile
          workgroupBarrier();
        }
        
        // verlet position update
        bodies[bodyIndex] = vec4f(body.xyz + uni.dt * (vel[bodyIndex] + 0.5 * accel[bodyIndex] * uni.dt), body.w);

        // verlet velocity update
        vel[bodyIndex] = vel[bodyIndex] + 0.5 * (accel[bodyIndex] + newAccel) * uni.dt;

        // save acceleration for the next time step
        accel[bodyIndex] = newAccel;
      }
    `
  })

  const renderModule = device.createShaderModule({
    code: `
      struct Uniforms {
        matrix: mat4x4f,
        sizeRatio: vec2f,
        f: f32,
        dt: f32,
        cameraPos: vec3f,
        G: f32
      };

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
        @location(1) r: f32
      };

      @group(0) @binding(0) var<storage, read> bodies: array<vec4f>;
      @group(0) @binding(1) var<uniform> uni: Uniforms;

      
      @vertex fn vs(
        @builtin(instance_index) instNdx: u32,
        @builtin(vertex_index) vNdx: u32,
      ) -> VSOutput {
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
        let radius = pow(body.w / 4.189, 1.0/3.0);
        
        // billboards parallel to view plane with min size
        // let clipPos = uni.matrix * vec4f(body.xyz, 1);
        // // ensure points are at least 2px wide
        // let clipOffset = vec4f(uv * 2 * uni.sizeRatio * max(clipPos.w, radius * uni.f), 0, 0); // / uni.resolution
        // vsOut.position = clipPos + clipOffset;

        // billboards perpendicular to camera
        
        let worldPos = body.xyz;

        // compute view vector from camera to particle
        let viewVec = worldPos - uni.cameraPos;
        let viewDir = normalize(viewVec);

        // compute billboard basis
        let right = normalize(cross(viewDir, vec3f(0.0, 1.0, 0.0)));
        let up = normalize(cross(right, viewDir));

        // offset in local camera-facing plane
        let worldOffset = (right * uv.x + up * uv.y) * max(radius, 2 * length(viewVec) / uni.f) * uni.sizeRatio.y;

        // final world position of the quad vertex
        let cornerPos = worldPos + worldOffset;

        var vsOut: VSOutput;
        vsOut.position = uni.matrix * vec4f(cornerPos, 1.0);
        vsOut.uv = uv;
        vsOut.r = radius;
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
        if (dist > -0.5 / vsOut.r) { return vec4f(0); }
        return vec4f(colorMap(vsOut.position.w), 1);
      }
    `,
  });

  // const renderBindGroupLayout = device.createBindGroupLayout({
  //   entries: [
  //     { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
  //     { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
  //   ],
  // });

  // const renderPipelineLayout = device.createPipelineLayout({
  //   bindGroupLayouts: [renderBindGroupLayout],
  // });

  const renderPipeline = device.createRenderPipeline({
    label: '3d point renderer',
    layout: 'auto', // renderPipelineLayout,
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
    vec3.scale(vec3.fromValues(Math.random()-.5, Math.random()-.5, Math.random()-.5), 5),
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

  const uniformValues = new Float32Array(24); // 16 mat + 2 res + 2 f dt + 3 right + 1 g + 3 up
  const uniformBuffer = device.createBuffer({
    size: uniformValues.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const kMatrixOffset = 0;
  const kSizeFactorOffset = 16;
  const kFOffset = 18;
  const kDtOffset = 19;
  const kCamPosOffset = 20;
  const kGOffset = 23;
  const uni = {
    matrixValue: uniformValues.subarray(kMatrixOffset, kMatrixOffset + 16),
    sizeFactorValue: uniformValues.subarray(kSizeFactorOffset, kSizeFactorOffset + 2),
    fValue: uniformValues.subarray(kFOffset, kFOffset + 1),
    dtValue: uniformValues.subarray(kDtOffset, kDtOffset + 1),
    cameraPosValue: uniformValues.subarray(kCamPosOffset, kCamPosOffset + 3),
    GValue: uniformValues.subarray(kGOffset, kGOffset + 1),
  }


  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0), // renderBindGroupLayout, 
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
    uni.matrixValue.set(matrix);

    // Set the f value in the uniform values
    uni.fValue[0] = fVal;
    
    uni.cameraPosValue.set(camera.position);


    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(renderPipeline);
    pass.setVertexBuffer(0, pointBuffer);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(6, kNumPoints);
    pass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    requestAnimationFrame(render);
  }

  window.onresize = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    aspect = canvas.clientWidth / canvas.clientHeight;
    uni.sizeFactorValue.set([1 / (sizeFactor * aspect), 1 / sizeFactor]);
    updateMatrix();
  };

  updateCameraPosition();
  uni.sizeFactorValue.set([1 / (sizeFactor * aspect), 1 / sizeFactor]);
  requestAnimationFrame(render);
}


function fail(msg) {
  alert(msg);
}

main();