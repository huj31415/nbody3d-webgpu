// point renderer from webgpufundamentals, added camera controls


Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };
Number.prototype.toRad = function () { return this * Math.PI / 180; }
Number.prototype.toDeg = function () { return this / Math.PI * 180; }

let adapter, device;

const ROT_SPEED = 0.005;
const PAN_SPEED = 0.001;

const ZOOM_SPEED = 0.002;
const FOV_SPEED = 0.0002;

// camera state
const camera = {
  // spherical coords around target:
  target: vec3.fromValues(0, 0, 0),
  radius: 2,
  azimuth: 0,               // horizontal angle, radians
  elevation: 0,             // vertical angle, radians
  // for projection:
  fov: (60).toRad(),
  near: 0.1,
  far: 100,
  viewDir: () => vec3.normalize(vec3.subtract(camera.target, camera.position)),
  viewRight: () => vec3.normalize(vec3.cross(camera.viewDir(), worldUp)),
  viewUp: () => vec3.normalize(vec3.cross(camera.viewRight(), camera.viewDir())),
};

// world up vector
const worldUp = vec3.fromValues(0, 1, 0);

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
 * @param {vec3} center  A length-3 array [cx,cy,cz]
 * @param {vec3} normal  A length-3 array [nx,ny,nz] (need not be unit)
 * @param {number} radius                     Disk radius
 * @param {number} count                      Number of points
 * @returns {Float32Array}                    Packed [x,y,z, x,y,z, ...]
 */
function randomPointsInDisk(center, normal, radius, count) {
  const out = new Float32Array(count * 4);

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
    const idx = i * 4;
    out[idx + 0] = center[0] + ux + vx;
    out[idx + 1] = center[1] + uy + vy;
    out[idx + 2] = center[2] + uz + vz;
    out[idx + 3] = Math.random() * 2000; // mass
  }

  return out;
}


async function main() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const halfWidth = Math.round(width / 2);
  const halfHeight = Math.round(height / 2);

  // WebGPU Setup
  const canvas = document.getElementById("canvas");
  canvas.width = width;
  canvas.height = height;
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
        size: f32,
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
        let quad = array(
          vec2f(-1, -1),
          vec2f( 1, -1),
          vec2f(-1,  1),
          vec2f(-1,  1),
          vec2f( 1, -1),
          vec2f( 1,  1),
        );
        var vsOut: VSOutput;
        let body = bodies[instNdx];
        let pos = quad[vNdx];
        let clipPos = uni.matrix * vec4f(body.pos, 1);
        // ensure points are at least 2px wide, get radius from mass (r = cbrt(mass * 3/4 / pi))
        let pointPos = vec4f(pos / uni.resolution * 2 * max(clipPos.w, pow(body.mass * 0.239, 0.333333)), 0, 0);
        vsOut.position = clipPos + pointPos;
        vsOut.uv = pos;
        return vsOut;
      }

      fn colorMap(value: f32) -> vec3f {
        return vec3<f32>(value, 1.0 - abs(value - 0.5), 1.0 - value); // rgb
      }

      @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        const innerRadius = 0.9;
        const outerRadius = 1;
        let distance = length(vsOut.uv);
        let alpha = 1.0 - smoothstep(innerRadius, outerRadius, distance);
        if (alpha < 0.5) {discard;}
        return vec4f(colorMap(vsOut.position.w) * alpha, alpha);
      }
    `,
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX,  buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX,  buffer: { type: "uniform" } },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ renderBindGroupLayout ],
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

  const pointData = randomPointsInDisk(
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(Math.random(), Math.random(), Math.random()),
    2,
    10000
  );
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
  const kSizeOffset = 18;
  const matrixValue = uniformValues.subarray(
    kMatrixOffset, kMatrixOffset + 16);
  const resolutionValue = uniformValues.subarray(
    kResolutionOffset, kResolutionOffset + 2);
  const sizeValue = uniformValues.subarray(
    kSizeOffset, kSizeOffset + 1);


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

  // compute position from spherical coords
  function updateCameraPosition() {
    const { azimuth: phi, elevation: theta, radius: r, target: T } = camera;
    const x = r * Math.cos(theta) * Math.sin(phi);
    const y = r * Math.sin(theta);
    const z = r * Math.cos(theta) * Math.cos(phi);
    camera.position = vec3.fromValues(T[0] + x, T[1] + y, T[2] + z);
  }
  updateCameraPosition();

  // camera interaction state
  let state = {
    orbitActive: false,
    panActive: false,
    lastX: 0,
    lastY: 0,
  };

  // DOM event handlers
  canvas.addEventListener('contextmenu', e => e.preventDefault()); // disable context menu

  canvas.addEventListener('mousedown', e => {
    if (e.button === 0) state.orbitActive = true; // left click to orbit
    if (e.button === 1) resetCam();
    if (e.button === 2) state.panActive = true;   // right click to pan
    state.lastX = e.clientX;
    state.lastY = e.clientY;
  });
  window.addEventListener('mouseup', e => {
    if (e.button === 0) state.orbitActive = false;
    if (e.button === 2) state.panActive = false;
  });
  canvas.addEventListener('mousemove', e => {
    const dx = e.clientX - state.lastX;
    const dy = e.clientY - state.lastY;
    state.lastX = e.clientX;
    state.lastY = e.clientY;


    // Orbit
    if (state.orbitActive) {
      camera.azimuth -= dx * ROT_SPEED;
      camera.elevation += dy * ROT_SPEED;
      // clamp elevation to [-89, +89] deg
      const limit = Math.PI / 2 - 0.01;
      camera.elevation = (camera.elevation).clamp(-limit, limit);
      updateCameraPosition();
    }

    // Pan within view-plane
    if (state.panActive) {
      // pan delta = (-right * dx + upReal * dy) * PAN_SPEED
      let adjustedPanSpeed = PAN_SPEED * camera.radius * camera.fov;
      let pan = vec3.scaleAndAdd(
        vec3.scale(camera.viewRight(), -dx * adjustedPanSpeed),
        camera.viewUp(),
        dy * adjustedPanSpeed
      );

      // apply to target and camera position
      camera.target = vec3.add(camera.target, pan);
      camera.position = vec3.add(camera.position, pan);
    }
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();

    if (e.altKey) {
      // adjust FOV
      const initial = Math.tan(camera.fov / 2) * camera.radius;
      camera.fov = (camera.fov + e.deltaY * FOV_SPEED).clamp((10).toRad(), (120).toRad());
      if (e.shiftKey) camera.radius = initial / Math.tan(camera.fov / 2);
    } else {
      // move camera in/out
      camera.radius = (camera.radius + e.deltaY * ZOOM_SPEED).clamp(camera.near, camera.far);
    }
    updateCameraPosition();
  }, { passive: false });

  function resetCam() {
    camera.target = vec3.fromValues(0, 0, 0);
    camera.radius = 2;
    camera.azimuth = 0;
    camera.elevation = 0;
    camera.fov = (60).toRad();
    updateCameraPosition();
  }

  window.addEventListener("keydown", (e) => {
    switch (e.key) {
      case "Alt":
        e.preventDefault();
        break;
      case "Home":
        resetCam();
        break;
    }
  });

  let rafId;
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

    // Set the size in the uniform values
    sizeValue[0] = 5;

    // Set the matrix in the uniform values
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const proj = mat4.perspective(camera.fov, aspect, camera.near, camera.far);
    const view = mat4.lookAt(
      camera.position,
      camera.target,
      worldUp
    );
    mat4.multiply(proj, view, matrixValue);

    // Update the resolution in the uniform values
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

    rafId = requestAnimationFrame(render);
  }

  rafId = requestAnimationFrame(render);

  window.onresize = () => {
    cancelAnimationFrame(rafId);
    main();
  };
}

function fail(msg) {
  alert(msg);
}

main();