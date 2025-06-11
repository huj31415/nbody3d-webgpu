// point renderer from webgpufundamentals, added camera controls

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
    const x = Math.cos(phi) * r * Math.random();
    const z = Math.sin(phi) * r * Math.random();
    vertices.push(x * radius, y * radius, z * radius);
  }
  return new Float32Array(vertices);
}

let adapter, device;

const ROT_SPEED = 0.005;
const PAN_SPEED = 0.001;

const ZOOM_SPEED = 0.002;
const FOV_SPEED = 0.0002;

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

  const module = device.createShaderModule({
    code: `
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
        // ensure points are at least 2px wide
        let pointPos = vec4f(pos / uni.resolution * max(2 * clipPos.w, uni.size), 0, 0);
        vsOut.position = clipPos + pointPos;
        return vsOut;
      }

      fn colorMap(value: f32) -> vec3<f32> {
        return vec3<f32>(value, 1.0 - abs(value - 0.5), 1.0 - value); // rgb
        // return vec3<f32>(value, abs(value) / 2 - 0.5, -value); // yrkbc
      }

      @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        return vec4f(colorMap(vsOut.position.w), 1); //vec4f(1, 0.5, 0.2, 1);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    label: '3d points',
    layout: 'auto',
    vertex: {
      module,
      buffers: [
        {
          arrayStride: 3 * 4, // 3 floats, 4 bytes each
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
          ],
        },
      ],
    },
    fragment: {
      module,
      targets: [
        {
          format: swapChainFormat,
        },
      ],
    },
  });

  const vertexData = createPoints({
    radius: 1,
    numSamples: 2000,
  });
  const kNumPoints = vertexData.length / 3;

  const vertexBuffer = device.createBuffer({
    label: 'vertex buffer vertices',
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

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
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
    ],
  });

  const renderPassDescriptor = {
    label: 'our basic canvas renderPass',
    colorAttachments: [
      {
        // view: <- to be filled out when we render
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };
  Number.prototype.toRad = function () { return this * Math.PI / 180; }
  Number.prototype.toDeg = function () { return this / Math.PI * 180; }

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

  const worldUp = vec3.fromValues(0, 1, 0);

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
      let pan = vec3.scaleAndAdd(vec3.scale(camera.viewRight(), -dx * PAN_SPEED), camera.viewUp(), dy * PAN_SPEED);

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
      camera.radius = initial / Math.tan(camera.fov / 2);
    } else {
      // move camera in/out
      camera.radius = (camera.radius + e.deltaY * ZOOM_SPEED).clamp(0.1, 100);
    }
    updateCameraPosition();
  }, { passive: false });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Alt") e.preventDefault();
  });

  function render(time) {
    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

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
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, kNumPoints);
    pass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

function fail(msg) {
  alert(msg);
}

main();