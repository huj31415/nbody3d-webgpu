// point renderer from webgpufundamentals, added camera controls

let adapter, device;

const TILE_SIZE = 256;

let G = 0.0001;
let dt = 1e-4;

let sizeFactor = window.outerHeight;

const canvas = document.getElementById("canvas");

let bodyBuffer, velBuffer, accelBuffer, nBodies;

const uni = {};

const uniformValues = new Float32Array(24); // 16 mat + 3 camPos + 1 size + 3 f,dt,g + 1 pad

const kMatrixOffset = 0;
const kCamPosOffset = 16;
const kSizeFactorOffset = 19;
const kFOffset = 20;
const kDtOffset = 21;
const kGOffset = 22;

uni.matrixValue = uniformValues.subarray(kMatrixOffset, kMatrixOffset + 16);
uni.cameraPosValue = uniformValues.subarray(kCamPosOffset, kCamPosOffset + 3);
uni.sizeFactorValue = uniformValues.subarray(kSizeFactorOffset, kSizeFactorOffset + 1);
uni.fValue = uniformValues.subarray(kFOffset, kFOffset + 1);
uni.dtValue = uniformValues.subarray(kDtOffset, kDtOffset + 1);
uni.GValue = uniformValues.subarray(kGOffset, kGOffset + 1);


/**
 * Generates N random points distributed inside
 * the disk of given radius, centered at `center`, whose
 * plane is oriented by `normal`.
 * 
 * Pass arguments as a list of lists of args, eg. [[center, norm, r, cnt], [center, norm, r, cnt]]
 *
 * @param {vec3} center  vec3 representing the center [cx,cy,cz]
 * @param {vec3} centerV  vec3 representing the velocity of the galaxy [vx,vy,vz]
 * @param {vec3} normal  vec3 representing the normal [nx,ny,nz] (need not be unit)
 * @param {number} radius  Disk radius
 * @param {number} count  Number of points
 * @returns {Array}  Packed coords and masses [x,y,z,m x,y,z,m ...]
 */
function generateGalaxy(list) {
  const pos = [];
  const vel = [];
  const CoM = vec3.create();
  let totalMass = 0;

  list.forEach(config => {
    const [center, centerV, normal, radius, count] = config;

    const cMass = 1e7;
    const maxOuterMass = 100;
    const minOuterMass = 10;
    const cRadius = (massToRadius(cMass) + massToRadius(maxOuterMass)) / sizeFactor;
    pos.push(...center);
    pos.push(cMass); // mass
    vel.push(...centerV, massToRadius(cMass) / 2);

    totalMass += cMass;
    vec3.add(CoM, vec3.scale(center, cMass), CoM);


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
      const mass = randRange(minOuterMass, maxOuterMass);
      // random radius is proportional sqrt(U) for uniform distribution
      const t = Math.sqrt(Math.random());
      const exp = 2;
      // const r = t ** exp * (radius - cRadius) + cRadius;
      const r = (cRadius + radius * (2 ** (-exp * (t - 1)) - 1) / (2 ** exp - 1)); // t -> i/count
      // random angle
      const theta = randRange(0, 2 * Math.PI);
      // const arms = 5, clockwise = true;
      // const theta = (i * Math.PI * (2 / arms + Math.sign(clockwise - 0.5) * Math.PI / count)) * randRange(1, 1.0001);

      // normal component for thickness
      const wPos = vec3.scale(n, randRange(-.1, .1) * (1 / (10 * (r / radius) ** 2 + 1)));

      // local offset = u*(r*cos theta) + v*(r*sin theta)
      const uPos = vec3.scale(u, Math.sqrt(r * r - vec3.length(wPos) ** 2) * Math.cos(theta));
      const vPos = vec3.scale(v, Math.sqrt(r * r - vec3.length(wPos) ** 2) * Math.sin(theta));

      // world position = center + offset_u + offset_v
      const uvw = vec3.add(vec3.add(center, wPos), vec3.add(uPos, vPos));
      pos.push(...uvw);
      pos.push(mass); // mass
      vec3.add(CoM, vec3.scale(uvw, mass), CoM);
      totalMass += mass;

      // tangent angle
      const tangent = theta + Math.PI / 2;
      const speed = Math.sqrt(G * (cMass) / r);

      const tangentX = speed * Math.cos(tangent);
      const tangentY = speed * Math.sin(tangent);

      const uVel = vec3.scale(u, tangentX);
      const vVel = vec3.scale(v, tangentY);

      vel.push(...vec3.add(centerV, vec3.add(uVel, vVel)), massToRadius(mass));
      // vel.push(0, 0, 0, 0);
    }
    camera.target = defaults.target = vec3.scale(CoM, 1 / totalMass);
  });

  return [new Float32Array(pos), new Float32Array(vel)];
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

  const [bodyData, velData] = generateGalaxy([
    [
      [0, 0, 0],
      // [0, 0, 0],
      [randRange(-5, 5), randRange(-5, 5), randRange(-5, 5)],
      // [1, 1, 1],
      [Math.random(), Math.random(), Math.random()],
      2 * Math.random() + 2,
      20000
    ],
    [
      [randRange(-5, 5), randRange(-5, 5), randRange(-5, 5)],
      [randRange(-5, 5), randRange(-5, 5), randRange(-5, 5)],
      [Math.random(), Math.random(), Math.random()],
      2 * Math.random() + 2,
      20000
    ],
    // [
    //   [randRange(-5, 5), randRange(-5, 5), randRange(-5, 5)],
    //   [randRange(-5, 5), randRange(-5, 5), randRange(-5, 5)],
    //   [Math.random(), Math.random(), Math.random()],
    //   2 * Math.random() + 2,
    //   20000
    // ],
  ]);
  // createPoints({
  //   radius: 1,
  //   numSamples: 1000,
  // });
  // const [bodyData, velData] = [
  //   new Float32Array([0,0,0,1000, 1,1,1,1000, 1,0,0,1000, 0,1,0,1000, 0,0,1,1000, 1,1,0,1000, 1,0,1,1000, 0,1,1,1000]),
  //   new Float32Array([0,0,0,0, 0,.01,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0])
  // ];

  nBodies = bodyData.length / 4;

  const accelData = new Float32Array(bodyData.length); // must use vec4f due to padding

  bodyBuffer = device.createBuffer({
    label: "body buffer",
    size: bodyData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(bodyBuffer, 0, bodyData);

  velBuffer = device.createBuffer({
    label: "velocity buffer",
    size: velData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(velBuffer, 0, velData);

  accelBuffer = device.createBuffer({
    label: "acceleration buffer",
    size: accelData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(accelBuffer, 0, accelData);

  const uniformBuffer = device.createBuffer({
    size: uniformValues.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformStruct = `
    struct Uniforms {
      matrix: mat4x4f,
      cameraPos: vec3f,
      sizeRatio: f32,
      f: f32,
      dt: f32,
      G: f32,
    };
  `;

  const computeModule = device.createShaderModule({
    code: `
      ${uniformStruct}

      // position and mass buffer
      @group(0) @binding(0) var<storage, read_write> bodies: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> vel: array<vec4f>;
      @group(0) @binding(2) var<storage, read_write> accel: array<vec4f>;
      @group(0) @binding(3) var<uniform> uni: Uniforms;

      const TILE_SIZE = ${TILE_SIZE};
      var<workgroup> tile: array<vec4f, TILE_SIZE>;

      // calculate gravitational acceleration between bodies b1 and b2
      fn bodyAccel(b1: vec4f, b2: vec4f) -> vec3f {
        let r = b2.xyz - b1.xyz;
        let distSqr = dot(r, r) + 1e-4; // eps = 1e-2
        let invDistCubed = inverseSqrt(distSqr * distSqr * distSqr);
        return uni.G * b2.w * invDistCubed * r;
      }
        
      // each thread accumulates accelerations for one body
      @compute @workgroup_size(TILE_SIZE)
      fn main(
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u
      ) {
        let bodyIndex = gid.x;
        let nBodies = arrayLength(&bodies);
        let isBody = bodyIndex < nBodies;
        let body = bodies[bodyIndex];

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
            let index2 = t * TILE_SIZE + i;
            if (index2 <= nBodies && index2 != bodyIndex) {
              newAccel += bodyAccel(body, tile[i]);
            }
          }

          // sync before loading next tile
          workgroupBarrier();
        }
        
        
        // if (bodyIndex < nBodies) {
        let vec4accel = vec4f(newAccel, 0);

        // velocity verlet with frame shift
        vel[bodyIndex] += 0.5 * (accel[bodyIndex] + vec4accel) * uni.dt;
        bodies[bodyIndex] += uni.dt * (vel[bodyIndex] + 0.5 * vec4accel * uni.dt); // use old accel?

        // euler
        // vel[bodyIndex] += vec4accel * uni.dt;
        // bodies[bodyIndex] += vel[bodyIndex] * uni.dt;

        // save acceleration for the next time step
        accel[bodyIndex] = vec4accel;
        // }
      }
    `,
    label: "compute module"
  });

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'main' },
    label: "compute pipeline"
  });

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bodyBuffer } },
      { binding: 1, resource: { buffer: velBuffer } },
      { binding: 2, resource: { buffer: accelBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
    label: "compute bind group"
  });

  const renderModule = device.createShaderModule({
    code: `
      ${uniformStruct}

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
        @location(1) @interpolate(flat) index: u32,
      };

      @group(0) @binding(0) var<storage, read> bodies: array<vec4f>;
      @group(0) @binding(1) var<storage, read> vel: array<vec4f>;
      @group(0) @binding(2) var<uniform> uni: Uniforms;

      
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
        let radius = vel[instNdx].w; //pow(body.w / 4.189, 1.0/3.0);
        
        let worldPos = body.xyz;

        // compute view vector from camera to particle
        let viewVec = worldPos - uni.cameraPos;
        let viewDir = normalize(viewVec);

        // compute billboard basis
        let right = normalize(cross(viewDir, vec3f(0.0, 1.0, 0.0)));
        let up = normalize(cross(right, viewDir));

        // offset in local camera-facing plane
        let worldOffset = (right * uv.x + up * uv.y) * max(radius, 2 * length(viewVec) / uni.f) * uni.sizeRatio;
        // let worldOffset = (right * uv.x + up * uv.y) * radius * uni.sizeRatio;

        // final world position of the quad vertex
        let cornerPos = worldPos + worldOffset;

        var vsOut: VSOutput;
        vsOut.position = uni.matrix * vec4f(cornerPos, 1.0);
        vsOut.uv = uv;
        vsOut.index = instNdx;
        return vsOut;
      }

      fn colorMap(value: f32) -> vec3f {
        return vec3<f32>(value, 1.0 - abs(value - 0.5), 1.0 - value); // rgb
      }

      @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        // circle SDF
        let dist = length(vsOut.uv) - 1;
        if (dist > 0.0) { discard; }
        // if (dist > -0.5 / vel[vsOut.index].w) { return vec4f(0); } // black border around circles
        // return vec4f(colorMap(vsOut.position.w), 1); // color by distance to camera

        return vec4f(colorMap(length(vel[vsOut.index].xyz) / 40f), 1); // color by magnitude
        // return vec4f(normalize(vel[vsOut.index].xyz) / 2 + vec3f(0.5), 1); // color by direction
      }
    `,
    label: "render module"
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

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0), // renderBindGroupLayout, 
    entries: [ // swith accel/vel buffer for different visualization
      { binding: 0, resource: { buffer: bodyBuffer } },
      { binding: 1, resource: { buffer: velBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
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
  const filterStrength = 10;

  function render() {
    const startTime = performance.now();
    deltaTime += (startTime - lastFrameTime - deltaTime) / filterStrength;
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    if (keyOrbit) camera.orbit((orbleft - orbright) * KEY_ROT_SPEED, (orbup - orbdown) * KEY_ROT_SPEED);
    if (keyPan) camera.pan((panleft - panright) * KEY_PAN_SPEED, (panup - pandown) * KEY_PAN_SPEED);
    if (keyZoom) camera.zoom((zoomout - zoomin) * KEY_ZOOM_SPEED);
    if (keyFOV) camera.adjFOV((zoomout - zoomin) * KEY_FOV_SPEED);
    if (keyFOVWithoutZoom) camera.adjFOVWithoutZoom((zoomout - zoomin) * KEY_FOV_SPEED);

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

    uni.GValue.set([G]);
    uni.dtValue.set([dt]);

    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(nBodies / TILE_SIZE));
    computePass.end();

    const renderPass = encoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, bodyBuffer);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(6, nBodies);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  intId = setInterval(() => {
    ui.fps.textContent = fps.toFixed(1);
    ui.jsTime.textContent = jsTime.toFixed(2);
    ui.frameTime.textContent = deltaTime.toFixed(1);
    // ui.gpuTime.textContent = deltaTime - jsTime;
  }, 100);

  camera.updatePosition();
  rafId = requestAnimationFrame(render);

}

const camera = new Camera(defaults);

main();
