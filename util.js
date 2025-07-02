const uiIDs = [
  "controls",
  "toggleSettings",
  "res",
  "dt",
  "dtValue",
  "G",
  "GValue",
  "jsTime",
  "frameTime",
  "fps",
  "computeTime",
  "renderTime",
  "nBodies",
  "camFOV",
  "camDist",
  "camTarget",
  "camPos",
  "camAlt",
  "camAz",
  "toggleSim",
  "restartSim",
  "export",
  "import",
  "bufferInput",
  "minBodies",
  "maxBodies",
  "numGalaxies",
];

const ui = {};

uiIDs.forEach((id) => ui[id] = document.getElementById(id));
Object.freeze(ui);

let oldDt;

ui.dt.addEventListener("input", (event) => {
  const val = parseFloat(event.target.value);
  ui.dtValue.textContent = val.toFixed(2);
  const newDt = 10 ** val;
  if (oldDt) oldDt = newDt;
  else dt = newDt;

  uni.dtValue.set([dt]);
});

ui.G.addEventListener("input", (event) => {
  const val = parseFloat(event.target.value);
  ui.GValue.textContent = val.toFixed(2);
  G = 10 ** val;

  uni.GValue.set([G]);
});

ui.toggleSim.addEventListener("click", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
  } else {
    oldDt = dt;
    dt = 0;
  }
});

// requestAnimationFrame id, fps update id
let rafId, intId;

ui.restartSim.addEventListener("click", () => {
  cancelAnimationFrame(rafId);
  clearInterval(intId);
  main();
});

ui.toggleSettings.addEventListener("click", () => {
  ui.toggleSettings.innerText = ui.toggleSettings.innerText === ">" ? "<" : ">";
  if (ui.controls.classList.contains("hidden")) {
    ui.controls.classList.remove("hidden");
    ui.toggleSettings.classList.remove("inactive");
  } else {
    ui.controls.classList.add("hidden");
    ui.toggleSettings.classList.add("inactive");
  }
});

// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0, computeTime = 0, renderTime = 0;

window.onresize = window.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  camera.updateMatrix();
  ui.res.textContent = [window.innerWidth, window.innerHeight];
};

/**
 * Clamps a number between between specified values
 * @param {Number} min Lower bound to clamp
 * @param {Number} max Upper bound to clamp
 * @returns Original number clamped between min and max
 */
Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };

/**
 * Converts degrees to radians
 * @returns Degree value in radians
 */
Number.prototype.toRad = function () { return this * Math.PI / 180; }

/**
 * Converts radians to degrees
 * @returns Radian value in degrees
 */
Number.prototype.toDeg = function () { return this / Math.PI * 180; }

/**
 * Converts mass to radius at density 1
 * @param {Number} mass Mass of body
 * @returns Radius of the body
 */
const massToRadius = (mass) => Math.cbrt(mass / (4 / 3 * Math.PI));

/**
 * Generates a random number within a range
 * @param {Number} min Lower bound, inclusive
 * @param {Number} max Upper bound, exclusive
 * @returns Random number between [min, max)
 */
const randRange = (min, max) => Math.random() * (max - min) + min;
const randMax = (max) => Math.random() * max;


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
 * Exports simulation and camera state to json
 * @param {device} device GPU device
 * @param {Array<GPUBuffer>} buffers Array containing bodyBuffer, velBuffer, accelBuffer to export
 * @param {Camera} camera Camera to export settings
 */
async function exportSimulation(device, buffers, camera) {
  const [bodyBuffer, velBuffer, accelBuffer] = buffers;

  async function readBuffer(buffer) {
    const size = buffer.size ?? buffer.getMappedRange().byteLength;
    const readBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copy = readBuffer.getMappedRange().slice(0); // clone
    readBuffer.unmap();
    return new Float32Array(copy);
  }

  const [bodies, velocities, accelerations] = await Promise.all([
    readBuffer(bodyBuffer),
    readBuffer(velBuffer),
    readBuffer(accelBuffer),
  ]);

  const exportData = {
    bodies: Array.from(bodies),
    vel: Array.from(velocities),
    accel: Array.from(accelerations),
    camera: {
      target: Array.from(camera.target),
      position: Array.from(camera.position),
      radius: camera.radius,
      azimuth: camera.azimuth,
      elevation: camera.elevation,
      fov: camera.fov,
      near: camera.near,
      far: camera.far,
    },
    G: parseFloat(ui.G.value).toFixed(2)
  };

  const blob = new Blob([JSON.stringify(exportData)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "simulation_export.json";
  a.click();
}

/**
 * Imports simulation data from json to buffers
 * @param {GPUDevice} device GPU device
 * @param {File} file JSON file to read data from
 * @param {Array<GPUBuffer>} bufferArray Array containing bodyBuffer, velBuffer, accelBuffer to import to
 * @param {Camera} camera Camera to import settings to
 */
async function importSimulation(device, file, bufferArray, camera) {
  if (!file) return;
  const text = await file.text();
  const json = JSON.parse(text);

  const [bodyBuffer, velBuffer, accelBuffer] = bufferArray;

  const buffers = [
    { name: "bodies", gpu: bodyBuffer },
    { name: "vel", gpu: velBuffer },
    { name: "accel", gpu: accelBuffer },
  ];

  for (const { name, gpu } of buffers) {
    const data = new Float32Array(json[name]);
    const upload = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });

    new Float32Array(upload.getMappedRange()).set(data);
    upload.unmap();

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(upload, 0, gpu, 0, data.byteLength);
    device.queue.submit([encoder.finish()]);
  }

  // Apply camera state
  if (json.camera) {
    camera.target = vec3.fromValues(...json.camera.target);
    camera.position = vec3.fromValues(...json.camera.position);
    camera.radius = json.camera.radius;
    camera.azimuth = json.camera.azimuth;
    camera.elevation = json.camera.elevation;
    camera.fov = json.camera.fov;
    camera.near = json.camera.near;
    camera.far = json.camera.far;

    camera.updatePosition();
  }
  if (json.G) {
    ui.GValue.textContent = ui.G.value = json.G;
    G = 10 ** json.G;
  }
}

ui.export.addEventListener("click", () => exportSimulation(device, [bodyBuffer, velBuffer, accelBuffer], camera));

ui.import.addEventListener("click", () => {
  const file = ui.bufferInput.files[0];
  if (file) {
    importSimulation(device, file, [bodyBuffer, velBuffer, accelBuffer], camera);
    ui.bufferInput.value = null;
    camera.reset();
  }
});



function assert(cond, msg = '') {
  if (!cond) {
    throw new Error(msg);
  }
}

// We track command buffers so we can generate an error if
// we try to read the result before the command buffer has been executed.
const s_unsubmittedCommandBuffer = new Set();

/* global GPUQueue */
GPUQueue.prototype.submit = (function (origFn) {
  return function (commandBuffers) {
    origFn.call(this, commandBuffers);
    commandBuffers.forEach(cb => s_unsubmittedCommandBuffer.delete(cb));
  };
})(GPUQueue.prototype.submit);

// See https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
class TimingHelper {
  #canTimestamp;
  #device;
  #querySet;
  #resolveBuffer;
  #resultBuffer;
  #commandBuffer;
  #resultBuffers = [];
  // state can be 'free', 'need resolve', 'wait for result'
  #state = 'free';

  constructor(device) {
    this.#device = device;
    this.#canTimestamp = device.features.has('timestamp-query');
    if (this.#canTimestamp) {
      this.#querySet = device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      this.#resolveBuffer = device.createBuffer({
        size: this.#querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  #beginTimestampPass(encoder, fnName, descriptor) {
    if (this.#canTimestamp) {
      assert(this.#state === 'free', 'state not free');
      this.#state = 'need resolve';

      const pass = encoder[fnName]({
        ...descriptor,
        ...{
          timestampWrites: {
            querySet: this.#querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          },
        },
      });

      const resolve = () => this.#resolveTiming(encoder);
      const trackCommandBuffer = (cb) => this.#trackCommandBuffer(cb);
      pass.end = (function (origFn) {
        return function () {
          origFn.call(this);
          resolve();
        };
      })(pass.end);

      encoder.finish = (function (origFn) {
        return function () {
          const cb = origFn.call(this);
          trackCommandBuffer(cb);
          return cb;
        };
      })(encoder.finish);

      return pass;
    } else {
      return encoder[fnName](descriptor);
    }
  }

  beginRenderPass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, 'beginRenderPass', descriptor);
  }

  beginComputePass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, 'beginComputePass', descriptor);
  }

  #trackCommandBuffer(cb) {
    if (!this.#canTimestamp) {
      return;
    }
    assert(this.#state === 'need finish', 'you must call encoder.finish');
    this.#commandBuffer = cb;
    s_unsubmittedCommandBuffer.add(cb);
    this.#state = 'wait for result';
  }

  #resolveTiming(encoder) {
    if (!this.#canTimestamp) {
      return;
    }
    assert(
      this.#state === 'need resolve',
      'you must use timerHelper.beginComputePass or timerHelper.beginRenderPass',
    );
    this.#state = 'need finish';

    this.#resultBuffer = this.#resultBuffers.pop() || this.#device.createBuffer({
      size: this.#resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.resolveQuerySet(this.#querySet, 0, this.#querySet.count, this.#resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#resultBuffer, 0, this.#resultBuffer.size);
  }

  async getResult() {
    if (!this.#canTimestamp) {
      return 0;
    }
    assert(
      this.#state === 'wait for result',
      'you must call encoder.finish and submit the command buffer before you can read the result',
    );
    assert(!!this.#commandBuffer); // internal check
    assert(
      !s_unsubmittedCommandBuffer.has(this.#commandBuffer),
      'you must submit the command buffer before you can read the result',
    );
    this.#commandBuffer = undefined;
    this.#state = 'free';

    const resultBuffer = this.#resultBuffer;
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(resultBuffer.getMappedRange());
    const duration = Number(times[1] - times[0]);
    resultBuffer.unmap();
    this.#resultBuffers.push(resultBuffer);
    return duration;
  }
}