const uiIDs = [
  "controls",
  "toggleSettings",
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
  "bufferInput"
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

window.onresize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  camera.updateMatrix();
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



/*
 * vec3 and mat4 utility classes
 * Provides basic vector and matrix operations for 3D graphics
 */

class vec3 {
  // create a zero vector
  static create() {
    return new Float32Array(3);
  }

  // create a vector from values
  static fromValues(x, y, z) {
    const out = new Float32Array(3);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  }

  // out = a + b
  static add(a, b, out = this.create()) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
  }

  // out = a - b
  static subtract(a, b, out = this.create()) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  }

  // out = a * scalar
  static scale(a, scalar, out = this.create()) {
    out[0] = a[0] * scalar;
    out[1] = a[1] * scalar;
    out[2] = a[2] * scalar;
    return out;
  }

  // out = a + (b * scalar)
  static scaleAndAdd(a, b, scalar, out = this.create()) {
    out[0] = a[0] + b[0] * scalar;
    out[1] = a[1] + b[1] * scalar;
    out[2] = a[2] + b[2] * scalar;
    return out;
  }

  // compute dot product of a and b
  static dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // compute cross product of a and b, store in out
  static cross(a, b, out = this.create()) {
    const ax = a[0], ay = a[1], az = a[2];
    const bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  }

  static length(a) {
    return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  }

  // normalize vector a, store in out
  static normalize(a, out = this.create()) {
    let len = this.length(a);
    if (len > 0) {
      out = this.scale(a, 1 / len);
    }
    return out;
  }

  static clone(a, out = this.create()) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    return out;
  }

  static toString(a) {
    return Array.from(a).map((i) => parseFloat(i).toFixed(2));
  }
}

class mat4 {
  // create identity matrix
  static create() {
    const out = new Float32Array(16);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return out;
  }

  // generate perspective projection matrix
  static perspective(fovy, aspect, near, far, out = this.create()) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
  }

  // generate lookAt view matrix
  static lookAt(eye, center, up, out = this.create()) {
    // z along view direction
    let z = vec3.normalize(vec3.subtract(eye, center));

    // x = ||up x z||
    let x = vec3.normalize(vec3.cross(up, z));

    // y = ||z x x||
    let y = vec3.cross(z, x);

    out[0] = x[0];
    out[1] = y[0];
    out[2] = z[0];
    out[3] = 0;

    out[4] = x[1];
    out[5] = y[1];
    out[6] = z[1];
    out[7] = 0;

    out[8] = x[2];
    out[9] = y[2];
    out[10] = z[2];
    out[11] = 0;

    out[12] = -vec3.dot(x, eye);
    out[13] = -vec3.dot(y, eye);
    out[14] = -vec3.dot(z, eye);
    out[15] = 1;

    return out;
  }

  // multiply two 4x4 matrices: out = a * b
  static multiply(a, b, out = this.create()) {
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i] = ai0 * b[0] + ai1 * b[1] + ai2 * b[2] + ai3 * b[3];
      out[i + 4] = ai0 * b[4] + ai1 * b[5] + ai2 * b[6] + ai3 * b[7];
      out[i + 8] = ai0 * b[8] + ai1 * b[9] + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }

  // rotate matrix a by rad around X axis, store in out
  static rotateX(rad, a, out = this.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    // copy a into out
    out.set(a);

    // rows 1 and 2 transform
    out[4] = a[4] * c + a[8] * s;
    out[5] = a[5] * c + a[9] * s;
    out[6] = a[6] * c + a[10] * s;
    out[7] = a[7] * c + a[11] * s;

    out[8] = a[8] * c - a[4] * s;
    out[9] = a[9] * c - a[5] * s;
    out[10] = a[10] * c - a[6] * s;
    out[11] = a[11] * c - a[7] * s;
    return out;
  }

  // rotate matrix a by rad around Y axis, store in out
  static rotateY(rad, a, out = this.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    out.set(a);

    // rows 0 and 2 transform
    out[0] = a[0] * c - a[8] * s;
    out[1] = a[1] * c - a[9] * s;
    out[2] = a[2] * c - a[10] * s;
    out[3] = a[3] * c - a[11] * s;

    out[8] = a[0] * s + a[8] * c;
    out[9] = a[1] * s + a[9] * c;
    out[10] = a[2] * s + a[10] * c;
    out[11] = a[3] * s + a[11] * c;
    return out;
  }
}


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