const ROT_SPEED = 0.005;
const PAN_SPEED = 0.001;
const ZOOM_SPEED = 0.0005;
const FOV_SPEED = 0.0002;

const KEY_ROT_SPEED = 3;
const KEY_PAN_SPEED = 5;
const KEY_ZOOM_SPEED = 0.01;
const KEY_FOV_SPEED = 0.005;

const minFOV = (10).toRad(), maxFOV = (100).toRad();

const defaults = {
  target: vec3.fromValues(0, 0, 0),
  radius: 5,
  position: vec3.create(),
  azimuth: 0,
  elevation: 0,
  fov: (60).toRad(),
  near: 0.1,
  far: 1e5,
}

// camera state and interaction
const camera = {
  // spherical coords around target:
  target: defaults.target,
  radius: defaults.radius,
  position: defaults.position,
  azimuth: defaults.azimuth,               // horizontal angle, radians
  elevation: defaults.elevation,             // vertical angle, radians

  // for projection:
  fov: defaults.fov,
  near: defaults.near,
  far: defaults.far,
  worldUp: vec3.fromValues(0, 1, 0),
  viewDir: () => vec3.normalize(vec3.subtract(camera.target, camera.position)),
  viewRight: () => vec3.normalize(vec3.cross(camera.viewDir(), camera.worldUp)),
  viewUp: () => vec3.normalize(vec3.cross(camera.viewRight(), camera.viewDir())),

  // interaction:
  orbit: (dx, dy) => {
    camera.azimuth -= dx * ROT_SPEED;
    camera.elevation += dy * ROT_SPEED;
    // clamp elevation to [-89, +89] deg
    const limit = Math.PI / 2 - 0.01;
    camera.elevation = (camera.elevation).clamp(-limit, limit);
    updateCameraPosition();
  },
  pan: (dx, dy) => {
    // pan delta = (-right * dx + upReal * dy) * PAN_SPEED
    const adjustedPanSpeed = PAN_SPEED * camera.radius * camera.fov;
    const pan = vec3.scaleAndAdd(
      vec3.scale(camera.viewRight(), -dx * adjustedPanSpeed),
      camera.viewUp(),
      dy * adjustedPanSpeed
    );
    camera.target = vec3.add(camera.target, pan);
    camera.position = vec3.add(camera.position, pan);
    updateMatrix();
  },
  zoom: (delta) => {
    camera.radius = ((delta + 1) * camera.radius).clamp(camera.near, camera.far);
    updateCameraPosition();
  },
  adjFOV: (delta) => {
    camera.fov = (camera.fov + delta).clamp(minFOV, maxFOV);
    updateCameraPosition();
  },
  adjFOVWithoutZoom: (delta) => {
    const initial = Math.tan(camera.fov / 2) * camera.radius;
    camera.fov = (camera.fov + delta).clamp(minFOV, maxFOV);
    camera.radius = initial / Math.tan(camera.fov / 2);
    updateCameraPosition();
  },
  reset: (e = { altKey: false, ctrlKey: false }) => {
    camera.fov = defaults.fov;
    if (!e.ctrlKey) camera.radius = defaults.radius;
    if (!e.altKey && !e.ctrlKey) {
      camera.azimuth = defaults.azimuth;
      camera.elevation = defaults.elevation;
      camera.target = defaults.target;
    }
    updateCameraPosition();
  }
};

function updateMatrix() {
  aspect = canvas.clientWidth / canvas.clientHeight;
  const proj = mat4.perspective(camera.fov, aspect, camera.near, camera.far);
  const view = mat4.lookAt(camera.position, camera.target, camera.worldUp);
  mat4.multiply(proj, view, uni.matrixValue);

  uni.fValue.set([proj[5]]);
  uni.cameraPosValue.set(camera.position);
}

// compute position from spherical coords
function updateCameraPosition() {
  const { azimuth: phi, elevation: theta, radius: r, target: T } = camera;
  const x = r * Math.cos(theta) * Math.sin(phi);
  const y = r * Math.sin(theta);
  const z = r * Math.cos(theta) * Math.cos(phi);
  camera.position = vec3.fromValues(T[0] + x, T[1] + y, T[2] + z);
  updateMatrix();
}

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
  if (e.button === 1) {
    camera.reset(e);
    updateCameraPosition();
  }
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
  if (state.orbitActive) camera.orbit(dx, dy);

  // Pan within view-plane
  if (state.panActive) camera.pan(dx, dy);
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();

  if (e.altKey) {
    // adjust FOV without zoom
    camera.adjFOVWithoutZoom(e.deltaY * FOV_SPEED)
  } else if (e.ctrlKey) {
    // FOV zoom only
    camera.adjFOV(e.deltaY * FOV_SPEED);
  } else {
    // Zoom only - move camera in/out
    camera.zoom(e.deltaY * ZOOM_SPEED);
  }
}, { passive: false });

let orbup = false, orbdown = false, orbleft = false, orbright = false;
let panup = false, pandown = false, panleft = false, panright = false;
let zoomin = false, zoomout = false;
let keyOrbit = false, keyPan = false, keyZoom = false, keyFOV = false, keyFOVWithoutZoom = false;

function keyCamera(e, val) {
  if (["w", "a", "s", "d", "f", "c"].includes(e.key) || e.key.includes("Arrow"))
    e.preventDefault();

  switch (e.key) {
    case "ArrowUp":
      orbup = val;
      break;
    case "w":
      panup = val;
      break;
    case "ArrowDown":
      orbdown = val;
      break;
    case "s":
      pandown = val;
      break;
    case "ArrowLeft":
      orbleft = val;
      break;
    case "a":
      panleft = val;
      break;
    case "ArrowRight":
      orbright = val;
      break;
    case "d":
      panright = val;
      break;
    case "f":
      zoomin = val;
      break;
    case "c":
      zoomout = val;
      break;
  }

  const zoom = zoomin || zoomout;

  keyOrbit = orbup || orbdown || orbleft || orbright;
  keyPan = panup || pandown || panleft || panright;
  keyZoom = !(e.ctrlKey || e.altKey) && zoom;
  keyFOV = e.ctrlKey && zoom;
  keyFOVWithoutZoom = e.altKey && zoom;
}
window.addEventListener("keydown", (e) => {
  console.log(e.key);
  switch (e.key) {
    case "Alt":
      e.preventDefault();
      break;
    case " ":
      camera.reset(e);
      break;
  }

  keyCamera(e, true);
});
window.addEventListener("keyup", (e) => {
  keyCamera(e, false);
});