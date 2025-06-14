const ROT_SPEED = 0.005;
const KEY_ROT_MULT = 5;
const PAN_SPEED = 0.001;

const ZOOM_SPEED = 0.0005;
const FOV_SPEED = 0.0002;

const minFOV = (10).toRad(), maxFOV = (100).toRad();

const matrix = mat4.create();
let fVal;
let aspect;

const defaults = {
  target: vec3.fromValues(0, 0, 0),
  radius: 2,
  position: vec3.create(),
  azimuth: 0,
  elevation: 0,
  fov: (60).toRad(),
  near: 0.1,
  far: 1000,
}

// camera state
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
};

function updateMatrix() {
  aspect = canvas.clientWidth / canvas.clientHeight;
  const proj = mat4.perspective(camera.fov, aspect, camera.near, camera.far);
  const view = mat4.lookAt(camera.position, camera.target, camera.worldUp);
  mat4.multiply(proj, view, matrix);
  // Set the f value in the uniform values
  fVal = proj[5];
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
    if (e.ctrlKey) {
      camera.fov = defaults.fov;
    } else if (e.altKey) {
      camera.radius = defaults.radius;
      camera.fov = defaults.fov;
    } else {
      resetCam();
    }
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
  if (state.orbitActive) {
    orbitCam(dx, dy);
  }

  // Pan within view-plane
  if (state.panActive) {
    // pan delta = (-right * dx + upReal * dy) * PAN_SPEED
    const adjustedPanSpeed = PAN_SPEED * camera.radius * camera.fov;
    let pan = vec3.scaleAndAdd(
      vec3.scale(camera.viewRight(), -dx * adjustedPanSpeed),
      camera.viewUp(),
      dy * adjustedPanSpeed
    );

    // apply to target and camera position
    camera.target = vec3.add(camera.target, pan);
    camera.position = vec3.add(camera.position, pan);
    updateMatrix();
  }
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();

  if (e.altKey) {
    // adjust FOV without zoom
    const initial = Math.tan(camera.fov / 2) * camera.radius;
    camera.fov = (camera.fov + e.deltaY * FOV_SPEED).clamp(minFOV, maxFOV);
    camera.radius = initial / Math.tan(camera.fov / 2);
  } else if (e.ctrlKey) {
    // FOV zoom only
    camera.fov = (camera.fov + e.deltaY * FOV_SPEED).clamp(minFOV, maxFOV);
  } else {
    // Zoom only - move camera in/out
    camera.radius = (camera.radius + e.deltaY * ZOOM_SPEED * (camera.radius)).clamp(camera.near, camera.far);
  }
  updateCameraPosition();
}, { passive: false });


function orbitCam(dx, dy) {
  camera.azimuth -= dx * ROT_SPEED;
  camera.elevation += dy * ROT_SPEED;
  // clamp elevation to [-89, +89] deg
  const limit = Math.PI / 2 - 0.01;
  camera.elevation = (camera.elevation).clamp(-limit, limit);
  updateCameraPosition();
}

function resetCam() {
  camera.target = defaults.target;
  camera.radius = defaults.radius;
  camera.azimuth = defaults.azimuth;
  camera.elevation = defaults.elevation;
  camera.fov = defaults.fov;
}

let up = 0, left = 0;
window.addEventListener("keydown", (e) => {
  switch (e.key) {
    case "Alt":
      e.preventDefault();
      break;
    case "Home":
      resetCam();
      updateCameraPosition();
      break;
  }
  
  if (e.key == "ArrowUp") up = 1;
  else if (e.key == "ArrowDown") up = -1;
  if (e.key == "ArrowLeft") left = 1;
  else if (e.key == "ArrowRight") left = -1;
  orbitCam(left * KEY_ROT_MULT, up * KEY_ROT_MULT);
});
window.addEventListener("keyup", (e) => {
  if (e.key == "ArrowUp" || e.key == "ArrowDown") up = 0;
  if (e.key == "ArrowLeft" || e.key == "ArrowRight") left = 0;
});