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

  // compute cross product of a and b, store in out
  static cross(a, b, out = this.create()) {
    const ax = a[0], ay = a[1], az = a[2];
    const bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  }

  // normalize vector a, store in out
  static normalize(a, out = this.create()) {
    const x = a[0], y = a[1], z = a[2];
    let len = x*x + y*y + z*z;
    if (len > 0) {
      len = 1 / Math.sqrt(len);
      out[0] = x * len;
      out[1] = y * len;
      out[2] = z * len;
    }
    return out;
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
  static perspective(fovy, aspect, near, far, out = mat4.create()) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0]  = f / aspect;
    out[1]  = 0;
    out[2]  = 0;
    out[3]  = 0;

    out[4]  = 0;
    out[5]  = f;
    out[6]  = 0;
    out[7]  = 0;

    out[8]  = 0;
    out[9]  = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
  }

  // generate lookAt view matrix
  static lookAt(eye, center, up, out = mat4.create()) {
    const x0 = eye[0], x1 = eye[1], x2 = eye[2];
    const y0 = center[0], y1 = center[1], y2 = center[2];
    const z0 = up[0],  z1 = up[1],  z2 = up[2];

    let fx = x0 - y0;
    let fy = x1 - y1;
    let fz = x2 - y2;
    // normalize f
    let rlen = 1 / Math.hypot(fx, fy, fz);
    fx *= rlen;
    fy *= rlen;
    fz *= rlen;

    // compute s = up × f
    let sx = z1 * fz - z2 * fy;
    let sy = z2 * fx - z0 * fz;
    let sz = z0 * fy - z1 * fx;
    // normalize s
    rlen = 1 / Math.hypot(sx, sy, sz);
    sx *= rlen;
    sy *= rlen;
    sz *= rlen;

    // compute u = f × s
    let ux = fy * sz - fz * sy;
    let uy = fz * sx - fx * sz;
    let uz = fx * sy - fy * sx;

    out[0] = sx;
    out[1] = ux;
    out[2] = fx;
    out[3] = 0;

    out[4] = sy;
    out[5] = uy;
    out[6] = fy;
    out[7] = 0;

    out[8] = sz;
    out[9] = uz;
    out[10]= fz;
    out[11]= 0;

    out[12]= -(sx * x0 + sy * x1 + sz * x2);
    out[13]= -(ux * x0 + uy * x1 + uz * x2);
    out[14]= -(fx * x0 + fy * x1 + fz * x2);
    out[15]= 1;

    return out;
  }

  // multiply two 4x4 matrices: out = a * b
  static multiply(a, b, out = mat4.create()) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    for (let i = 0; i < 4; ++i) {
      const ai0 = a[i], ai1 = a[i+4], ai2 = a[i+8], ai3 = a[i+12];
      out[i]    = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i+4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i+8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i+12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }

  // rotate matrix a by rad around X axis, store in out
  static rotateX(rad, a, out = mat4.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    // copy a into out
    out.set(a);
    
    // rows 1 and 2 transform
    out[4] = a[4] * c + a[8] * s;
    out[5] = a[5] * c + a[9] * s;
    out[6] = a[6] * c + a[10]* s;
    out[7] = a[7] * c + a[11]* s;

    out[8] = a[8] * c - a[4] * s;
    out[9] = a[9] * c - a[5] * s;
    out[10]= a[10]* c - a[6] * s;
    out[11]= a[11]* c - a[7] * s;
    return out;
  }

  // rotate matrix a by rad around Y axis, store in out
  static rotateY(rad, a, out = mat4.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    out.set(a);

    // rows 0 and 2 transform
    out[0] = a[0] * c - a[8]  * s;
    out[1] = a[1] * c - a[9]  * s;
    out[2] = a[2] * c - a[10] * s;
    out[3] = a[3] * c - a[11] * s;

    out[8] = a[0] * s + a[8]  * c;
    out[9] = a[1] * s + a[9]  * c;
    out[10]= a[2] * s + a[10] * c;
    out[11]= a[3] * s + a[11] * c;
    return out;
  }
}
