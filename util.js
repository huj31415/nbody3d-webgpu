
Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };
Number.prototype.toRad = function () { return this * Math.PI / 180; }
Number.prototype.toDeg = function () { return this / Math.PI * 180; }


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
