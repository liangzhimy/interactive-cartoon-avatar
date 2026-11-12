/**
 * Damped harmonic spring solver for organic cartoon secondary motion
 * (head rotation smoothing, bouncy hair follow-through, and landing recovery).
 */
export class SpringValue {
  public value: number;
  public target: number;
  public velocity = 0;
  public stiffness: number;
  public damping: number;

  constructor(initial = 0, stiffness = 170, damping = 18) {
    this.value = initial;
    this.target = initial;
    this.stiffness = stiffness;
    this.damping = damping;
  }

  public impulse(dv: number): void {
    this.velocity += dv;
  }

  public snap(v: number): void {
    this.value = v;
    this.target = v;
    this.velocity = 0;
  }

  public step(dt: number): number {
    // Sub-step clamp for numerical stability across tab switches
    const clampedDt = Math.min(dt, 0.05);
    const steps = clampedDt > 0.02 ? 2 : 1;
    const h = clampedDt / steps;

    for (let i = 0; i < steps; i++) {
      const force = (this.target - this.value) * this.stiffness;
      const drag = -this.velocity * this.damping;
      const accel = force + drag;
      this.velocity += accel * h;
      this.value += this.velocity * h;
    }

    if (Math.abs(this.target - this.value) < 1e-4 && Math.abs(this.velocity) < 1e-4) {
      this.value = this.target;
      this.velocity = 0;
    }
    return this.value;
  }
}

export class Spring2D {
  public x: SpringValue;
  public y: SpringValue;

  constructor(ix = 0, iy = 0, stiffness = 150, damping = 16) {
    this.x = new SpringValue(ix, stiffness, damping);
    this.y = new SpringValue(iy, stiffness, damping);
  }

  public setTarget(tx: number, ty: number): void {
    this.x.target = tx;
    this.y.target = ty;
  }

  public impulse(vx: number, vy: number): void {
    this.x.impulse(vx);
    this.y.impulse(vy);
  }

  public step(dt: number): { x: number; y: number } {
    return {
      x: this.x.step(dt),
      y: this.y.step(dt),
    };
  }
}
