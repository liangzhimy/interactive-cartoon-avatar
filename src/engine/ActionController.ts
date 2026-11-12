import { SpringValue } from './SpringPhysics';
import type {
  AvatarParameters,
  CheerOptions,
  JumpOptions,
  KickOptions,
  LaughOptions,
  LimbJointAngles,
  LimbPoseInput,
  NodOptions,
  ShakeOptions,
  SmileOptions,
  VisemeWeights,
  WalkOptions,
  WaveOptions,
} from './types';

interface ActiveTask {
  id: string;
  elapsed: number;
  duration: number;
  resolve: () => void;
  evaluate: (tNorm: number, elapsed: number) => void;
  onFinish?: () => void;
}

class LimbSprings {
  private s0 = new SpringValue(0, 190, 17);
  private s1 = new SpringValue(0, 195, 17);
  private s2 = new SpringValue(0, 210, 18);

  public setTarget(a0: number, a1: number, a2: number): void {
    this.s0.target = a0;
    this.s1.target = a1;
    this.s2.target = a2;
  }

  public step(dt: number, out: LimbJointAngles): void {
    out[0] = this.s0.step(dt);
    out[1] = this.s1.step(dt);
    out[2] = this.s2.step(dt);
  }
}

function smoothHoldEnvelope(t: number, riseEnd = 0.2, fallStart = 0.78): number {
  if (t <= 0 || t >= 1) return 0;
  if (t < riseEnd) {
    const u = t / riseEnd;
    return u * u * (3 - 2 * u);
  }
  if (t > fallStart) {
    const u = (1 - t) / Math.max(0.01, 1 - fallStart);
    return u * u * (3 - 2 * u);
  }
  return 1;
}

/**
 * Procedural Cartoon Action & Expression Mixer:
 * Combines base manual/mouse pose, spring-smoothed facial expressions,
 * 4-limb skeletal gestures (`wave`, `cheer`, `walk`, `kick`, `jump`),
 * idle breathing/random blinks, and real-time speech lip-sync parameters.
 */
export class ActionController {
  // Base pose springs (controlled by turnHead / lookAt / sliders)
  private baseYaw = new SpringValue(0, 140, 17);
  private basePitch = new SpringValue(0, 140, 17);
  private baseRoll = new SpringValue(0, 130, 15);

  // Expression & whole-body springs
  private smileSpring = new SpringValue(0, 150, 18);
  private laughSpring = new SpringValue(0, 170, 16);
  private eyeSmileSpring = new SpringValue(0, 160, 18);
  private squashSpring = new SpringValue(0, 240, 17);
  private armSpring = new SpringValue(0, 180, 15);

  // 4-limb skeletal springs
  private leftArmSprings = new LimbSprings();
  private rightArmSprings = new LimbSprings();
  private leftLegSprings = new LimbSprings();
  private rightLegSprings = new LimbSprings();

  // Persistent programmatic limb pose (set via `setLimbPose`)
  private baseLeftArm: LimbJointAngles = [0, 0, 0];
  private baseRightArm: LimbJointAngles = [0, 0, 0];
  private baseLeftLeg: LimbJointAngles = [0, 0, 0];
  private baseRightLeg: LimbJointAngles = [0, 0, 0];

  // Per-frame additive gesture offsets
  private gestureYaw = 0;
  private gesturePitch = 0;
  private gestureRoll = 0;
  private gestureBlink = 0;
  private gestureJumpY = 0;
  private gestureSquash = 0;
  private gestureArm = 0;
  private gestureLaughBounce = 0;
  private gestureBodySwayX = 0;
  private gLeftArm: LimbJointAngles = [0, 0, 0];
  private gRightArm: LimbJointAngles = [0, 0, 0];
  private gLeftLeg: LimbJointAngles = [0, 0, 0];
  private gRightLeg: LimbJointAngles = [0, 0, 0];

  // Lip-sync externally driven values
  public speechMouthOpen = 0;
  public speechViseme: VisemeWeights = { a: 0.5, o: 0.25, e: 0.25 };
  public speechHeadBob = 0;

  // Idle life toggles & timers
  public idleEnabled = true;
  private breathPhase = 0;
  private idleClock = 0;
  private nextAutoBlinkIn = 2.6;

  private tasks: ActiveTask[] = [];

  public readonly currentParams: AvatarParameters = {
    headYaw: 0,
    headPitch: 0,
    headRoll: 0,
    eyeBlink: 0,
    eyeSmile: 0,
    smile: 0,
    laugh: 0,
    mouthOpen: 0,
    viseme: { a: 0.5, o: 0.25, e: 0.25 },
    jumpY: 0,
    bodySquash: 0,
    armSwing: 0,
    bodySwayX: 0,
    breath: 0,
    laughBounce: 0,
    leftArm: [0, 0, 0],
    rightArm: [0, 0, 0],
    leftLeg: [0, 0, 0],
    rightLeg: [0, 0, 0],
  };

  private addOrReplaceTask(
    id: string,
    duration: number,
    evaluate: (tNorm: number, elapsed: number) => void,
    onFinish?: () => void
  ): Promise<void> {
    const existingIdx = this.tasks.findIndex((t) => t.id === id);
    if (existingIdx >= 0) {
      const prev = this.tasks[existingIdx];
      prev.onFinish?.();
      prev.resolve();
      this.tasks.splice(existingIdx, 1);
    }

    return new Promise<void>((resolve) => {
      this.tasks.push({
        id,
        elapsed: 0,
        duration: Math.max(0.05, duration),
        resolve,
        evaluate,
        onFinish,
      });
    });
  }

  /** Trigger a crisp natural blink (or double-blink). */
  public blink(doubleBlink = false): Promise<void> {
    const dur = doubleBlink ? 0.36 : 0.16;
    const singleEnv = (u: number): number => {
      if (u <= 0 || u >= 1) return 0;
      if (u < 0.26) return u / 0.26;
      if (u < 0.62) return 1.0;
      return 1 - (u - 0.62) / 0.38;
    };
    return this.addOrReplaceTask('blink', dur, (t) => {
      if (!doubleBlink) {
        this.gestureBlink = Math.max(this.gestureBlink, singleEnv(t));
      } else {
        const sub = t < 0.47 ? t / 0.47 : t > 0.53 ? (t - 0.53) / 0.47 : 0;
        this.gestureBlink = Math.max(this.gestureBlink, singleEnv(sub));
      }
    });
  }

  /** Warm, friendly smile with subtle cheek lift and gentle head tilt. */
  public smile(options: SmileOptions = {}): Promise<void> {
    const intensity = options.intensity ?? 1.0;
    const durationSec = (options.duration ?? 2000) / 1000;
    this.laughSpring.target = 0;

    return this.addOrReplaceTask(
      'expression',
      durationSec,
      (t) => {
        const env = t < 0.14 ? t / 0.14 : t > 0.82 ? (1 - t) / 0.18 : 1.0;
        this.smileSpring.target = intensity * env;
        this.eyeSmileSpring.target = 0;
        this.gestureRoll += 0.16 * intensity * env;
      },
      () => {
        this.smileSpring.target = 0;
        this.eyeSmileSpring.target = 0;
      }
    );
  }

  /** Joyful big laugh: crescent happy eyes (^_^), wide open laugh mouth, and cheerful forearm bounce! */
  public laugh(options: LaughOptions = {}): Promise<void> {
    const durationSec = (options.duration ?? 2300) / 1000;
    this.smileSpring.target = 0;

    return this.addOrReplaceTask(
      'expression',
      durationSec,
      (t, elapsed) => {
        const env = t < 0.10 ? t / 0.10 : t > 0.84 ? (1 - t) / 0.16 : 1.0;
        this.laughSpring.target = env;
        this.eyeSmileSpring.target = env;
        const chuckleWave = Math.abs(Math.sin(elapsed * Math.PI * 5.6));
        this.gestureLaughBounce += chuckleWave * 7.5 * env;
        this.gesturePitch -= 0.24 * env + chuckleWave * 0.10 * env;
        this.gestureRoll += Math.sin(elapsed * Math.PI * 2.8) * 0.14 * env;
        this.gestureArm += (0.28 + chuckleWave * 0.22) * env;

        // Lively forearm & hand chuckle bounce
        this.gLeftArm[0] += (0.38 + chuckleWave * 0.16) * env;
        this.gLeftArm[1] += (0.46 + chuckleWave * 0.18) * env;
        this.gLeftArm[2] += 0.12 * chuckleWave * env;
        this.gRightArm[0] -= (0.38 + chuckleWave * 0.16) * env;
        this.gRightArm[1] -= (0.46 + chuckleWave * 0.18) * env;
        this.gRightArm[2] -= 0.12 * chuckleWave * env;
      },
      () => {
        this.laughSpring.target = 0;
        this.eyeSmileSpring.target = 0;
      }
    );
  }

  /**
   * Wave hand hello (`avatar.wave()`):
   * Smoothly raises chosen arm (`'right'` = viewer-right hand by default) and
   * oscillates forearm + phase-lagged wrist with a warm welcoming smile!
   */
  public wave(options: WaveOptions = {}): Promise<void> {
    const hand = options.hand ?? 'right';
    const times = Math.max(1, options.times ?? 3);
    const durationSec = options.duration ? options.duration / 1000 : 0.44 + times * 0.46;
    const isRight = hand === 'right';
    const sign = isRight ? -1 : 1;

    return this.addOrReplaceTask(
      'limb-upper',
      durationSec,
      (t) => {
        const lift = smoothHoldEnvelope(t, 0.20, 0.78);
        const waveT = Math.max(0, Math.min(1, (t - 0.12) / 0.74));
        const osc = Math.sin(waveT * Math.PI * 2 * times);
        const wristOsc = Math.sin(waveT * Math.PI * 2 * times - 0.68);

        const active = isRight ? this.gRightArm : this.gLeftArm;
        const passive = isRight ? this.gLeftArm : this.gRightArm;

        active[0] += sign * (1.98 + 0.06 * osc) * lift;
        active[1] += sign * (0.45 + 0.26 * osc) * lift;
        active[2] += sign * (0.24 * wristOsc) * lift;
        // Subtle natural counter-balance on relaxed arm
        passive[0] += -sign * 0.12 * lift;

        // Warm welcoming smile & friendly head tilt
        this.smileSpring.target = Math.max(this.smileSpring.target, 0.9 * lift);
        this.gestureRoll += -sign * 0.14 * lift + osc * 0.04 * lift;
        this.gestureBodySwayX += -sign * 0.10 * lift;
      },
      () => {
        this.smileSpring.target = 0;
      }
    );
  }

  /**
   * Celebratory two-hand cheer (`avatar.cheer()`):
   * Throws both arms high overhead with rhythmic forearm pumps, happy crescent eyes, and bounce!
   */
  public cheer(options: CheerOptions = {}): Promise<void> {
    const times = Math.max(1, options.times ?? 3);
    const durationSec = (options.duration ?? 1900) / 1000;

    return this.addOrReplaceTask(
      'limb-upper',
      durationSec,
      (t) => {
        const env = smoothHoldEnvelope(t, 0.16, 0.80);
        const pump = Math.abs(Math.sin(t * Math.PI * times));

        this.gLeftArm[0] += (1.96 + 0.18 * pump) * env;
        this.gLeftArm[1] += (0.48 - 0.22 * pump) * env;
        this.gLeftArm[2] += 0.16 * Math.sin(t * Math.PI * times * 2) * env;

        this.gRightArm[0] -= (1.96 + 0.18 * pump) * env;
        this.gRightArm[1] -= (0.48 - 0.22 * pump) * env;
        this.gRightArm[2] -= 0.16 * Math.sin(t * Math.PI * times * 2) * env;

        this.gestureJumpY = Math.max(this.gestureJumpY, pump * 18 * env);
        this.gesturePitch -= 0.16 * env + 0.08 * pump * env;
        this.laughSpring.target = 0.85 * env;
        this.eyeSmileSpring.target = env;
      },
      () => {
        this.laughSpring.target = 0;
        this.eyeSmileSpring.target = 0;
      }
    );
  }

  /**
   * Energetic marching / walking steps (`avatar.walk()`):
   * Alternates left & right hip/knee/ankle lifts with opposite arm swings and bouncy stride!
   */
  public walk(options: WalkOptions = {}): Promise<void> {
    const steps = Math.max(2, options.steps ?? 4);
    const durationSec = options.duration ? options.duration / 1000 : steps * 0.46;

    return this.addOrReplaceTask('limb-full', durationSec, (t) => {
      const env = smoothHoldEnvelope(t, 0.14, 0.86);
      const phase = t * Math.PI * steps;
      const s = Math.sin(phase);
      const liftL = Math.max(0, s);
      const liftR = Math.max(0, -s);

      // Alternating high-spirited march knees + level lifted sneakers
      this.gLeftLeg[0] += (0.36 * liftL + 0.05 * liftR) * env;
      this.gLeftLeg[1] += (-0.42 * liftL) * env;
      this.gLeftLeg[2] += (0.12 * liftL) * env;

      this.gRightLeg[0] += (-0.36 * liftR - 0.05 * liftL) * env;
      this.gRightLeg[1] += (0.42 * liftR) * env;
      this.gRightLeg[2] += (-0.12 * liftR) * env;

      // Opposite arm swing (left leg up -> right arm swings out-up, and vice versa)
      this.gRightArm[0] -= (0.10 + 0.42 * liftL) * env;
      this.gRightArm[1] -= (0.24 * liftL) * env;
      this.gLeftArm[0] += (0.10 + 0.42 * liftR) * env;
      this.gLeftArm[1] += (0.24 * liftR) * env;

      // Bouncy stride bob & hip/head rhythm
      const strideBob = Math.abs(s);
      this.gestureJumpY = Math.max(this.gestureJumpY, strideBob * 14 * env);
      this.gestureBodySwayX += -s * 0.22 * env;
      this.gestureRoll += s * 0.09 * env;
    });
  }

  /**
   * Playful leg kick (`avatar.kick()`):
   * Anticipation lean onto standing leg -> snappy leg kick with balancing arms -> cushioned return!
   */
  public kick(options: KickOptions = {}): Promise<void> {
    const leg = options.leg ?? 'right';
    const durationSec = (options.duration ?? 960) / 1000;
    const isRight = leg === 'right';
    const sign = isRight ? -1 : 1;

    return this.addOrReplaceTask(
      'limb-full',
      durationSec,
      (t) => {
        const activeLeg = isRight ? this.gRightLeg : this.gLeftLeg;
        const supportLeg = isRight ? this.gLeftLeg : this.gRightLeg;

        if (t < 0.22) {
          // Stage 1: Quick wind-up chamber
          const p = Math.sin((t / 0.22) * (Math.PI * 0.5));
          activeLeg[0] += sign * 0.22 * p;
          activeLeg[1] += -sign * 0.36 * p;
          this.gestureSquash -= 0.06 * p;
          this.gestureBodySwayX += sign * 0.18 * p;
          this.gLeftArm[0] += 0.28 * p;
          this.gRightArm[0] -= 0.28 * p;
        } else {
          // Stage 2 & 3: Snappy kick extension & smooth ease-out settle
          const u = (t - 0.22) / 0.78;
          const kickArc = u < 0.36 ? Math.sin((u / 0.36) * (Math.PI * 0.5)) : Math.cos(((u - 0.36) / 0.64) * (Math.PI * 0.5));
          const k = kickArc * kickArc * (3 - 2 * kickArc);

          activeLeg[0] += sign * 0.68 * k;
          activeLeg[1] += sign * 0.06 * k;
          activeLeg[2] += sign * 0.20 * k;
          supportLeg[0] += -sign * 0.05 * k;

          // Counter-balance arms & torso lean over standing foot
          this.gLeftArm[0] += 0.52 * k;
          this.gLeftArm[1] += 0.18 * k;
          this.gRightArm[0] -= 0.52 * k;
          this.gRightArm[1] -= 0.18 * k;
          this.gestureBodySwayX += sign * 0.30 * k;
          this.gestureRoll += sign * 0.14 * k;
          this.smileSpring.target = Math.max(this.smileSpring.target, 0.8 * k);
        }
      },
      () => {
        this.smileSpring.target = 0;
      }
    );
  }

  /** Nod head up & down affirmatively (`times` defaults to 2). */
  public nod(options: NodOptions = {}): Promise<void> {
    const times = Math.max(1, options.times ?? 2);
    const speed = options.speed ?? 1.0;
    const durationSec = (times * 0.46) / speed;

    return this.addOrReplaceTask('head-gesture', durationSec, (t) => {
      const env = Math.sin(t * Math.PI);
      const cycle = Math.sin(t * Math.PI * 2 * times);
      this.gesturePitch += (0.48 * Math.max(0, cycle) + 0.18 * Math.min(0, cycle) + 0.12) * (0.35 + 0.65 * env);
    });
  }

  /** Shake head left & right playfully (`times` defaults to 2). */
  public shakeHead(options: ShakeOptions = {}): Promise<void> {
    const times = Math.max(1, options.times ?? 2);
    const speed = options.speed ?? 1.0;
    const durationSec = (times * 0.52) / speed;

    return this.addOrReplaceTask('head-gesture', durationSec, (t) => {
      const env = Math.sin(t * Math.PI);
      const wave = Math.sin(t * Math.PI * 2 * times);
      this.gestureYaw += wave * 0.72 * env;
      this.gestureRoll += wave * 0.24 * env;
      this.gestureBodySwayX += wave * 0.16 * env;
    });
  }

  /**
   * Four-stage physical cartoon jump with full 4-limb skeletal coordination:
   * 1. Anticipation squat + arm swing back (0.00..0.18)
   * 2. Launch & parabolic flight with cheerful raised arms & pointed toes (0.18..0.78)
   * 3. Cushioned landing squash & spring recovery (0.78..1.00)
   */
  public jump(options: JumpOptions = {}): Promise<void> {
    const peakHeight = options.height ?? 128;
    const durationSec = (options.duration ?? 920) / 1000;

    return this.addOrReplaceTask('jump', durationSec, (t) => {
      const tLaunch = 0.18;
      const tLand = 0.78;

      if (t < tLaunch) {
        const p = Math.sin((t / tLaunch) * Math.PI);
        this.gestureSquash -= 0.16 * p;
        this.gestureArm -= 0.32 * p;
        this.gesturePitch += 0.18 * p;
        this.gLeftArm[0] += 0.28 * p;
        this.gLeftArm[1] += 0.32 * p;
        this.gRightArm[0] -= 0.28 * p;
        this.gRightArm[1] -= 0.32 * p;
      } else if (t < tLand) {
        const u = (t - tLaunch) / (tLand - tLaunch);
        const parabola = 4 * u * (1 - u);
        const air = Math.sin(u * Math.PI);
        this.gestureJumpY = peakHeight * parabola;

        const verticalVelocityShape = Math.abs(1 - 2 * u);
        this.gestureSquash += 0.12 * verticalVelocityShape;
        this.gestureArm += 0.82 * air;
        this.gesturePitch += (u - 0.5) * 0.34;

        // Joyful mid-air arm throw & leg tuck!
        this.gLeftArm[0] += 1.55 * air;
        this.gLeftArm[1] += 0.32 * air;
        this.gLeftArm[2] += 0.12 * air;
        this.gRightArm[0] -= 1.55 * air;
        this.gRightArm[1] -= 0.32 * air;
        this.gRightArm[2] -= 0.12 * air;

        this.gLeftLeg[0] += 0.20 * air;
        this.gLeftLeg[1] += -0.24 * air;
        this.gLeftLeg[2] += 0.18 * air;
        this.gRightLeg[0] -= 0.24 * air;
        this.gRightLeg[1] += 0.28 * air;
        this.gRightLeg[2] -= 0.18 * air;
      } else {
        const v = (t - tLand) / (1 - tLand);
        const impact = Math.sin(v * Math.PI) * (1 - v * 0.45);
        this.gestureSquash -= 0.13 * impact;
        this.gestureArm -= 0.16 * impact;
        this.gesturePitch += 0.14 * impact;
        this.gLeftArm[0] += 0.26 * impact;
        this.gRightArm[0] -= 0.26 * impact;
      }
    });
  }

  /** Set persistent programmatic 4-limb skeletal angles (in radians). */
  public setLimbPose(pose: LimbPoseInput): void {
    if (pose.leftArm) this.baseLeftArm = [...pose.leftArm];
    if (pose.rightArm) this.baseRightArm = [...pose.rightArm];
    if (pose.leftLeg) this.baseLeftLeg = [...pose.leftLeg];
    if (pose.rightLeg) this.baseRightLeg = [...pose.rightLeg];
  }

  /** Set continuous 2.5D head orientation (`yaw`, `pitch`, `roll` in `[-1, +1]`). */
  public turnHead(yaw: number, pitch = 0, roll = 0): void {
    this.baseYaw.target = Math.max(-1, Math.min(1, yaw));
    this.basePitch.target = Math.max(-1, Math.min(1, pitch));
    this.baseRoll.target = Math.max(-1, Math.min(1, roll));
  }

  /** Set persistent smile level (`0..1`) for manual slider control. */
  public setSmileLevel(level: number): void {
    this.smileSpring.target = Math.max(0, Math.min(1, level));
    this.eyeSmileSpring.target = 0;
  }

  /** Set persistent blink override (`0..1`) or `null` to return to automatic blinking. */
  private manualBlink: number | null = null;
  public setManualBlink(level: number | null): void {
    this.manualBlink = level === null ? null : Math.max(0, Math.min(1, level));
  }

  /** Reset manual head orientation, expressions, and custom limb poses to neutral idle. */
  public resetPose(): void {
    this.baseYaw.target = 0;
    this.basePitch.target = 0;
    this.baseRoll.target = 0;
    this.smileSpring.target = 0;
    this.laughSpring.target = 0;
    this.eyeSmileSpring.target = 0;
    this.manualBlink = null;
    this.baseLeftArm = [0, 0, 0];
    this.baseRightArm = [0, 0, 0];
    this.baseLeftLeg = [0, 0, 0];
    this.baseRightLeg = [0, 0, 0];
    this.tasks = [];
  }

  public step(dt: number): AvatarParameters {
    this.gestureYaw = 0;
    this.gesturePitch = 0;
    this.gestureRoll = 0;
    this.gestureBlink = 0;
    this.gestureJumpY = 0;
    this.gestureSquash = 0;
    this.gestureArm = 0;
    this.gestureLaughBounce = 0;
    this.gestureBodySwayX = 0;
    this.gLeftArm[0] = 0;
    this.gLeftArm[1] = 0;
    this.gLeftArm[2] = 0;
    this.gRightArm[0] = 0;
    this.gRightArm[1] = 0;
    this.gRightArm[2] = 0;
    this.gLeftLeg[0] = 0;
    this.gLeftLeg[1] = 0;
    this.gLeftLeg[2] = 0;
    this.gRightLeg[0] = 0;
    this.gRightLeg[1] = 0;
    this.gRightLeg[2] = 0;

    // 1. Advance active animation tasks
    for (let i = this.tasks.length - 1; i >= 0; i--) {
      const task = this.tasks[i];
      task.elapsed += dt;
      const tNorm = Math.min(1, task.elapsed / task.duration);
      task.evaluate(tNorm, task.elapsed);
      if (task.elapsed >= task.duration) {
        task.onFinish?.();
        task.resolve();
        this.tasks.splice(i, 1);
      }
    }

    // 2. Advance idle breathing, micro-arm relaxation & spontaneous natural blinking
    let idleYaw = 0;
    let idleRoll = 0;
    let idleSway = 0;
    let idleArmBreath = 0;
    if (this.idleEnabled) {
      this.breathPhase = (this.breathPhase + dt / 3.2) % 1.0;
      this.idleClock += dt;
      idleYaw = Math.sin(this.idleClock * 0.9) * 0.045;
      idleRoll = Math.sin(this.idleClock * 0.65 + 1.1) * 0.035;
      idleSway = Math.sin(this.idleClock * 0.75) * 0.06;
      idleArmBreath = (0.5 + 0.5 * Math.sin(this.breathPhase * Math.PI * 2)) * 0.03;

      this.nextAutoBlinkIn -= dt;
      if (this.nextAutoBlinkIn <= 0) {
        const isDouble = Math.random() < 0.18;
        void this.blink(isDouble);
        this.nextAutoBlinkIn = 2.6 + Math.random() * 2.8;
      }
    }

    // 3. Step springs
    const bYaw = this.baseYaw.step(dt);
    const bPitch = this.basePitch.step(dt);
    const bRoll = this.baseRoll.step(dt);
    const smileVal = this.smileSpring.step(dt);
    const laughVal = this.laughSpring.step(dt);
    const eyeSmileVal = this.eyeSmileSpring.step(dt);
    this.squashSpring.target = this.gestureSquash;
    this.armSpring.target = this.gestureArm;
    const smoothSquash = this.squashSpring.step(dt);
    const smoothArm = this.armSpring.step(dt);

    const p = this.currentParams;
    this.leftArmSprings.setTarget(
      this.baseLeftArm[0] + this.gLeftArm[0] + idleArmBreath,
      this.baseLeftArm[1] + this.gLeftArm[1],
      this.baseLeftArm[2] + this.gLeftArm[2]
    );
    this.rightArmSprings.setTarget(
      this.baseRightArm[0] + this.gRightArm[0] - idleArmBreath,
      this.baseRightArm[1] + this.gRightArm[1],
      this.baseRightArm[2] + this.gRightArm[2]
    );
    this.leftLegSprings.setTarget(
      this.baseLeftLeg[0] + this.gLeftLeg[0],
      this.baseLeftLeg[1] + this.gLeftLeg[1],
      this.baseLeftLeg[2] + this.gLeftLeg[2]
    );
    this.rightLegSprings.setTarget(
      this.baseRightLeg[0] + this.gRightLeg[0],
      this.baseRightLeg[1] + this.gRightLeg[1],
      this.baseRightLeg[2] + this.gRightLeg[2]
    );

    this.leftArmSprings.step(dt, p.leftArm);
    this.rightArmSprings.step(dt, p.rightArm);
    this.leftLegSprings.step(dt, p.leftLeg);
    this.rightLegSprings.step(dt, p.rightLeg);

    // 4. Assemble final clamped parameters
    p.headYaw = Math.max(-1, Math.min(1, bYaw + this.gestureYaw + idleYaw));
    p.headPitch = Math.max(-1, Math.min(1, bPitch + this.gesturePitch + this.speechHeadBob));
    p.headRoll = Math.max(-1, Math.min(1, bRoll + this.gestureRoll + idleRoll));
    p.eyeBlink =
      this.manualBlink !== null
        ? this.manualBlink
        : Math.max(0, Math.min(1, this.gestureBlink));
    p.eyeSmile = Math.max(0, Math.min(1, eyeSmileVal));
    p.smile = Math.max(0, Math.min(1, smileVal));
    p.laugh = Math.max(0, Math.min(1, laughVal));
    p.mouthOpen = Math.max(0, Math.min(1, this.speechMouthOpen));
    p.viseme.a = this.speechViseme.a;
    p.viseme.o = this.speechViseme.o;
    p.viseme.e = this.speechViseme.e;
    p.jumpY = Math.max(0, this.gestureJumpY);
    p.bodySquash = Math.max(-0.28, Math.min(0.24, smoothSquash));
    p.armSwing = Math.max(-0.6, Math.min(1.1, smoothArm));
    p.bodySwayX = Math.max(-1, Math.min(1, this.gestureBodySwayX + idleSway));
    p.breath = this.breathPhase;
    p.laughBounce = this.gestureLaughBounce;

    return p;
  }
}
