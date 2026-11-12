export interface LayerSpec {
  texture: string;
  x: number;
  y: number;
  width: number;
  height: number;
  meshCols: number;
  meshRows: number;
}

/** `[proximal, middle, distal]` joint angles in radians (`[shoulder, elbow, wrist]` or `[hip, knee, ankle]`). */
export type LimbJointAngles = [number, number, number];

export interface AvatarLandmarks {
  headCenter: [number, number];
  leftEye: [number, number];
  rightEye: [number, number];
  nose: [number, number];
  mouth: [number, number];
  neckPivot: [number, number];
  chest: [number, number];
  hips: [number, number];
  knees: [number, number];
  feetGround: [number, number];
  leftShoulder?: [number, number];
  leftElbow?: [number, number];
  leftWrist?: [number, number];
  rightShoulder?: [number, number];
  rightElbow?: [number, number];
  rightWrist?: [number, number];
  leftHip?: [number, number];
  leftKnee?: [number, number];
  leftAnkle?: [number, number];
  rightHip?: [number, number];
  rightKnee?: [number, number];
  rightAnkle?: [number, number];
  [key: string]: [number, number] | undefined;
}

export interface AvatarManifest {
  version: string;
  id: string;
  name: string;
  outfitMode?: 'pants' | 'skirt';
  hasGlasses?: boolean;
  canvasSize: { width: number; height: number };
  characterBounds: { x: number; y: number; width: number; height: number };
  layers: {
    head: LayerSpec;
    body: LayerSpec;
    torso?: LayerSpec;
    leftArm?: LayerSpec;
    rightArm?: LayerSpec;
    leftLeg?: LayerSpec;
    rightLeg?: LayerSpec;
  };
  patches: {
    eyes_closed?: string;
    eyes_happy?: string;
    mouth_smile?: string;
    mouth_laugh?: string;
    mouth_a?: string;
    mouth_e?: string;
    mouth_o?: string;
    [key: string]: string | undefined;
  };
  landmarks: AvatarLandmarks;
}

export interface VisemeWeights {
  a: number;
  o: number;
  e: number;
}

export interface AvatarParameters {
  /** -1 (turn left) to +1 (turn right) */
  headYaw: number;
  /** -1 (look up) to +1 (nod down) */
  headPitch: number;
  /** -1 (tilt left) to +1 (tilt right) */
  headRoll: number;
  /** 0 (open) to 1 (closed blink) */
  eyeBlink: number;
  /** 0 (normal) to 1 (happy crescent eyes ^_^) */
  eyeSmile: number;
  /** 0 to 1 warm closed/gentle smile */
  smile: number;
  /** 0 to 1 wide joyful open laugh */
  laugh: number;
  /** 0 to 1 overall speaking jaw opening */
  mouthOpen: number;
  /** Speaking vowel weights */
  viseme: VisemeWeights;
  /** Upward airborne jump offset in canvas pixels (>= 0) */
  jumpY: number;
  /** -0.25 (crouch squash) to +0.20 (launch stretch) */
  bodySquash: number;
  /** -0.5 (arms back) to +1.0 (arms cheerful outward lift, fallback for non-segmented rigs) */
  armSwing: number;
  /** -1 to +1 subtle horizontal torso sway */
  bodySwayX: number;
  /** 0 to 1 chest breathing phase */
  breath: number;
  /** Rhythmic chuckle vertical bounce in pixels */
  laughBounce: number;
  /** Stage-left arm `[shoulder, elbow, wrist]` in radians (+ raises outward-up) */
  leftArm: LimbJointAngles;
  /** Stage-right arm `[shoulder, elbow, wrist]` in radians (- raises outward-up) */
  rightArm: LimbJointAngles;
  /** Stage-left leg `[hip, knee, ankle]` in radians (+ swings outward-up) */
  leftLeg: LimbJointAngles;
  /** Stage-right leg `[hip, knee, ankle]` in radians (- swings outward-up) */
  rightLeg: LimbJointAngles;
}

export type SpeechMode = 'browser' | 'synth' | 'edge';

export interface SpeakOptions {
  mode?: SpeechMode;
  voice?: string;
  rate?: number;
  pitch?: number;
}

export interface SmileOptions {
  intensity?: number;
  duration?: number;
}

export interface LaughOptions {
  duration?: number;
}

export interface NodOptions {
  times?: number;
  speed?: number;
}

export interface ShakeOptions {
  times?: number;
  speed?: number;
}

export interface JumpOptions {
  height?: number;
  duration?: number;
}

export interface WaveOptions {
  /** Which side on stage to wave (`'right'` = viewer-right raised hand, `'left'` = viewer-left) */
  hand?: 'right' | 'left';
  /** Number of forearm/wrist wave cycles (default `3`) */
  times?: number;
  /** Total wave duration in ms (optional override) */
  duration?: number;
}

export interface CheerOptions {
  /** Number of雙手欢呼 bounces (default `3`) */
  times?: number;
  /** Total duration in ms (default `1900`) */
  duration?: number;
}

export interface WalkOptions {
  /** Number of alternating marching steps (default `4`) */
  steps?: number;
  /** Total duration in ms (optional override) */
  duration?: number;
}

export interface KickOptions {
  /** Which leg on stage to kick (`'right'` = viewer-right leg, `'left'` = viewer-left) */
  leg?: 'right' | 'left';
  /** Total duration in ms (default `960`) */
  duration?: number;
}

export interface LimbPoseInput {
  leftArm?: LimbJointAngles;
  rightArm?: LimbJointAngles;
  leftLeg?: LimbJointAngles;
  rightLeg?: LimbJointAngles;
}
