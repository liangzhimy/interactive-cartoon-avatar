import { Container, Graphics, Mesh, MeshGeometry, Text, Texture } from 'pixi.js';
import { Spring2D } from './SpringPhysics';
import type {
  AvatarLandmarks,
  AvatarManifest,
  AvatarParameters,
  LayerSpec,
  LimbJointAngles,
} from './types';

const LANDMARK_LABELS: Record<string, string> = {
  leftEye: '左眼',
  rightEye: '右眼',
  mouth: '嘴',
  neckPivot: '颈轴',
  leftShoulder: '左肩',
  leftElbow: '左肘',
  leftWrist: '左腕',
  rightShoulder: '右肩',
  rightElbow: '右肘',
  rightWrist: '右腕',
  leftHip: '左髋',
  leftKnee: '左膝',
  leftAnkle: '左踝',
  rightHip: '右髋',
  rightKnee: '右膝',
  rightAnkle: '右踝',
  feetGround: '地面',
};

const BONE_SEGMENTS: Array<[string, string, number]> = [
  ['neckPivot', 'headCenter', 0xe0522d],
  ['neckPivot', 'chest', 0x2f7d6d],
  ['chest', 'hips', 0x2f7d6d],
  ['neckPivot', 'leftShoulder', 0xd97706],
  ['leftShoulder', 'leftElbow', 0xd97706],
  ['leftElbow', 'leftWrist', 0xd97706],
  ['neckPivot', 'rightShoulder', 0xd97706],
  ['rightShoulder', 'rightElbow', 0xd97706],
  ['rightElbow', 'rightWrist', 0xd97706],
  ['hips', 'leftHip', 0x2563eb],
  ['leftHip', 'leftKnee', 0x2563eb],
  ['leftKnee', 'leftAnkle', 0x2563eb],
  ['hips', 'rightHip', 0x2563eb],
  ['rightHip', 'rightKnee', 0x2563eb],
  ['rightKnee', 'rightAnkle', 0x2563eb],
];

interface GridData {
  cols: number;
  rows: number;
  restPositions: Float32Array;
  deformedPositions: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  geometry: MeshGeometry;
}

export interface RigTextures {
  head: Texture;
  body: Texture;
  torso?: Texture;
  leftArm?: Texture;
  rightArm?: Texture;
  leftLeg?: Texture;
  rightLeg?: Texture;
  patches: Record<string, Texture>;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function buildGridGeometry(spec: LayerSpec): GridData {
  const { cols, rows } = { cols: spec.meshCols, rows: spec.meshRows };
  const numVerts = (cols + 1) * (rows + 1);
  const restPositions = new Float32Array(numVerts * 2);
  const deformedPositions = new Float32Array(numVerts * 2);
  const uvs = new Float32Array(numVerts * 2);
  const indices = new Uint32Array(cols * rows * 6);

  let vIdx = 0;
  for (let r = 0; r <= rows; r++) {
    const v = r / rows;
    const py = spec.y + v * spec.height;
    for (let c = 0; c <= cols; c++) {
      const u = c / cols;
      const px = spec.x + u * spec.width;
      restPositions[vIdx] = px;
      restPositions[vIdx + 1] = py;
      deformedPositions[vIdx] = px;
      deformedPositions[vIdx + 1] = py;
      uvs[vIdx] = u;
      uvs[vIdx + 1] = v;
      vIdx += 2;
    }
  }

  let iIdx = 0;
  const stride = cols + 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tl = r * stride + c;
      const tr = tl + 1;
      const bl = (r + 1) * stride + c;
      const br = bl + 1;

      indices[iIdx++] = tl;
      indices[iIdx++] = tr;
      indices[iIdx++] = br;

      indices[iIdx++] = tl;
      indices[iIdx++] = br;
      indices[iIdx++] = bl;
    }
  }

  const geometry = new MeshGeometry({
    positions: deformedPositions,
    uvs,
    indices,
  });

  return { cols, rows, restPositions, deformedPositions, uvs, indices, geometry };
}

/**
 * 2.5D Parametric & 6-Layer Skeletal Mesh Rig for Full-Body Cartoon Character:
 * - Shared GPU `headGeometry` across base head + feathered eye/mouth patch meshes
 *   with distortion-free 3D cranial ellipsoid projection.
 * - Full 4-limb skeletal rig (`torso`, `leftArm`, `rightArm`, `leftLeg`, `rightLeg`)
 *   driven by 2D Dual-Quaternion Skinning (zero volume collapse at elbows/knees),
 *   with automatic fallback to single `body` mesh for unsegmented custom imports.
 */
export class MeshRig {
  public readonly root = new Container();
  public readonly characterContainer = new Container();
  private readonly shadowGraphics = new Graphics();
  private readonly debugGraphics = new Graphics();
  private readonly labelContainer = new Container();
  private labelTexts: Record<string, Text> = {};

  private manifest: AvatarManifest;
  private headGrid: GridData;
  private bodyGrid: GridData | null = null;
  private torsoGrid: GridData | null = null;
  private leftArmGrid: GridData | null = null;
  private rightArmGrid: GridData | null = null;
  private leftLegGrid: GridData | null = null;
  private rightLegGrid: GridData | null = null;
  private hasSkeletalLimbs = false;

  private headBaseMesh: Mesh;
  private patchMeshes: Record<string, Mesh> = {};

  private hairLag = new Spring2D(0, 0, 125, 11);
  private prevHeadMotion = { x: 0, y: 0 };

  public showWireframe = false;
  public showLandmarks = false;
  public hoveredLandmark: string | null = null;
  public activeLandmark: string | null = null;

  constructor(manifest: AvatarManifest, textures: RigTextures) {
    this.manifest = manifest;
    this.headGrid = buildGridGeometry(manifest.layers.head);

    // 1. Ground shadow beneath feet
    this.root.addChild(this.shadowGraphics);
    this.root.addChild(this.characterContainer);

    const layers = manifest.layers;
    if (
      layers.torso &&
      layers.leftArm &&
      layers.rightArm &&
      layers.leftLeg &&
      layers.rightLeg &&
      textures.torso &&
      textures.leftArm &&
      textures.rightArm &&
      textures.leftLeg &&
      textures.rightLeg
    ) {
      this.hasSkeletalLimbs = true;
      this.leftLegGrid = buildGridGeometry(layers.leftLeg);
      this.rightLegGrid = buildGridGeometry(layers.rightLeg);
      this.torsoGrid = buildGridGeometry(layers.torso);
      this.leftArmGrid = buildGridGeometry(layers.leftArm);
      this.rightArmGrid = buildGridGeometry(layers.rightArm);

      // Z-order back-to-front: Legs -> Torso/Shorts -> Head + Expressions -> Arms (so waving hand passes cleanly in front of hair!)
      this.characterContainer.addChild(
        new Mesh({ geometry: this.leftLegGrid.geometry, texture: textures.leftLeg })
      );
      this.characterContainer.addChild(
        new Mesh({ geometry: this.rightLegGrid.geometry, texture: textures.rightLeg })
      );
      this.characterContainer.addChild(
        new Mesh({ geometry: this.torsoGrid.geometry, texture: textures.torso })
      );
    } else {
      this.hasSkeletalLimbs = false;
      this.bodyGrid = buildGridGeometry(layers.body);
      this.characterContainer.addChild(
        new Mesh({ geometry: this.bodyGrid.geometry, texture: textures.body })
      );
    }

    // Head base mesh + shared-geometry expression/viseme patch meshes
    this.headBaseMesh = new Mesh({
      geometry: this.headGrid.geometry,
      texture: textures.head,
    });
    this.characterContainer.addChild(this.headBaseMesh);

    const patchOrder = [
      'mouth_smile',
      'mouth_e',
      'mouth_a',
      'mouth_o',
      'mouth_laugh',
      'eyes_happy',
      'eyes_closed',
    ];
    for (const key of patchOrder) {
      const tex = textures.patches[key];
      if (tex) {
        const m = new Mesh({
          geometry: this.headGrid.geometry,
          texture: tex,
        });
        m.alpha = 0;
        m.visible = false;
        this.patchMeshes[key] = m;
        this.characterContainer.addChild(m);
      }
    }

    // Foreground arms on top of torso and outer hair tips
    if (this.hasSkeletalLimbs && this.leftArmGrid && this.rightArmGrid && textures.leftArm && textures.rightArm) {
      this.characterContainer.addChild(
        new Mesh({ geometry: this.leftArmGrid.geometry, texture: textures.leftArm })
      );
      this.characterContainer.addChild(
        new Mesh({ geometry: this.rightArmGrid.geometry, texture: textures.rightArm })
      );
    }

    // Optional wireframe & landmark calibration overlay
    this.root.addChild(this.debugGraphics);
    this.labelContainer.visible = false;
    for (const [key, label] of Object.entries(LANDMARK_LABELS)) {
      const txt = new Text({
        text: label,
        style: {
          fontFamily: '"Noto Sans SC", system-ui, sans-serif',
          fontSize: 13,
          fontWeight: '700',
          fill: 0xffffff,
          stroke: { color: 0x1e2022, width: 3.5 },
        },
      });
      txt.anchor.set(0.5, 0.5);
      this.labelTexts[key] = txt;
      this.labelContainer.addChild(txt);
    }
    this.root.addChild(this.labelContainer);
  }

  public getLandmarks(): AvatarLandmarks {
    return this.manifest.landmarks;
  }

  public updateLandmark(name: string, x: number, y: number): void {
    this.manifest.landmarks[name] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
  }

  private setPatchAlpha(key: string, alpha: number): void {
    const m = this.patchMeshes[key];
    if (!m) return;
    const clamped = Math.max(0, Math.min(1, alpha));
    m.alpha = clamped;
    m.visible = clamped > 0.008;
  }

  private deformLimbArm(
    grid: GridData,
    j0: [number, number],
    j1: [number, number],
    j2: [number, number],
    angles: LimbJointAngles,
    attachDX: number,
    attachDY: number
  ): void {
    const [a1, a2, a3] = angles;
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    const c12 = Math.cos(a1 + a2);
    const s12 = Math.sin(a1 + a2);

    const b1x = j1[0] - j0[0];
    const b1y = j1[1] - j0[1];
    const len1 = Math.hypot(b1x, b1y) || 1;
    const u1x = b1x / len1;
    const u1y = b1y / len1;

    const b2x = j2[0] - j1[0];
    const b2y = j2[1] - j1[1];
    const len2 = Math.hypot(b2x, b2y) || 1;
    const u2x = b2x / len2;
    const u2y = b2y / len2;

    const j1px = j0[0] + b1x * c1 - b1y * s1;
    const j1py = j0[1] + b1x * s1 + b1y * c1;
    const j2px = j1px + b2x * c12 - b2y * s12;
    const j2py = j1py + b2x * s12 + b2y * c12;

    const rest = grid.restPositions;
    const def = grid.deformedPositions;
    for (let i = 0; i < rest.length; i += 2) {
      const rx = rest[i];
      const ry = rest[i + 1];

      const d1 = (rx - j1[0]) * u1x + (ry - j1[1]) * u1y;
      const wm = smoothstep(-38, 38, d1);
      const ang1 = a1 + wm * a2;
      const ca1 = Math.cos(ang1);
      const sa1 = Math.sin(ang1);
      const dx1 = rx - j1[0];
      const dy1 = ry - j1[1];
      const qx = j1px + dx1 * ca1 - dy1 * sa1;
      const qy = j1py + dx1 * sa1 + dy1 * ca1;

      const d2 = (rx - j2[0]) * u2x + (ry - j2[1]) * u2y;
      const we = smoothstep(-22, 22, d2);
      const ang2 = we * a3;
      const ca2 = Math.cos(ang2);
      const sa2 = Math.sin(ang2);
      const dx2 = qx - j2px;
      const dy2 = qy - j2py;

      def[i] = j2px + dx2 * ca2 - dy2 * sa2 + attachDX;
      def[i + 1] = j2py + dx2 * sa2 + dy2 * ca2 + attachDY;
    }
    grid.geometry.getBuffer('aPosition').update();
  }

  private deformLimbLeg(
    grid: GridData,
    j0: [number, number],
    j1: [number, number],
    j2: [number, number],
    angles: LimbJointAngles,
    hipDX: number,
    hipDYNoJump: number,
    jumpY: number,
    squash: number,
    feetY: number,
    sideSign: number
  ): void {
    const [a1, a2, a3] = angles;
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    const c12 = Math.cos(a1 + a2);
    const s12 = Math.sin(a1 + a2);

    const b1x = j1[0] - j0[0];
    const b1y = j1[1] - j0[1];
    const len1 = Math.hypot(b1x, b1y) || 1;
    const u1x = b1x / len1;
    const u1y = b1y / len1;

    const b2x = j2[0] - j1[0];
    const b2y = j2[1] - j1[1];
    const len2 = Math.hypot(b2x, b2y) || 1;
    const u2x = b2x / len2;
    const u2y = b2y / len2;

    const j1px = j0[0] + b1x * c1 - b1y * s1;
    const j1py = j0[1] + b1x * s1 + b1y * c1;
    const j2px = j1px + b2x * c12 - b2y * s12;
    const j2py = j1py + b2x * s12 + b2y * c12;

    // When planted on the ground with near-zero leg swing, pin the sneaker sole during crouch squats;
    // when kicking, marching, or jumping airborne, move the sneaker rigidly without squashing it!
    const groundPin =
      Math.max(0, 1 - jumpY / 24) * Math.max(0, 1 - (Math.abs(a1) + Math.abs(a2)) * 2.5);
    const legSpan = Math.max(80, feetY - j0[1]);

    const rest = grid.restPositions;
    const def = grid.deformedPositions;
    for (let i = 0; i < rest.length; i += 2) {
      const rx = rest[i];
      const ry = rest[i + 1];

      const d1 = (rx - j1[0]) * u1x + (ry - j1[1]) * u1y;
      const wm = smoothstep(-38, 38, d1);
      const ang1 = a1 + wm * a2;
      const ca1 = Math.cos(ang1);
      const sa1 = Math.sin(ang1);
      const dx1 = rx - j1[0];
      const dy1 = ry - j1[1];
      const qx = j1px + dx1 * ca1 - dy1 * sa1;
      const qy = j1py + dx1 * sa1 + dy1 * ca1;

      const d2 = (rx - j2[0]) * u2x + (ry - j2[1]) * u2y;
      const we = smoothstep(-22, 22, d2);
      const ang2 = we * a3;
      const ca2 = Math.cos(ang2);
      const sa2 = Math.sin(ang2);
      const dx2 = qx - j2px;
      const dy2 = qy - j2py;

      const fx = j2px + dx2 * ca2 - dy2 * sa2;
      const fy = j2py + dx2 * sa2 + dy2 * ca2;

      const hFrac = Math.max(0, Math.min(1, (feetY - fy) / legSpan));
      const uw = hFrac * hFrac * (3 - 2 * hFrac);
      const wHip = 1 - groundPin * (1 - uw);
      const kneeBow =
        squash < 0 && groundPin > 0.01
          ? sideSign * (-squash) * 20 * Math.sin(hFrac * Math.PI) * groundPin
          : 0;

      def[i] = fx + hipDX * wHip + kneeBow;
      def[i + 1] = fy + hipDYNoJump * wHip - jumpY;
    }
    grid.geometry.getBuffer('aPosition').update();
  }

  public update(params: AvatarParameters, dt: number): void {
    const lm = this.manifest.landmarks;
    const [neckX, neckY] = lm.neckPivot;
    const [faceX, faceY] = lm.headCenter;
    const [, mouthY] = lm.mouth;
    const [, eyeY] = lm.leftEye;
    const [feetX, feetY] = lm.feetGround;
    const [, hipsY] = lm.hips;

    // --- 1. Blend Facial Expression & Viseme Patch Opacities ---
    const snap = (v: number): number => {
      const u = Math.max(0, Math.min(1, (v - 0.25) / 0.5));
      return u * u * (3 - 2 * u);
    };
    const blink = snap(params.eyeBlink);
    const happyEyeWeight = snap(Math.max(params.eyeSmile, params.laugh)) * (1 - blink);
    this.setPatchAlpha('eyes_closed', blink);
    this.setPatchAlpha('eyes_happy', happyEyeWeight);

    const laughW = Math.max(0, Math.min(1, params.laugh));
    const speakGate = Math.max(0, Math.min(1, params.mouthOpen * 1.35)) * (1 - laughW * 0.85);
    const vSum = params.viseme.a + params.viseme.o + params.viseme.e + 1e-5;
    const normA = params.viseme.a / vSum;
    const normO = params.viseme.o / vSum;
    const normE = params.viseme.e / vSum;

    const smileW = Math.max(0, Math.min(1, params.smile)) * (1 - laughW) * (1 - speakGate * 0.8);

    this.setPatchAlpha('mouth_laugh', laughW);
    this.setPatchAlpha('mouth_smile', smileW);
    this.setPatchAlpha('mouth_a', speakGate * normA * Math.min(1, params.mouthOpen * 1.55));
    this.setPatchAlpha('mouth_o', speakGate * normO * Math.min(1, params.mouthOpen * 1.55));
    this.setPatchAlpha('mouth_e', speakGate * normE * Math.min(1, params.mouthOpen * 1.45));

    // --- 2. Secondary Hair Spring Lag ---
    const headMotionX = params.headYaw * 18 + params.headRoll * 14 + params.bodySwayX * 8;
    const headMotionY = params.headPitch * 14 - params.jumpY * 0.22 - params.laughBounce * 0.6;
    const dvx = (headMotionX - this.prevHeadMotion.x) / Math.max(dt, 0.008);
    const dvy = (headMotionY - this.prevHeadMotion.y) / Math.max(dt, 0.008);
    this.prevHeadMotion.x = headMotionX;
    this.prevHeadMotion.y = headMotionY;

    this.hairLag.setTarget(
      Math.max(-14, Math.min(14, -dvx * 0.022)),
      Math.max(-16, Math.min(16, -dvy * 0.024))
    );
    const hairOffset = this.hairLag.step(dt);

    // --- 3. Compute Whole-Body Torso Transform Helper ---
    const squash = params.bodySquash;
    const jumpY = Math.max(0, params.jumpY);
    const breathSin = Math.sin(params.breath * Math.PI * 2);
    const breathLift = breathSin * 2.4;
    const chuckleLift = params.laughBounce;
    const yaw = Math.max(-1, Math.min(1, params.headYaw));
    const pitch = Math.max(-1, Math.min(1, params.headPitch));

    const bodyHeightSpan = Math.max(100, feetY - neckY);
    const crouchDropAtNeck = -squash * bodyHeightSpan * 0.34;
    const totalNeckDeltaY = -jumpY - crouchDropAtNeck - breathLift - chuckleLift;
    const totalNeckDeltaX = params.bodySwayX * 10;

    const evalTorsoPoint = (rx: number, ry: number): [number, number] => {
      const hFrac = Math.max(0, Math.min(1.15, (feetY - ry) / bodyHeightSpan));
      const upperWeight = hFrac * hFrac * (3 - 2 * Math.min(1, hFrac));
      const midBodyBell = Math.sin(Math.min(1, Math.max(0, hFrac)) * Math.PI);
      const horizScale = 1 - squash * 0.32 * midBodyBell + breathSin * 0.008 * upperWeight;

      let x = neckX + (rx - neckX) * horizScale;
      const chestFactor = Math.max(0, ry < hipsY ? (hipsY - ry) / Math.max(1, hipsY - neckY) : 0);
      const y = ry - crouchDropAtNeck * upperWeight - (breathLift + chuckleLift) * chestFactor - jumpY;

      const neckFollow = Math.max(0, Math.min(1, (neckY + 30 - ry) / 58));
      x += yaw * 4.2 * neckFollow * neckFollow + totalNeckDeltaX * upperWeight;
      return [x, y];
    };

    // --- 4. Deform Skeletal 5-Part Body OR Single Fallback Body ---
    if (
      this.hasSkeletalLimbs &&
      this.torsoGrid &&
      this.leftArmGrid &&
      this.rightArmGrid &&
      this.leftLegGrid &&
      this.rightLegGrid
    ) {
      // 4a. Torso + Shorts
      const tRest = this.torsoGrid.restPositions;
      const tDef = this.torsoGrid.deformedPositions;
      for (let i = 0; i < tRest.length; i += 2) {
        const [tx, ty] = evalTorsoPoint(tRest[i], tRest[i + 1]);
        tDef[i] = tx;
        tDef[i + 1] = ty;
      }
      this.torsoGrid.geometry.getBuffer('aPosition').update();

      // 4b. Left & Right Arms (exact shoulder socket tracking!)
      const lSh = lm.leftShoulder ?? [341.0, 417.0];
      const lEl = lm.leftElbow ?? [326.0, 523.0];
      const lWr = lm.leftWrist ?? [306.0, 657.0];
      const [lShX, lShY] = evalTorsoPoint(lSh[0], lSh[1]);
      this.deformLimbArm(
        this.leftArmGrid,
        lSh,
        lEl,
        lWr,
        params.leftArm,
        lShX - lSh[0],
        lShY - lSh[1]
      );

      const rSh = lm.rightShoulder ?? [554.0, 417.0];
      const rEl = lm.rightElbow ?? [569.0, 523.0];
      const rWr = lm.rightWrist ?? [589.0, 657.0];
      const [rShX, rShY] = evalTorsoPoint(rSh[0], rSh[1]);
      this.deformLimbArm(
        this.rightArmGrid,
        rSh,
        rEl,
        rWr,
        params.rightArm,
        rShX - rSh[0],
        rShY - rSh[1]
      );

      // 4c. Left & Right Legs (exact hip socket tracking + grounded squat pin!)
      const lHip = lm.leftHip ?? [394.0, 715.0];
      const lKnee = lm.leftKnee ?? [393.0, 900.0];
      const lAnk = lm.leftAnkle ?? [383.0, 1050.0];
      const [lHipX, lHipY] = evalTorsoPoint(lHip[0], lHip[1]);
      this.deformLimbLeg(
        this.leftLegGrid,
        lHip,
        lKnee,
        lAnk,
        params.leftLeg,
        lHipX - lHip[0],
        lHipY + jumpY - lHip[1],
        jumpY,
        squash,
        feetY,
        -1
      );

      const rHip = lm.rightHip ?? [501.0, 715.0];
      const rKnee = lm.rightKnee ?? [502.0, 900.0];
      const rAnk = lm.rightAnkle ?? [512.0, 1050.0];
      const [rHipX, rHipY] = evalTorsoPoint(rHip[0], rHip[1]);
      this.deformLimbLeg(
        this.rightLegGrid,
        rHip,
        rKnee,
        rAnk,
        params.rightLeg,
        rHipX - rHip[0],
        rHipY + jumpY - rHip[1],
        jumpY,
        squash,
        feetY,
        1
      );
    } else if (this.bodyGrid) {
      // Single-body fallback for unsegmented custom imported characters
      const bRest = this.bodyGrid.restPositions;
      const bDef = this.bodyGrid.deformedPositions;
      const shoulderY = neckY + 46;
      const leftShoulderX = neckX - 78;
      const rightShoulderX = neckX + 78;
      const armAngle = params.armSwing * 0.28 + params.laugh * 0.11 * Math.sin(params.breath * 18);

      for (let i = 0; i < bRest.length; i += 2) {
        const rx = bRest[i];
        const ry = bRest[i + 1];
        let [x, y] = evalTorsoPoint(rx, ry);
        y += jumpY;

        if (squash < 0 && ry > hipsY && ry < feetY - 25) {
          const kneeBell = Math.sin(((ry - hipsY) / Math.max(1, feetY - 25 - hipsY)) * Math.PI);
          x += (rx < neckX ? -1 : 1) * (-squash) * 24 * kneeBell;
        }
        if (Math.abs(armAngle) > 0.002 && ry > shoulderY - 10 && ry < hipsY + 95) {
          if (rx < leftShoulderX - 6) {
            const ang = armAngle * Math.min(1, (leftShoulderX - 6 - rx) / 36);
            const dx = x - leftShoulderX;
            const dy = y - shoulderY;
            x = leftShoulderX + dx * Math.cos(ang) + dy * Math.sin(ang);
            y = shoulderY - dx * Math.sin(ang) + dy * Math.cos(ang);
          } else if (rx > rightShoulderX + 6) {
            const ang = armAngle * Math.min(1, (rx - (rightShoulderX + 6)) / 36);
            const dx = x - rightShoulderX;
            const dy = y - shoulderY;
            x = rightShoulderX + dx * Math.cos(ang) - dy * Math.sin(ang);
            y = shoulderY + dx * Math.sin(ang) + dy * Math.cos(ang);
          }
        }
        bDef[i] = x;
        bDef[i + 1] = y - jumpY;
      }
      this.bodyGrid.geometry.getBuffer('aPosition').update();
    }

    // --- 5. Deform 2.5D Head Mesh (`headGrid`) via Distortion-Free 3D Cranial Ellipsoid ---
    const hRest = this.headGrid.restPositions;
    const hDef = this.headGrid.deformedPositions;
    const headRx = this.manifest.layers.head.width * 0.50;
    const headRy = this.manifest.layers.head.height * 0.50;

    const thetaY = yaw * 0.36;
    const thetaX = pitch * 0.28;
    const cosY = Math.cos(thetaY);
    const sinY = Math.sin(thetaY);
    const cosX = Math.cos(thetaX);
    const sinX = Math.sin(thetaX);

    const rollRad = Math.max(-1, Math.min(1, params.headRoll)) * 0.24 + yaw * 0.035;
    const cosR = Math.cos(rollRad);
    const sinR = Math.sin(rollRad);

    const hScale = headRx / 161.0;
    const rzX = headRx * 0.535;
    const rzY = headRy * 0.445;
    const jawDropPx = Math.max(params.mouthOpen * 5.2, laughW * 4.5) * Math.max(0.65, hScale);
    const cheekLiftPx = Math.max(smileW * 2.2, laughW * 3.2) * Math.max(0.65, hScale);
    const cheekCenterOffset = 38 * hScale;
    const cheekSigmaX2 = Math.max(140, 480 * hScale * hScale);
    const cheekSigmaY2 = Math.max(100, 320 * hScale * hScale);
    const jawSigmaX2 = Math.max(900, 3600 * hScale * hScale);
    const jawSigmaY2 = Math.max(160, 520 * hScale * hScale);

    for (let i = 0; i < hRest.length; i += 2) {
      const rx = hRest[i];
      const ry = hRest[i + 1];

      const u = Math.max(-1.05, Math.min(1.05, (rx - faceX) / headRx));
      const v = Math.max(-1.05, Math.min(1.05, (ry - faceY) / headRy));

      const zDome = Math.cos(u * 1.15) * Math.cos((v - 0.08) * 0.48);
      const bangsRelief = 0.07 * Math.exp(-((u * u) / 0.32 + ((v + 0.28) * (v + 0.28)) / 0.14));
      const zRel = zDome + bangsRelief - 0.66;

      const X = rx - faceX;
      const Y = ry - faceY;
      let Xp = X * cosY + rzX * zRel * sinY - pitch * u * (v - 0.05) * (4.5 * hScale);
      let Yp = Y * cosX + rzY * zRel * sinX;

      if (cheekLiftPx > 0.05 && ry > eyeY + 6 * hScale && ry < mouthY + 18 * hScale) {
        const cheekBand =
          Math.exp(-((ry - (mouthY - 8 * hScale)) ** 2) / cheekSigmaY2) *
          Math.exp(-((Math.abs(rx - faceX) - cheekCenterOffset) ** 2) / cheekSigmaX2);
        Yp -= cheekLiftPx * cheekBand;
      }

      if (jawDropPx > 0.05 && ry > mouthY - 4 * hScale) {
        const jawW =
          Math.exp(-((rx - faceX) ** 2) / jawSigmaX2) *
          Math.exp(-((ry - (mouthY + 18 * hScale)) ** 2) / jawSigmaY2);
        Yp += jawDropPx * jawW;
      }

      const faceGuard = 1 - Math.exp(-((u / 0.54) ** 4 + ((v - 0.18) / 0.68) ** 4));
      const rNorm = Math.hypot(u, v);
      const hairPeriphery = Math.max(0, Math.min(1, (rNorm - 0.62) * 1.4)) * faceGuard;
      Xp += Math.max(-7, Math.min(7, hairOffset.x * 0.6)) * hScale * hairPeriphery;
      Yp += Math.max(-8, Math.min(8, hairOffset.y * 0.6)) * hScale * hairPeriphery;

      const px = faceX + Xp - neckX;
      const py = faceY + Yp - neckY;
      const rotX = neckX + px * cosR - py * sinR;
      const rotY = neckY + px * sinR + py * cosR;

      hDef[i] = rotX + totalNeckDeltaX + yaw * (5.2 * Math.max(0.7, hScale));
      hDef[i + 1] = rotY + totalNeckDeltaY + pitch * (4.8 * Math.max(0.7, hScale));
    }
    this.headGrid.geometry.getBuffer('aPosition').update();

    // --- 6. Update Dynamic Ground Shadow ---
    const heightRatio = Math.min(1, jumpY / 180);
    const legSpread =
      Math.abs(params.leftLeg[0]) + Math.abs(params.rightLeg[0]) > 0.05
        ? 1 + (Math.abs(params.leftLeg[0]) + Math.abs(params.rightLeg[0])) * 0.22
        : 1;
    const crouchScale = squash < 0 ? 1 + (-squash) * 0.28 : 1 - squash * 0.1;
    const shadowScale = Math.max(0.38, (1 - heightRatio * 0.52) * crouchScale * legSpread);
    const shadowAlpha = Math.max(0.10, 0.26 * (1 - heightRatio * 0.62));
    const shadowCenterX = feetX + totalNeckDeltaX * 0.35;
    const shadowCenterY = feetY + 8;

    this.shadowGraphics.clear();
    this.shadowGraphics
      .ellipse(shadowCenterX, shadowCenterY, 138 * shadowScale, 22 * shadowScale)
      .fill({ color: 0x2c2520, alpha: shadowAlpha * 0.55 });
    this.shadowGraphics
      .ellipse(shadowCenterX, shadowCenterY, 96 * shadowScale, 14 * shadowScale)
      .fill({ color: 0x2c2520, alpha: shadowAlpha });

    // --- 7. Optional Debug Wireframe & Rig Landmarks ---
    this.debugGraphics.clear();
    if (this.showWireframe) {
      if (
        this.hasSkeletalLimbs &&
        this.torsoGrid &&
        this.leftArmGrid &&
        this.rightArmGrid &&
        this.leftLegGrid &&
        this.rightLegGrid
      ) {
        this.drawGridWireframe(this.leftLegGrid, 0x2563eb, 0.26);
        this.drawGridWireframe(this.rightLegGrid, 0x2563eb, 0.26);
        this.drawGridWireframe(this.torsoGrid, 0x2f7d6d, 0.28);
        this.drawGridWireframe(this.leftArmGrid, 0xd97706, 0.32);
        this.drawGridWireframe(this.rightArmGrid, 0xd97706, 0.32);
      } else if (this.bodyGrid) {
        this.drawGridWireframe(this.bodyGrid, 0x2f7d6d, 0.28);
      }
      this.drawGridWireframe(this.headGrid, 0xd95f3b, 0.36);
    }
    this.labelContainer.visible = this.showLandmarks;
    if (this.showLandmarks) {
      // 1. Draw bone chain segments with high-contrast white halo
      for (const [aKey, bKey, boneColor] of BONE_SEGMENTS) {
        const aPt = lm[aKey];
        const bPt = lm[bKey];
        if (!aPt || !bPt) continue;
        const ay = aPt[1] - (aKey === 'feetGround' ? 0 : jumpY);
        const by = bPt[1] - (bKey === 'feetGround' ? 0 : jumpY);
        this.debugGraphics
          .moveTo(aPt[0], ay)
          .lineTo(bPt[0], by)
          .stroke({ color: 0xffffff, width: 5.5, alpha: 0.72 });
        this.debugGraphics
          .moveTo(aPt[0], ay)
          .lineTo(bPt[0], by)
          .stroke({ color: boneColor, width: 3.0, alpha: 0.92 });
      }

      // 2. Draw draggable joint handles + position Chinese labels
      for (const [key, coords] of Object.entries(lm)) {
        const txt = this.labelTexts[key];
        if (!coords) {
          if (txt) txt.visible = false;
          continue;
        }
        const [lx, ly] = coords;
        const drawY = ly - (key === 'feetGround' ? 0 : jumpY);
        const isPivot =
          key === 'neckPivot' ||
          key === 'feetGround' ||
          key.endsWith('Shoulder') ||
          key.endsWith('Hip');
        const isFace = key === 'leftEye' || key === 'rightEye' || key === 'mouth';
        const isSelected = key === this.activeLandmark || key === this.hoveredLandmark;

        if (isSelected) {
          this.debugGraphics
            .circle(lx, drawY, 14)
            .fill({ color: 0xf59e0b, alpha: 0.28 })
            .stroke({ color: 0xfbbf24, width: 2.5 });
        }

        const fillCol = isSelected
          ? 0xf59e0b
          : isPivot
            ? 0xe0522d
            : isFace
              ? 0x8b5cf6
              : 0x1d7a85;
        const rad = isSelected ? 9.5 : isPivot ? 8.0 : 6.2;
        this.debugGraphics
          .circle(lx, drawY, rad)
          .fill({ color: fillCol, alpha: 0.96 })
          .stroke({ color: 0xffffff, width: 2.4 });

        if (txt) {
          txt.visible = true;
          const dx = key.startsWith('left') ? -24 : key.startsWith('right') ? 24 : 0;
          const dy = dx === 0 ? (key === 'mouth' || key === 'feetGround' ? 18 : -17) : 0;
          txt.x = lx + dx;
          txt.y = drawY + dy;
        }
      }
    }
  }

  private drawGridWireframe(grid: GridData, color: number, alpha: number): void {
    const pos = grid.deformedPositions;
    const idx = grid.indices;
    for (let i = 0; i < idx.length; i += 3) {
      const a = idx[i] * 2;
      const b = idx[i + 1] * 2;
      const c = idx[i + 2] * 2;
      this.debugGraphics
        .moveTo(pos[a], pos[a + 1])
        .lineTo(pos[b], pos[b + 1])
        .lineTo(pos[c], pos[c + 1])
        .lineTo(pos[a], pos[a + 1])
        .stroke({ color, width: 1, alpha });
    }
  }

  public destroy(): void {
    this.headGrid.geometry.destroy();
    this.bodyGrid?.geometry.destroy();
    this.torsoGrid?.geometry.destroy();
    this.leftArmGrid?.geometry.destroy();
    this.rightArmGrid?.geometry.destroy();
    this.leftLegGrid?.geometry.destroy();
    this.rightLegGrid?.geometry.destroy();
    this.root.destroy({ children: true });
  }
}
