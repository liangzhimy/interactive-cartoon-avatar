import { Application, Assets, FederatedPointerEvent, Texture } from 'pixi.js';
import { ActionController } from './ActionController';
import { LipSyncEngine } from './LipSyncEngine';
import { MeshRig } from './MeshRig';
import type {
  AvatarLandmarks,
  AvatarManifest,
  AvatarParameters,
  CheerOptions,
  JumpOptions,
  KickOptions,
  LaughOptions,
  LimbPoseInput,
  NodOptions,
  ShakeOptions,
  SmileOptions,
  SpeakOptions,
  WalkOptions,
  WaveOptions,
} from './types';

export interface CartoonAvatarConfig {
  container: HTMLElement;
  manifestUrl: string;
  backgroundAlpha?: number;
  mouseFollow?: boolean;
  onFrame?: (params: AvatarParameters, audioLevel: number) => void;
  onSpeechStatus?: (status: { speaking: boolean; mic: boolean; mode: string }) => void;
  onLandmarkChange?: (landmarks: AvatarLandmarks, changedKey: string) => void;
}

/**
 * Public Programmable Cartoon Avatar SDK (`CartoonAvatar`):
 * Turns an `AvatarBundle` (compiled from a single AI-generated cartoon illustration)
 * into a 60fps interactive, expressive web character with 2.5D head orientation,
 * lip-sync speech, and 4-limb skeletal gestures controllable via clean JS/TS methods.
 */
export class CartoonAvatar {
  private container: HTMLElement;
  private app: Application | null = null;
  private rig: MeshRig | null = null;
  private manifest: AvatarManifest | null = null;
  private manifestUrl: string;

  public readonly actions = new ActionController();
  public readonly lipSync = new LipSyncEngine();

  public mouseFollow = false;
  public calibrationMode = false;
  public mirrorCalibration = true;
  private savedIdleBeforeCalibration = true;
  private draggingLandmark: string | null = null;
  private resizeObserver: ResizeObserver | null = null;

  private onFrameCallback?: (params: AvatarParameters, audioLevel: number) => void;
  public onLandmarkChange?: (landmarks: AvatarLandmarks, changedKey: string) => void;

  constructor(config: CartoonAvatarConfig) {
    this.container = config.container;
    this.manifestUrl = config.manifestUrl;
    this.mouseFollow = config.mouseFollow ?? false;
    this.onFrameCallback = config.onFrame;
    this.onLandmarkChange = config.onLandmarkChange;
    this.lipSync.onStatusChange = config.onSpeechStatus;
  }

  public async init(): Promise<void> {
    const app = new Application();
    const initW = Math.max(320, this.container.clientWidth || 560);
    const initH = Math.max(380, this.container.clientHeight || 560);
    await app.init({
      preference: 'webgl',
      width: initW,
      height: initH,
      resizeTo: this.container,
      backgroundAlpha: 0,
      antialias: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      autoDensity: true,
    });
    this.container.innerHTML = '';
    this.container.appendChild(app.canvas);
    this.app = app;

    await this.loadBundle(this.manifestUrl);

    app.stage.eventMode = 'static';
    app.stage.hitArea = app.screen;

    app.stage.on('pointermove', (e: FederatedPointerEvent) => this.handlePointerMove(e));
    app.stage.on('pointerdown', (e: FederatedPointerEvent) => this.handlePointerDown(e));
    const endDrag = () => {
      this.draggingLandmark = null;
      if (this.rig) this.rig.activeLandmark = null;
      if (this.app) {
        this.app.canvas.style.cursor =
          this.calibrationMode && this.rig?.hoveredLandmark ? 'grab' : 'default';
      }
    };
    app.stage.on('pointerup', endDrag);
    app.stage.on('pointerupoutside', endDrag);

    this.resizeObserver = new ResizeObserver(() => {
      this.app?.resize();
      this.fitStage();
    });
    this.resizeObserver.observe(this.container);
    requestAnimationFrame(() => {
      this.app?.resize();
      this.fitStage();
    });

    app.ticker.add((ticker) => {
      const dt = ticker.deltaMS / 1000;
      const lip = this.lipSync.step(dt);
      this.actions.speechMouthOpen = lip.mouthOpen;
      this.actions.speechViseme = lip.viseme;
      this.actions.speechHeadBob = lip.headBob;

      const params = this.actions.step(dt);
      this.rig?.update(params, dt);
      this.onFrameCallback?.(params, lip.level);
    });
  }

  public async loadBundle(manifestUrl: string): Promise<AvatarManifest> {
    this.manifestUrl = manifestUrl;
    const cacheBust = `?t=${Date.now()}`;
    const res = await fetch(manifestUrl + cacheBust);
    if (!res.ok) {
      throw new Error(`Failed to load avatar manifest: ${manifestUrl}`);
    }
    const manifest = (await res.json()) as AvatarManifest;
    this.manifest = manifest;

    const baseUrl = manifestUrl.substring(0, manifestUrl.lastIndexOf('/') + 1);
    const loadTex = async (relPath: string): Promise<Texture> => {
      const texRes = await fetch(baseUrl + relPath + cacheBust);
      if (!texRes.ok) {
        throw new Error(`Failed to fetch texture: ${baseUrl + relPath}`);
      }
      const blob = await texRes.blob();
      if (typeof createImageBitmap === 'function') {
        const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none' });
        return Texture.from(bitmap);
      }
      return await Assets.load<Texture>(baseUrl + relPath);
    };

    const loadOpt = async (spec?: { texture: string }): Promise<Texture | undefined> =>
      spec ? loadTex(spec.texture) : undefined;

    const [headTex, bodyTex, torsoTex, leftArmTex, rightArmTex, leftLegTex, rightLegTex] =
      await Promise.all([
        loadTex(manifest.layers.head.texture),
        loadTex(manifest.layers.body.texture),
        loadOpt(manifest.layers.torso),
        loadOpt(manifest.layers.leftArm),
        loadOpt(manifest.layers.rightArm),
        loadOpt(manifest.layers.leftLeg),
        loadOpt(manifest.layers.rightLeg),
      ]);

    const patchTextures: Record<string, Texture> = {};
    await Promise.all(
      Object.entries(manifest.patches).map(async ([key, rel]) => {
        if (rel) {
          patchTextures[key] = await loadTex(rel);
        }
      })
    );

    const prevWire = this.rig?.showWireframe ?? false;
    const prevLm = this.rig?.showLandmarks ?? false;
    if (this.rig && this.app) {
      this.app.stage.removeChild(this.rig.root);
      this.rig.destroy();
    }

    this.rig = new MeshRig(manifest, {
      head: headTex,
      body: bodyTex,
      torso: torsoTex,
      leftArm: leftArmTex,
      rightArm: rightArmTex,
      leftLeg: leftLegTex,
      rightLeg: rightLegTex,
      patches: patchTextures,
    });
    this.rig.showWireframe = prevWire;
    this.rig.showLandmarks = prevLm;

    this.app?.stage.addChild(this.rig.root);
    this.actions.resetPose();
    this.fitStage();
    return manifest;
  }

  private fitStage(): void {
    if (!this.app || !this.rig || !this.manifest) return;
    const vw = Math.max(240, this.app.screen.width || this.container.clientWidth || 560);
    const vh = Math.max(320, this.app.screen.height || this.container.clientHeight || 560);
    this.app.stage.hitArea = this.app.screen;

    const bounds = this.manifest.characterBounds;
    // Generous horizontal & vertical margin so high waves, cheers, jumps, and kicks never clip!
    const paddedHeight = bounds.height + 230;
    const paddedWidth = bounds.width + 290;
    const scale = Math.max(0.15, Math.min(vw / paddedWidth, vh / paddedHeight) * 0.95);

    this.rig.root.scale.set(scale);
    const charCenterX = bounds.x + bounds.width * 0.5;
    const charBottomY = bounds.y + bounds.height;

    this.rig.root.x = vw * 0.5 - charCenterX * scale;
    this.rig.root.y = vh * 0.91 - charBottomY * scale;
  }

  private findNearestLandmark(localX: number, localY: number): string | null {
    if (!this.rig) return null;
    const lm = this.rig.getLandmarks();
    let bestKey: string | null = null;
    let bestDist = 28;
    for (const [k, coords] of Object.entries(lm)) {
      if (!coords) continue;
      const [lx, ly] = coords;
      const d = Math.hypot(localX - lx, localY - ly);
      if (d < bestDist) {
        bestDist = d;
        bestKey = k;
      }
    }
    return bestKey;
  }

  private handlePointerDown(e: FederatedPointerEvent): void {
    if (!this.calibrationMode || !this.rig) return;
    const local = this.rig.root.toLocal(e.global);
    const bestKey = this.findNearestLandmark(local.x, local.y);
    this.draggingLandmark = bestKey;
    this.rig.activeLandmark = bestKey;
    if (this.app) {
      this.app.canvas.style.cursor = bestKey ? 'grabbing' : 'default';
    }
  }

  private handlePointerMove(e: FederatedPointerEvent): void {
    if (this.calibrationMode && this.rig) {
      const local = this.rig.root.toLocal(e.global);
      if (this.draggingLandmark) {
        const key = this.draggingLandmark;
        this.rig.updateLandmark(key, local.x, local.y);

        if (this.mirrorCalibration) {
          const lm = this.rig.getLandmarks();
          const axisX = lm.neckPivot?.[0] ?? (this.manifest?.canvasSize.width ?? 896) * 0.5;
          let twinKey: string | null = null;
          if (key.startsWith('left')) {
            twinKey = 'right' + key.slice(4);
          } else if (key.startsWith('right')) {
            twinKey = 'left' + key.slice(5);
          }
          if (twinKey && lm[twinKey]) {
            this.rig.updateLandmark(twinKey, 2 * axisX - local.x, local.y);
          }
        }

        // Keep center `hips` / `knees` synced if left/right hips or knees move
        const lmNow = this.rig.getLandmarks();
        if ((key === 'leftHip' || key === 'rightHip') && lmNow.leftHip && lmNow.rightHip) {
          this.rig.updateLandmark(
            'hips',
            0.5 * (lmNow.leftHip[0] + lmNow.rightHip[0]),
            0.5 * (lmNow.leftHip[1] + lmNow.rightHip[1])
          );
        }
        if ((key === 'leftKnee' || key === 'rightKnee') && lmNow.leftKnee && lmNow.rightKnee) {
          this.rig.updateLandmark(
            'knees',
            0.5 * (lmNow.leftKnee[0] + lmNow.rightKnee[0]),
            0.5 * (lmNow.leftKnee[1] + lmNow.rightKnee[1])
          );
        }

        this.onLandmarkChange?.(this.rig.getLandmarks(), key);
        return;
      }

      const hovered = this.findNearestLandmark(local.x, local.y);
      this.rig.hoveredLandmark = hovered;
      if (this.app) {
        this.app.canvas.style.cursor = hovered ? 'grab' : 'default';
      }
      return;
    }

    if (this.mouseFollow && this.app) {
      const nx = (e.global.x / Math.max(1, this.app.screen.width)) * 2 - 1;
      const ny = (e.global.y / Math.max(1, this.app.screen.height)) * 2 - 1;
      this.lookAt(nx * 0.85, ny * 0.65);
    }
  }

  // ==================== High-Level Programmable SDK Methods ====================

  /** Blink once (or double blink when `doubleBlink = true`). */
  public blink(doubleBlink = false): Promise<void> {
    return this.actions.blink(doubleBlink);
  }

  /** Warm cheerful smile. */
  public smile(options?: SmileOptions): Promise<void> {
    return this.actions.smile(options);
  }

  /** Big joyful open-mouth laugh with happy crescent eyes and forearm bounce! */
  public laugh(options?: LaughOptions): Promise<void> {
    return this.actions.laugh(options);
  }

  /** Raise one hand and wave forearm + wrist warmly (`hand: 'right' | 'left'`). */
  public wave(options?: WaveOptions): Promise<void> {
    return this.actions.wave(options);
  }

  /** Celebratory two-hand overhead cheer with happy eyes and bounce! */
  public cheer(options?: CheerOptions): Promise<void> {
    return this.actions.cheer(options);
  }

  /** Alternating energetic marching/walking steps with opposite arm swings! */
  public walk(options?: WalkOptions): Promise<void> {
    return this.actions.walk(options);
  }

  /** Playful leg kick (`leg: 'right' | 'left'`) with balancing arms! */
  public kick(options?: KickOptions): Promise<void> {
    return this.actions.kick(options);
  }

  /** Programmatically hold custom 4-limb joint angles in radians (`[proximal, middle, distal]`). */
  public setLimbPose(pose: LimbPoseInput): void {
    this.actions.setLimbPose(pose);
  }

  /** Nod head up & down affirmatively. */
  public nod(options?: NodOptions): Promise<void> {
    return this.actions.nod(options);
  }

  /** Shake head left & right playfully. */
  public shakeHead(options?: ShakeOptions): Promise<void> {
    return this.actions.shakeHead(options);
  }

  /** Continuous 2.5D head orientation (`yaw`, `pitch`, `roll` in `[-1..+1]`). */
  public turnHead(yaw: number, pitch = 0, roll = 0): void {
    this.actions.turnHead(yaw, pitch, roll);
  }

  /** Point head/gaze toward normalized screen target `(x, y)` in `[-1..+1]`. */
  public lookAt(normX: number, normY: number): void {
    this.actions.turnHead(normX, normY, normX * 0.22);
  }

  /** Four-stage physical jump with squat anticipation, parabolic flight, and landing bounce! */
  public jump(options?: JumpOptions): Promise<void> {
    return this.actions.jump(options);
  }

  /** Speak text aloud with real-time synchronized mouth visemes! */
  public speak(text: string, options?: SpeakOptions): Promise<void> {
    return this.lipSync.speak(text, options);
  }

  /** Drive mouth in real-time from an uploaded or remote audio file. */
  public speakAudio(fileOrUrl: File | string): Promise<void> {
    return this.lipSync.speakAudioFile(fileOrUrl);
  }

  /** Start/stop live microphone lip-sync puppeteering. */
  public async toggleMicrophone(): Promise<boolean> {
    if (this.lipSync.isMicRunning) {
      this.lipSync.stopMicrophone();
      return false;
    }
    await this.lipSync.startMicrophone();
    return true;
  }

  public stopSpeaking(): void {
    this.lipSync.stopSpeech();
  }

  public reset(): void {
    this.lipSync.stopSpeech();
    this.actions.resetPose();
  }

  public setWireframe(visible: boolean): void {
    if (this.rig) this.rig.showWireframe = visible;
  }

  public setCalibrationMode(enabled: boolean): void {
    if (enabled && !this.calibrationMode) {
      this.savedIdleBeforeCalibration = this.actions.idleEnabled;
      this.actions.idleEnabled = false;
      this.actions.resetPose();
    } else if (!enabled && this.calibrationMode) {
      this.actions.idleEnabled = this.savedIdleBeforeCalibration;
      this.draggingLandmark = null;
      if (this.rig) {
        this.rig.hoveredLandmark = null;
        this.rig.activeLandmark = null;
      }
      if (this.app) {
        this.app.canvas.style.cursor = 'default';
      }
    }
    this.calibrationMode = enabled;
    if (this.rig) this.rig.showLandmarks = enabled;
  }

  public getManifest(): AvatarManifest | null {
    return this.manifest;
  }

  public getLandmarks(): AvatarLandmarks | null {
    return this.rig?.getLandmarks() ?? null;
  }

  public destroy(): void {
    this.resizeObserver?.disconnect();
    this.lipSync.destroy();
    this.rig?.destroy();
    this.app?.destroy(true);
  }
}
