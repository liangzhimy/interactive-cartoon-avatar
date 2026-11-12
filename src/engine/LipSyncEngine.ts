import type { SpeakOptions, SpeechMode, VisemeWeights } from './types';

export interface LipSyncFrame {
  mouthOpen: number;
  viseme: VisemeWeights;
  headBob: number;
  active: boolean;
  level: number;
}

/**
 * Multi-Modal Speech & Real-Time Lip-Sync Engine:
 * 1. `browser` mode : Speaks natural Chinese/English via `window.speechSynthesis` paired
 *                     with syllable-accurate vowel viseme curves (`A`, `O`, `E`).
 * 2. `synth` / `edge`: Requests audio from `/api/tts` and analyzes real-time FFT spectrum
 *                     via Web Audio `AnalyserNode` (RMS volume + formant band ratios).
 * 3. `mic` mode     : Streams live microphone audio (`getUserMedia`) into `AnalyserNode`
 *                     without speaker echo so your own voice drives the cartoon character!
 * 4. `audio` file   : Plays any uploaded or external `.mp3`/`.wav` file with FFT lip-sync.
 */
export class LipSyncEngine {
  private audioCtx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private freqBins: Uint8Array<ArrayBuffer> | null = null;
  private timeBins: Uint8Array<ArrayBuffer> | null = null;

  private activeSource: AudioBufferSourceNode | MediaStreamAudioSourceNode | null = null;
  private micStream: MediaStream | null = null;
  private isMicActive = false;
  private isAudioPlaying = false;

  // Browser speechSynthesis prosody state
  private isBrowserSpeaking = false;
  private browserSyllables: Array<{ start: number; end: number; v: VisemeWeights }> = [];
  private browserStartTime = 0;
  private browserTotalDuration = 0;

  // Smoothed state
  private smoothOpen = 0;
  private smoothViseme: VisemeWeights = { a: 0.5, o: 0.25, e: 0.25 };
  private talkClock = 0;

  public onStatusChange?: (status: { speaking: boolean; mic: boolean; mode: string }) => void;

  private ensureAudioContext(): AudioContext {
    if (!this.audioCtx) {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioCtx = new Ctx();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.55;
      this.freqBins = new Uint8Array(this.analyser.frequencyBinCount);
      this.timeBins = new Uint8Array(this.analyser.fftSize);
    }
    if (this.audioCtx.state === 'suspended') {
      void this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  public stopSpeech(): void {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    this.isBrowserSpeaking = false;

    if (this.activeSource && !this.isMicActive) {
      try {
        (this.activeSource as AudioBufferSourceNode).stop?.();
      } catch {
        // ignore already stopped
      }
      this.activeSource.disconnect();
      this.activeSource = null;
    }
    this.isAudioPlaying = false;
    this.notifyStatus('idle');
  }

  public async speak(text: string, options: SpeakOptions = {}): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) return;

    this.stopSpeech();
    const mode: SpeechMode = options.mode ?? 'browser';

    if (mode === 'synth' || mode === 'edge') {
      try {
        await this.speakViaBackendAudio(trimmed, mode, options.voice);
        return;
      } catch {
        // Graceful fallback to browser speechSynthesis if backend or Edge-TTS is unreachable
      }
    }

    await this.speakViaBrowserTTS(trimmed, options);
  }

  private async speakViaBackendAudio(text: string, mode: SpeechMode, voice?: string): Promise<void> {
    const ctx = this.ensureAudioContext();
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        mode,
        voice: voice || 'zh-CN-XiaoyiNeural',
      }),
    });
    if (!res.ok) {
      throw new Error(`TTS HTTP ${res.status}`);
    }
    const arrayBuf = await res.arrayBuffer();
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
    await this.playAudioBuffer(audioBuf, mode);
  }

  public async speakAudioFile(fileOrUrl: File | string): Promise<void> {
    this.stopSpeech();
    const ctx = this.ensureAudioContext();
    const arrayBuf =
      typeof fileOrUrl === 'string'
        ? await (await fetch(fileOrUrl)).arrayBuffer()
        : await fileOrUrl.arrayBuffer();
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
    await this.playAudioBuffer(audioBuf, 'audio-file');
  }

  private playAudioBuffer(audioBuf: AudioBuffer, label: string): Promise<void> {
    const ctx = this.ensureAudioContext();
    return new Promise<void>((resolve) => {
      const src = ctx.createBufferSource();
      src.buffer = audioBuf;
      src.connect(this.analyser!);
      this.analyser!.connect(ctx.destination);

      this.activeSource = src;
      this.isAudioPlaying = true;
      this.notifyStatus(label);

      src.onended = () => {
        if (this.activeSource === src) {
          this.isAudioPlaying = false;
          this.activeSource = null;
          this.notifyStatus('idle');
        }
        resolve();
      };
      src.start(0);
    });
  }

  private buildSyllableTimeline(text: string, rate: number): {
    timeline: Array<{ start: number; end: number; v: VisemeWeights }>;
    total: number;
  } {
    const timeline: Array<{ start: number; end: number; v: VisemeWeights }> = [];
    let cursor = 0.04;
    const speedScale = 1 / Math.max(0.5, Math.min(2.0, rate));

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (/\s/.test(ch)) {
        cursor += 0.04 * speedScale;
        continue;
      }
      if (/[，,、；;：:]/.test(ch)) {
        cursor += 0.18 * speedScale;
        continue;
      }
      if (/[。.！!？?]/.test(ch)) {
        cursor += 0.26 * speedScale;
        continue;
      }

      const code = ch.charCodeAt(0);
      const isCJK = code >= 0x4e00 && code <= 0x9fff;
      const dur = (isCJK ? 0.175 : 0.095) * speedScale;

      // Map character/vowel to natural A / O / E viseme weights
      let v: VisemeWeights;
      const lower = ch.toLowerCase();
      if ('aoɑ'.includes(lower) || (isCJK && code % 3 === 0)) {
        v = { a: 0.82, o: 0.10, e: 0.08 };
      } else if ('ouwu'.includes(lower) || (isCJK && code % 3 === 1)) {
        v = { a: 0.12, o: 0.78, e: 0.10 };
      } else {
        v = { a: 0.15, o: 0.10, e: 0.75 };
      }

      timeline.push({ start: cursor, end: cursor + dur * 0.84, v });
      cursor += dur;
    }
    return { timeline, total: Math.max(0.35, cursor + 0.12) };
  }

  private speakViaBrowserTTS(text: string, options: SpeakOptions): Promise<void> {
    const rate = options.rate ?? 1.02;
    const pitch = options.pitch ?? 1.12;
    const { timeline, total } = this.buildSyllableTimeline(text, rate);

    this.browserSyllables = timeline;
    this.browserStartTime = performance.now() / 1000;
    this.browserTotalDuration = total;
    this.isBrowserSpeaking = true;
    this.notifyStatus('browser-tts');

    return new Promise<void>((resolve) => {
      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        this.isBrowserSpeaking = false;
        this.notifyStatus('idle');
        resolve();
      };

      // Fallback timer in case headless/muted browser doesn't fire utterance.onend
      const fallbackMs = (total + 0.8) * 1000;
      const safetyTimer = window.setTimeout(finish, fallbackMs);

      if (typeof window.speechSynthesis === 'undefined') {
        globalThis.setTimeout(() => {
          clearTimeout(safetyTimer);
          finish();
        }, total * 1000);
        return;
      }

      const utter = new SpeechSynthesisUtterance(text);
      const hasCJK = /[\u4e00-\u9fff]/.test(text);
      utter.lang = hasCJK ? 'zh-CN' : 'en-US';
      utter.rate = rate;
      utter.pitch = pitch;

      // Pick the best matching natural voice available in the user's browser
      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(
        (v) =>
          (hasCJK ? v.lang.startsWith('zh') : v.lang.startsWith('en')) &&
          /Xiaoxiao|Xiaoyi|Tingting|Meijia|Natural|Google/i.test(v.name)
      ) || voices.find((v) => (hasCJK ? v.lang.startsWith('zh') : v.lang.startsWith('en')));
      if (preferred) {
        utter.voice = preferred;
      }

      utter.onstart = () => {
        this.browserStartTime = performance.now() / 1000;
      };
      utter.onend = () => {
        clearTimeout(safetyTimer);
        finish();
      };
      utter.onerror = () => {
        // Keep syllable animation running for visual feedback even if browser audio is blocked
        window.setTimeout(() => {
          clearTimeout(safetyTimer);
          finish();
        }, Math.max(200, total * 1000 - (performance.now() / 1000 - this.browserStartTime) * 1000));
      };

      window.speechSynthesis.speak(utter);
    });
  }

  public async startMicrophone(): Promise<void> {
    this.stopSpeech();
    const ctx = this.ensureAudioContext();
    // Disconnect analyser from destination so mic input does NOT echo through speakers!
    try {
      this.analyser!.disconnect();
    } catch {
      // ignore
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.micStream = stream;
    const micSrc = ctx.createMediaStreamSource(stream);
    micSrc.connect(this.analyser!);
    this.activeSource = micSrc;
    this.isMicActive = true;
    this.notifyStatus('microphone');
  }

  public stopMicrophone(): void {
    if (this.micStream) {
      for (const track of this.micStream.getTracks()) {
        track.stop();
      }
      this.micStream = null;
    }
    if (this.isMicActive && this.activeSource) {
      this.activeSource.disconnect();
      this.activeSource = null;
    }
    this.isMicActive = false;
    this.notifyStatus('idle');
  }

  public get isMicRunning(): boolean {
    return this.isMicActive;
  }

  private notifyStatus(mode: string): void {
    this.onStatusChange?.({
      speaking: this.isAudioPlaying || this.isBrowserSpeaking,
      mic: this.isMicActive,
      mode,
    });
  }

  public step(dt: number): LipSyncFrame {
    this.talkClock += dt;
    let targetOpen = 0;
    let targetV: VisemeWeights = this.smoothViseme;

    if ((this.isAudioPlaying || this.isMicActive) && this.analyser && this.freqBins && this.timeBins) {
      this.analyser.getByteTimeDomainData(this.timeBins);
      this.analyser.getByteFrequencyData(this.freqBins);

      // 1. Compute RMS volume envelope from time-domain waveform
      let sumSq = 0;
      for (let i = 0; i < this.timeBins.length; i++) {
        const centered = (this.timeBins[i] - 128) / 128;
        sumSq += centered * centered;
      }
      const rms = Math.sqrt(sumSq / this.timeBins.length);
      // Noise gate + soft knee gain
      const gated = Math.max(0, rms - (this.isMicActive ? 0.022 : 0.012));
      targetOpen = Math.min(1, Math.pow(gated * (this.isMicActive ? 5.2 : 4.4), 0.78));

      // 2. Compute Low / Mid / High formant band energies to classify O vs A vs E!
      const sr = this.audioCtx?.sampleRate ?? 48000;
      const binHz = sr / this.analyser.fftSize;
      const binIdx = (hz: number) => Math.min(this.freqBins!.length - 1, Math.max(1, Math.round(hz / binHz)));

      const bandEnergy = (f0: number, f1: number) => {
        const i0 = binIdx(f0);
        const i1 = Math.max(i0 + 1, binIdx(f1));
        let s = 0;
        for (let i = i0; i <= i1; i++) s += this.freqBins![i];
        return s / (i1 - i0 + 1) / 255;
      };

      const lowO = bandEnergy(180, 680);   // Strong F1+F2 proximity -> 'O / U'
      const midA = bandEnergy(700, 1750);  // Strong open F1+F2 -> 'A'
      const highE = bandEnergy(1800, 3800); // High F2 spread -> 'E / I'

      if (targetOpen > 0.05) {
        targetV = {
          o: Math.max(0.05, lowO * 1.15),
          a: Math.max(0.05, midA * 1.45),
          e: Math.max(0.05, highE * 1.65),
        };
      }
    } else if (this.isBrowserSpeaking) {
      const now = performance.now() / 1000 - this.browserStartTime;
      if (now <= this.browserTotalDuration) {
        const syl = this.browserSyllables.find((s) => now >= s.start && now <= s.end);
        if (syl) {
          const u = (now - syl.start) / Math.max(0.02, syl.end - syl.start);
          targetOpen = Math.sin(u * Math.PI) * 0.88;
          targetV = syl.v;
        } else {
          targetOpen = 0.04;
        }
      } else {
        this.isBrowserSpeaking = false;
        this.notifyStatus('idle');
      }
    }

    // Fast attack (30ms) so consonants/vowels pop crisply, smooth release (85ms)
    const alpha = targetOpen > this.smoothOpen ? Math.min(1, dt * 32) : Math.min(1, dt * 13);
    this.smoothOpen += (targetOpen - this.smoothOpen) * alpha;

    const vAlpha = Math.min(1, dt * 16);
    this.smoothViseme = {
      a: this.smoothViseme.a + (targetV.a - this.smoothViseme.a) * vAlpha,
      o: this.smoothViseme.o + (targetV.o - this.smoothViseme.o) * vAlpha,
      e: this.smoothViseme.e + (targetV.e - this.smoothViseme.e) * vAlpha,
    };

    // Subtle natural head cadence while actively talking
    const active = this.smoothOpen > 0.03 || this.isBrowserSpeaking || this.isAudioPlaying;
    const headBob = active ? Math.sin(this.talkClock * 7.5) * this.smoothOpen * 0.11 : 0;

    return {
      mouthOpen: this.smoothOpen,
      viseme: this.smoothViseme,
      headBob,
      active,
      level: this.smoothOpen,
    };
  }

  public destroy(): void {
    this.stopSpeech();
    this.stopMicrophone();
    if (this.audioCtx) {
      void this.audioCtx.close();
      this.audioCtx = null;
    }
  }
}
