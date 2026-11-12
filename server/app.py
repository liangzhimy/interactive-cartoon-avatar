#!/usr/bin/env python3
"""
FastAPI Backend for Interactive Cartoon Avatar Studio (`server/app.py`)

Provides:
  - `GET    /api/health`                  : Health & capabilities check
  - `GET    /api/avatars`                 : List compiled 6-layer `AvatarBundle`s (built-ins + custom imports)
  - `POST   /api/tts`                     : Synthesize speech audio (`edge-tts` or formant-vowel WAV synthesis)
  - `POST   /api/import-character`        : Universal single-image character rigging (`universal_rig.py`)
  - `PUT    /api/avatars/{id}/landmarks`  : Save calibrated joint/facial landmarks to `manifest.json`
  - `POST   /api/avatars/{id}/rerig`      : Re-segment 6 layers & facial patches around calibrated landmarks / outfit mode
  - `GET    /api/avatars/{id}/export`     : Download a self-contained `.zip` of the `AvatarBundle` + SDK usage guide
  - `DELETE /api/avatars/{id}`            : Delete a custom imported character bundle
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import tempfile
import time
import wave
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import edge_tts
import edge_tts.communicate as ec
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from server.tools.universal_rig import build_universal_bundle

ROOT_DIR = Path(__file__).resolve().parent.parent
AVATARS_DIR = ROOT_DIR / "public" / "avatars"
BUILTIN_ORDER = ["sunny", "male_teacher", "female_teacher"]
BUILTIN_IDS = set(BUILTIN_ORDER)

if os.environ.get("SSL_CERT_FILE") and os.path.exists(os.environ["SSL_CERT_FILE"]):
    try:
        ec._SSL_CTX.load_verify_locations(cafile=os.environ["SSL_CERT_FILE"])
    except Exception:
        pass

app = FastAPI(title="Interactive Cartoon Avatar Universal Rigging API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoyiNeural"
    mode: str = "auto"  # "edge" | "synth" | "auto"
    rate: str = "+0%"
    pitch: str = "+2Hz"


class LandmarksUpdate(BaseModel):
    landmarks: Dict[str, List[float]]


class RerigRequest(BaseModel):
    outfitMode: str = "auto"  # "auto" | "pants" | "skirt"
    landmarks: Optional[Dict[str, List[float]]] = None


def _sanitize_avatar_id(avatar_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", avatar_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid avatar id")
    return safe_id


def synthesize_cute_formant_wav(text: str, base_f0: float = 265.0) -> bytes:
    """
    Generate an expressive cartoon formant-vowel speech WAV stream (24kHz mono PCM)
    whose F1/F2 formants vary across syllables (`A`, `O`, `E`, `I`, `U`) so Web Audio
    FFT lip-sync produces rich, crisp real-waveform mouth movement in any network.
    """
    sr = 24000
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        clean = "你好呀！"

    vowel_formants = [
        (820.0, 1320.0, 1.00),  # 'A' - wide open
        (460.0, 920.0, 0.85),   # 'O' - rounded
        (530.0, 1880.0, 0.82),  # 'E' - mid smile/talk
        (340.0, 2250.0, 0.72),  # 'I' - bright narrow
        (380.0, 860.0, 0.78),   # 'U' - rounded small
    ]

    segments: list[np.ndarray] = []
    idx = 0
    for ch in clean[:160]:
        if ch in "，,、；;":
            segments.append(np.zeros(int(sr * 0.14), dtype=np.float32))
            continue
        if ch in "。.！!？?\n":
            segments.append(np.zeros(int(sr * 0.22), dtype=np.float32))
            continue
        if ch == " ":
            segments.append(np.zeros(int(sr * 0.06), dtype=np.float32))
            continue

        code = ord(ch)
        f1, f2, amp = vowel_formants[(code + idx * 3) % len(vowel_formants)]
        dur = 0.135 if code > 127 else 0.085
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        tone_type = (code + idx) % 4
        if tone_type == 0:
            f0 = base_f0 * 1.08 * np.ones_like(t)
        elif tone_type == 1:
            f0 = base_f0 * np.linspace(0.92, 1.18, n, dtype=np.float32)
        elif tone_type == 2:
            f0 = base_f0 * (0.95 - 0.10 * np.sin(np.pi * t / dur) + 0.14 * (t / dur))
        else:
            f0 = base_f0 * np.linspace(1.18, 0.86, n, dtype=np.float32)

        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        glottal = (
            0.55 * np.sin(phase)
            + 0.30 * np.sin(2.0 * phase)
            + 0.15 * np.sin(3.0 * phase)
        )
        formant_1 = np.sin(2.0 * np.pi * f1 * t) * (0.5 + 0.5 * np.sin(phase))
        formant_2 = np.sin(2.0 * np.pi * f2 * t) * (0.5 + 0.5 * np.cos(phase))

        attack = np.clip(t / 0.018, 0.0, 1.0)
        release = np.clip((dur - t) / 0.038, 0.0, 1.0)
        env = attack * release * amp

        syllable = (0.48 * glottal + 0.36 * formant_1 + 0.16 * formant_2) * env
        gap = np.zeros(int(sr * 0.022), dtype=np.float32)
        segments.append(syllable.astype(np.float32))
        segments.append(gap)
        idx += 1

    audio = np.concatenate(segments) if segments else np.zeros(int(sr * 0.2), dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 1e-5:
        audio = audio / peak * 0.72

    pcm16 = np.round(audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "interactive-cartoon-avatar",
        "version": "1.2.0",
        "capabilities": ["universal-6layer-rigging", "pants-and-skirt-modes", "rerig", "zip-export", "tts"],
        "voices": [
            {"id": "browser-zh", "label": "浏览器自然语音 (推荐实时发声)", "mode": "browser"},
            {"id": "synth-cute", "label": "卡通萌音合成器 (真实声波 FFT 口型)", "mode": "synth"},
            {"id": "zh-CN-XiaoyiNeural", "label": "Edge 神经语音 · 活泼女声 (本地联网可用)", "mode": "edge"},
            {"id": "zh-CN-YunxiNeural", "label": "Edge 神经语音 · 阳光男声 (本地联网可用)", "mode": "edge"},
        ],
    }


@app.get("/api/avatars")
async def list_avatars() -> Dict[str, Any]:
    raw_items: List[Dict[str, Any]] = []
    if AVATARS_DIR.exists():
        for mpath in AVATARS_DIR.glob("*/manifest.json"):
            try:
                data = json.loads(mpath.read_text(encoding="utf-8"))
                folder = mpath.parent.name
                layers = data.get("layers", {})
                raw_items.append(
                    {
                        "id": data.get("id", folder),
                        "folder": folder,
                        "name": data.get("name", folder),
                        "manifestUrl": f"/avatars/{folder}/manifest.json",
                        "sourceUrl": f"/avatars/{folder}/source.png",
                        "outfitMode": data.get("outfitMode", "pants"),
                        "hasGlasses": bool(data.get("hasGlasses", False)),
                        "hasSkeletalLimbs": "torso" in layers and "leftArm" in layers and "leftLeg" in layers,
                        "isBuiltin": folder in BUILTIN_IDS,
                        "mtime": mpath.stat().st_mtime,
                    }
                )
            except Exception:
                continue

    def sort_key(item: Dict[str, Any]) -> tuple:
        fid = item["folder"]
        if fid in BUILTIN_IDS:
            return (0, BUILTIN_ORDER.index(fid), "")
        return (1, -int(item.get("mtime", 0)), fid)

    raw_items.sort(key=sort_key)
    for it in raw_items:
        it.pop("mtime", None)
        it.pop("folder", None)
    return {"avatars": raw_items}


@app.post("/api/tts")
async def synthesize_tts(req: TTSRequest) -> Response:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.mode in ("edge", "auto"):
        try:
            comm = edge_tts.Communicate(
                text,
                req.voice if req.voice.startswith(("zh-", "en-", "ja-")) else "zh-CN-XiaoyiNeural",
                rate=req.rate,
                pitch=req.pitch,
                proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
                connect_timeout=4,
            )
            mp3_chunks = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    mp3_chunks.extend(chunk["data"])
            if len(mp3_chunks) > 256:
                return Response(
                    content=bytes(mp3_chunks),
                    media_type="audio/mpeg",
                    headers={"X-TTS-Engine": "edge-tts"},
                )
        except Exception as exc:
            if req.mode == "edge":
                raise HTTPException(
                    status_code=503,
                    detail=f"Edge-TTS unavailable in current network ({type(exc).__name__}); use Browser TTS or Cute Synth mode.",
                )

    wav_bytes = synthesize_cute_formant_wav(text)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-TTS-Engine": "cartoon-formant-synth"},
    )


@app.post("/api/import-character")
async def import_character(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    outfit_mode: str = Form("auto"),
) -> JSONResponse:
    raw = await file.read()
    if len(raw) < 128:
        raise HTTPException(status_code=400, detail="上传的图片文件无效或为空")

    slug = f"custom-{int(time.time() * 1000)}"
    display_name = (name or Path(file.filename or "自定义角色").stem).strip() or "自定义角色"
    resolved_mode = outfit_mode if outfit_mode in ("auto", "pants", "skirt") else "auto"
    out_dir = AVATARS_DIR / slug

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        manifest = await asyncio.to_thread(
            build_universal_bundle,
            base_image_path=tmp_path,
            output_dir=out_dir,
            character_id=slug,
            character_name=display_name,
            outfit_mode=resolved_mode,
            variants=None,
            landmark_overrides=None,
        )
    except Exception as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"角色自动抠图与骨骼绑定失败: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(
        {
            "ok": True,
            "id": manifest["id"],
            "name": manifest["name"],
            "outfitMode": manifest.get("outfitMode", "pants"),
            "hasGlasses": manifest.get("hasGlasses", False),
            "manifestUrl": f"/avatars/{slug}/manifest.json",
            "manifest": manifest,
        }
    )


@app.put("/api/avatars/{avatar_id}/landmarks")
async def update_landmarks(avatar_id: str, payload: LandmarksUpdate) -> Dict[str, Any]:
    safe_id = _sanitize_avatar_id(avatar_id)
    mpath = AVATARS_DIR / safe_id / "manifest.json"
    if not mpath.exists():
        raise HTTPException(status_code=404, detail="Avatar manifest not found")

    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest.setdefault("landmarks", {}).update(payload.landmarks)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "landmarks": manifest["landmarks"]}


@app.post("/api/avatars/{avatar_id}/rerig")
async def rerig_avatar(avatar_id: str, req: RerigRequest) -> Dict[str, Any]:
    """
    Re-run universal 6-layer capsule segmentation & skin-matched facial patch synthesis
    around calibrated landmarks and/or an explicit `outfitMode` (`auto` | `pants` | `skirt`).
    """
    safe_id = _sanitize_avatar_id(avatar_id)
    avatar_dir = AVATARS_DIR / safe_id
    mpath = avatar_dir / "manifest.json"
    if not mpath.exists():
        raise HTTPException(status_code=404, detail="Avatar bundle not found")

    old_manifest = json.loads(mpath.read_text(encoding="utf-8"))
    src_path = avatar_dir / "source.png"
    if not src_path.exists():
        src_path = avatar_dir / "full_rgba.png"
    if not src_path.exists():
        raise HTTPException(status_code=400, detail="Source image missing in avatar bundle")

    merged_landmarks = dict(old_manifest.get("landmarks", {}))
    if req.landmarks:
        merged_landmarks.update(req.landmarks)

    mode = req.outfitMode if req.outfitMode in ("auto", "pants", "skirt") else "auto"
    char_name = old_manifest.get("name", safe_id)

    try:
        new_manifest = await asyncio.to_thread(
            build_universal_bundle,
            base_image_path=src_path,
            output_dir=avatar_dir,
            character_id=safe_id,
            character_name=char_name,
            outfit_mode=mode,
            variants=None,
            landmark_overrides=merged_landmarks,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重新切分绑定失败: {exc}")

    return {
        "ok": True,
        "id": safe_id,
        "outfitMode": new_manifest.get("outfitMode", "pants"),
        "manifestUrl": f"/avatars/{safe_id}/manifest.json",
        "manifest": new_manifest,
    }


@app.get("/api/avatars/{avatar_id}/export")
async def export_avatar_zip(avatar_id: str) -> Response:
    """Package a compiled `AvatarBundle` folder into a ready-to-deploy `.zip` archive."""
    safe_id = _sanitize_avatar_id(avatar_id)
    avatar_dir = AVATARS_DIR / safe_id
    mpath = avatar_dir / "manifest.json"
    if not mpath.exists():
        raise HTTPException(status_code=404, detail="Avatar bundle not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(avatar_dir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(avatar_dir).as_posix()
                zf.write(file_path, arcname=f"{safe_id}/{rel}")

        readme = f"""# {safe_id} - CartoonAvatar 6-Layer Skeletal Bundle

Put `{safe_id}/` under your web app's static `/avatars/{safe_id}/` directory and drive it with `CartoonAvatar`:

```ts
import {{ CartoonAvatar }} from './engine/CartoonAvatar';

const avatar = await CartoonAvatar.create({{
  container: document.getElementById('stage')!,
  bundleUrl: '/avatars/{safe_id}/manifest.json',
}});

avatar.smile(0.8);
avatar.wave('right');
await avatar.speakText('你好！我是可编程骨骼卡通角色！');
```
"""
        zf.writestr(f"{safe_id}/README_SDK.md", readme.encode("utf-8"))

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_id}-avatar-bundle.zip"',
        },
    )


@app.delete("/api/avatars/{avatar_id}")
async def delete_avatar(avatar_id: str) -> Dict[str, Any]:
    safe_id = _sanitize_avatar_id(avatar_id)
    if safe_id in BUILTIN_IDS:
        raise HTTPException(status_code=400, detail="内置预设角色不可删除")
    avatar_dir = AVATARS_DIR / safe_id
    if not avatar_dir.exists():
        raise HTTPException(status_code=404, detail="角色不存在")
    shutil.rmtree(avatar_dir, ignore_errors=True)
    return {"ok": True, "deleted": safe_id}
