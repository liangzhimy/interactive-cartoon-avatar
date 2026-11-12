import './style.css';
import { CartoonAvatar } from './engine/CartoonAvatar';
import type { AvatarManifest, SpeechMode } from './engine/types';

declare global {
  interface Window {
    avatar: CartoonAvatar;
  }
}

const BUILTIN_IDS = new Set(['sunny', 'male_teacher', 'female_teacher']);

const LANDMARK_CN_NAMES: Record<string, string> = {
  headCenter: '头部中心',
  leftEye: '左眼',
  rightEye: '右眼',
  nose: '鼻尖',
  mouth: '嘴部',
  neckPivot: '颈部转轴',
  chest: '胸腔中心',
  hips: '骨盆中心',
  knees: '双膝中线',
  feetGround: '脚底接地点',
  leftShoulder: '左肩轴',
  leftElbow: '左肘',
  leftWrist: '左腕',
  rightShoulder: '右肩轴',
  rightElbow: '右肘',
  rightWrist: '右腕',
  leftHip: '左髋轴',
  leftKnee: '左膝',
  leftAnkle: '左踝',
  rightHip: '右髋轴',
  rightKnee: '右膝',
  rightAnkle: '右踝',
};

const SCRIPT_TEMPLATES: Record<string, string> = {
  tpl1: `// 完整打招呼与全身动作编排：挥手说话 -> 踏步踢腿 -> 双手欢呼跳跃！
await Promise.all([
  avatar.wave({ hand: 'right', times: 3 }),
  avatar.speak('嗨！切换不同职业角色后，同一套 JS 代码可以直接复用！'),
]);
await avatar.walk({ steps: 4 });
await avatar.kick({ leg: 'right' });
await Promise.all([
  avatar.jump({ height: 140 }),
  avatar.laugh({ duration: 1500 }),
]);
await avatar.cheer({ times: 3 });`,

  tpl2: `// 跨角色轮播演示：少年 -> 男教师(裤装) -> 女教师(裙装)
for (const [url, line] of [
  ['/avatars/male_teacher/manifest.json', '同学们好，我是陈老师！'],
  ['/avatars/female_teacher/manifest.json', '大家好，我是穿百褶裙的林老师！'],
  ['/avatars/sunny/manifest.json', '哈喽，我是活力少年小阳！'],
]) {
  await avatar.loadBundle(url);
  await Promise.all([
    avatar.wave({ hand: 'right', times: 2 }),
    avatar.speak(line),
  ]);
  await avatar.walk({ steps: 2 });
}`,

  tpl3: `// 四肢关节弧度编程控制 (setLimbPose) + 双脚轮流踢腿
await avatar.kick({ leg: 'right' });
await avatar.kick({ leg: 'left' });
avatar.setLimbPose({
  leftArm: [1.85, 0.55, 0.15],
  rightArm: [-1.85, -0.55, -0.15],
});
await avatar.laugh({ duration: 1400 });
avatar.reset();`,
};

async function bootstrap(): Promise<void> {
  const stageEl = document.getElementById('avatar-stage')!;
  const hudYaw = document.getElementById('hud-yaw')!;
  const hudPitch = document.getElementById('hud-pitch')!;
  const hudJump = document.getElementById('hud-jump')!;
  const statusBadge = document.getElementById('speech-status-badge')!;
  const meterBar = document.getElementById('audio-meter-bar')!;
  const logBox = document.getElementById('api-log-box')!;

  const calBanner = document.getElementById('calibration-banner')!;
  const calReadout = document.getElementById('calibration-readout')!;

  const appendLog = (codeLine: string): void => {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    const prev = logBox.textContent?.startsWith('//') ? '' : logBox.textContent + '\n';
    logBox.textContent = `${prev}[${ts}] ${codeLine}`.trim();
    logBox.scrollTop = logBox.scrollHeight;
  };

  const avatar = new CartoonAvatar({
    container: stageEl,
    manifestUrl: '/avatars/sunny/manifest.json',
    onFrame: (p, level) => {
      hudYaw.textContent = `转头 Yaw: ${p.headYaw.toFixed(2)}`;
      hudPitch.textContent = `俯仰 Pitch: ${p.headPitch.toFixed(2)}`;
      hudJump.textContent = `离地 Jump: ${Math.round(p.jumpY)}px`;
      meterBar.style.width = `${Math.round(Math.min(1, level * 1.25) * 100)}%`;
    },
    onSpeechStatus: ({ speaking, mic, mode }) => {
      if (mic) {
        statusBadge.textContent = '麦克风实时对口型中';
        statusBadge.classList.add('active');
      } else if (speaking) {
        statusBadge.textContent = `正在说话 (${mode})`;
        statusBadge.classList.add('active');
      } else {
        statusBadge.textContent = avatar.calibrationMode ? '骨骼校准锁定中' : '待机呼吸中';
        statusBadge.classList.remove('active');
      }
    },
    onLandmarkChange: (landmarks, key) => {
      const pt = landmarks[key];
      if (pt) {
        const cn = LANDMARK_CN_NAMES[key] || key;
        calReadout.textContent = `正在微调「${cn}」(${key}): [${pt[0].toFixed(1)}, ${pt[1].toFixed(1)}]${
          avatar.mirrorCalibration ? ' · 左右对称联动已开启' : ''
        } — 调整后可点「保存关节」或「按校准重切分」`;
      }
    },
  });

  await avatar.init();
  window.avatar = avatar;

  const avatarSelect = document.getElementById('avatar-select') as HTMLSelectElement;
  const sourcePreviewImg = document.getElementById('source-preview-img') as HTMLImageElement;
  const rigLayersGrid = document.getElementById('rig-layers-grid')!;
  const rigBadgeOutfit = document.getElementById('rig-badge-outfit')!;
  const rigBadgeGlasses = document.getElementById('rig-badge-glasses')!;
  const rigOutfitSelect = document.getElementById('rig-outfit-mode') as HTMLSelectElement;
  const deleteBtn = document.getElementById('btn-workshop-delete') as HTMLButtonElement;

  const updateWorkshopForManifest = (manifest: AvatarManifest | null, manifestUrl: string) => {
    if (!manifest) return;
    const baseDir = manifestUrl.substring(0, manifestUrl.lastIndexOf('/'));
    const bust = `?t=${Date.now()}`;
    sourcePreviewImg.src = `${baseDir}/source.png${bust}`;

    const isSkirt = manifest.outfitMode === 'skirt';
    rigBadgeOutfit.textContent = isSkirt ? '裙装·长袍模式 (Skirt)' : '裤装·短裤模式 (Pants)';
    rigBadgeGlasses.hidden = !manifest.hasGlasses;
    deleteBtn.hidden = BUILTIN_IDS.has(manifest.id);

    const cards: Array<{ title: string; rel?: string; dim?: string }> = [
      {
        title: '头部 head',
        rel: manifest.layers.head.texture,
        dim: `${manifest.layers.head.width}×${manifest.layers.head.height}`,
      },
      {
        title: isSkirt ? '躯干+裙摆' : '躯干 torso',
        rel: manifest.layers.torso?.texture ?? manifest.layers.body.texture,
        dim: manifest.layers.torso
          ? `${manifest.layers.torso.width}×${manifest.layers.torso.height}`
          : `${manifest.layers.body.width}×${manifest.layers.body.height}`,
      },
      {
        title: '左臂胶囊',
        rel: manifest.layers.leftArm?.texture,
        dim: manifest.layers.leftArm
          ? `${manifest.layers.leftArm.width}×${manifest.layers.leftArm.height}`
          : '未分割',
      },
      {
        title: '右臂胶囊',
        rel: manifest.layers.rightArm?.texture,
        dim: manifest.layers.rightArm
          ? `${manifest.layers.rightArm.width}×${manifest.layers.rightArm.height}`
          : '未分割',
      },
      {
        title: '左腿球头',
        rel: manifest.layers.leftLeg?.texture,
        dim: manifest.layers.leftLeg
          ? `${manifest.layers.leftLeg.width}×${manifest.layers.leftLeg.height}`
          : '未分割',
      },
      {
        title: '右腿球头',
        rel: manifest.layers.rightLeg?.texture,
        dim: manifest.layers.rightLeg
          ? `${manifest.layers.rightLeg.width}×${manifest.layers.rightLeg.height}`
          : '未分割',
      },
      {
        title: '闭眼补丁',
        rel: manifest.patches.eyes_closed,
        dim: '自适应羽化',
      },
      {
        title: '大笑补丁',
        rel: manifest.patches.mouth_laugh,
        dim: '自适应羽化',
      },
    ];

    rigLayersGrid.innerHTML = '';
    for (const c of cards) {
      const card = document.createElement('div');
      card.className = 'rig-layer-card';
      card.innerHTML = `
        <div class="rig-layer-thumb-wrap">
          ${c.rel ? `<img src="${baseDir}/${c.rel}${bust}" alt="${c.title}" loading="lazy" />` : '<span>—</span>'}
        </div>
        <span class="rig-layer-title">${c.title}</span>
        <span class="rig-layer-dim">${c.dim ?? ''}</span>
      `;
      rigLayersGrid.appendChild(card);
    }
  };

  const refreshAvatarDropdown = async (selectedUrl?: string) => {
    try {
      const res = await fetch('/api/avatars');
      if (!res.ok) return;
      const data = (await res.json()) as {
        avatars: Array<{
          id: string;
          name: string;
          manifestUrl: string;
          sourceUrl: string;
          outfitMode?: string;
          isBuiltin?: boolean;
        }>;
      };
      if (data.avatars.length > 0) {
        const target = selectedUrl ?? avatarSelect.value;
        avatarSelect.innerHTML = '';
        for (const item of data.avatars) {
          const opt = document.createElement('option');
          opt.value = item.manifestUrl;
          opt.dataset.source = item.sourceUrl;
          opt.textContent = item.isBuiltin ? item.name : `${item.name} · 自定义`;
          if (item.manifestUrl === target) opt.selected = true;
          avatarSelect.appendChild(opt);
        }
      }
    } catch {
      // Static fallback keeps default 3 built-in <option>s
    }
  };

  await refreshAvatarDropdown('/avatars/sunny/manifest.json');
  updateWorkshopForManifest(avatar.getManifest(), '/avatars/sunny/manifest.json');

  // Tab navigation helper
  const tabBtns = document.querySelectorAll<HTMLButtonElement>('.panel-tab');
  const tabPanes: Record<string, HTMLElement> = {
    actions: document.getElementById('tab-pane-actions')!,
    rigging: document.getElementById('tab-pane-rigging')!,
    speech: document.getElementById('tab-pane-speech')!,
    code: document.getElementById('tab-pane-code')!,
  };

  const switchTab = (target: string) => {
    tabBtns.forEach((b) => {
      const active = b.dataset.tab === target;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', String(active));
    });
    Object.entries(tabPanes).forEach(([k, pane]) => {
      const show = k === target;
      pane.classList.toggle('active', show);
      pane.hidden = !show;
    });
  };

  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab!));
  });

  document.getElementById('btn-open-rigging-wizard')!.addEventListener('click', () => {
    switchTab('rigging');
    const dz = document.getElementById('rig-dropzone')!;
    dz.classList.add('dragover');
    window.setTimeout(() => dz.classList.remove('dragover'), 550);
  });

  avatarSelect.addEventListener('change', async () => {
    const url = avatarSelect.value;
    const manifest = await avatar.loadBundle(url);
    updateWorkshopForManifest(manifest, url);
    appendLog(`await avatar.loadBundle('${url}'); // outfitMode='${manifest.outfitMode ?? 'pants'}'`);
  });

  // ==================== Universal Single-Image Import Wizard ====================
  const dropzone = document.getElementById('rig-dropzone')!;
  const uploadInput = document.getElementById('input-upload-image') as HTMLInputElement;
  const uploadPreview = document.getElementById('rig-upload-preview') as HTMLImageElement;
  const dropzoneEmpty = document.getElementById('rig-dropzone-empty')!;
  const charNameInput = document.getElementById('rig-char-name') as HTMLInputElement;
  const runImportBtn = document.getElementById('btn-run-import') as HTMLButtonElement;
  const importStatus = document.getElementById('rig-import-status')!;

  let pendingFile: File | null = null;
  let autoImportOnPick = false;

  const stageSelectedFile = (file: File) => {
    pendingFile = file;
    const objUrl = URL.createObjectURL(file);
    uploadPreview.src = objUrl;
    uploadPreview.hidden = false;
    const stem = file.name.replace(/\.[^.]+$/, '');
    if (!charNameInput.value.trim()) {
      charNameInput.value = stem;
    }
    dropzoneEmpty.querySelector('strong')!.textContent = `已就绪：${file.name}`;
    dropzoneEmpty.querySelector('span')!.textContent =
      '确认名称与下装模式后，点击下方「开始自动抠图与 6 层骨骼绑定」（也可点击此处重选图片）';
  };

  const executeCharacterImport = async (file: File) => {
    runImportBtn.disabled = true;
    const prevBtnText = runImportBtn.textContent;
    runImportBtn.textContent = '正在净色抠图、切分四肢胶囊与合成五官补丁...';
    importStatus.classList.add('busy');
    importStatus.textContent =
      '正在执行通用 CV 绑定管线：边缘+内袋去背景 → 颈部曲线分离 → 6 层四肢胶囊切分 → 肤色/眼镜自适应表情生成...';

    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('name', charNameInput.value.trim() || file.name.replace(/\.[^.]+$/, ''));
      fd.append('outfit_mode', rigOutfitSelect.value);

      const res = await fetch('/api/import-character', { method: 'POST', body: fd });
      if (!res.ok) {
        const errData = (await res.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(errData?.detail || (await res.text()));
      }
      const result = (await res.json()) as {
        id: string;
        name: string;
        outfitMode: string;
        manifestUrl: string;
      };

      await refreshAvatarDropdown(result.manifestUrl);
      avatarSelect.value = result.manifestUrl;
      const loadedManifest = await avatar.loadBundle(result.manifestUrl);
      updateWorkshopForManifest(loadedManifest, result.manifestUrl);

      importStatus.classList.remove('busy');
      importStatus.textContent = `绑定成功！「${result.name}」已拆解为 6 层骨骼 (${
        result.outfitMode === 'skirt' ? '裙装模式' : '裤装模式'
      }) 并登上舞台，正在为您挥手演示！`;
      appendLog(`// 新角色单图 6 层绑定完成: ${result.manifestUrl} (${result.outfitMode})`);

      // Celebrate with an immediate wave + blink so user sees their new character come alive!
      void avatar.blink();
      void avatar.wave({ hand: 'right', times: 2 });
    } catch (err) {
      importStatus.classList.remove('busy');
      importStatus.textContent = `导入失败：${err instanceof Error ? err.message : String(err)}`;
    } finally {
      runImportBtn.disabled = false;
      runImportBtn.textContent = prevBtnText;
    }
  };

  dropzone.addEventListener('click', () => {
    autoImportOnPick = false;
    uploadInput.click();
  });
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer?.files?.[0];
    if (file) stageSelectedFile(file);
  });

  uploadInput.addEventListener('change', () => {
    const file = uploadInput.files?.[0];
    if (!file) return;
    stageSelectedFile(file);
    if (autoImportOnPick) {
      autoImportOnPick = false;
      void executeCharacterImport(file);
    }
    uploadInput.value = '';
  });

  runImportBtn.addEventListener('click', () => {
    if (!pendingFile) {
      autoImportOnPick = true;
      uploadInput.click();
      return;
    }
    void executeCharacterImport(pendingFile);
  });

  // Copy A-Pose Prompt Template
  const copyPromptBtn = document.getElementById('btn-copy-prompt')!;
  copyPromptBtn.addEventListener('click', async () => {
    const code = document.getElementById('apose-prompt-code')!.textContent ?? '';
    await navigator.clipboard.writeText(code);
    const orig = copyPromptBtn.textContent;
    copyPromptBtn.textContent = '已复制！';
    window.setTimeout(() => {
      copyPromptBtn.textContent = orig;
    }, 1500);
  });

  // Compare original source image card
  const compareCard = document.getElementById('source-compare-card')!;
  document.getElementById('btn-compare-source')!.addEventListener('click', () => {
    compareCard.hidden = !compareCard.hidden;
  });
  document.getElementById('btn-close-source')!.addEventListener('click', () => {
    compareCard.hidden = true;
  });

  // ==================== Stage Toolbar & Calibration / Re-Rig / Export ====================
  const mouseFollowToggle = document.getElementById('toggle-mouse-follow') as HTMLInputElement;
  mouseFollowToggle.addEventListener('change', () => {
    avatar.mouseFollow = mouseFollowToggle.checked;
    if (!mouseFollowToggle.checked) avatar.turnHead(0, 0, 0);
    appendLog(`avatar.mouseFollow = ${mouseFollowToggle.checked};`);
  });

  const wireToggle = document.getElementById('toggle-wireframe') as HTMLInputElement;
  wireToggle.addEventListener('change', () => {
    avatar.setWireframe(wireToggle.checked);
    appendLog(`avatar.setWireframe(${wireToggle.checked});`);
  });

  const calToggle = document.getElementById('toggle-calibration') as HTMLInputElement;
  const mirrorWrap = document.getElementById('wrap-mirror-cal')!;
  const mirrorToggle = document.getElementById('toggle-mirror-cal') as HTMLInputElement;
  const saveLmBtn = document.getElementById('btn-save-landmarks')!;
  const stageRerigBtn = document.getElementById('btn-stage-rerig')!;
  const workshopToggleCalBtn = document.getElementById('btn-workshop-toggle-cal')!;

  const syncCalibrationUI = (enabled: boolean) => {
    calToggle.checked = enabled;
    avatar.setCalibrationMode(enabled);
    mirrorWrap.hidden = !enabled;
    saveLmBtn.hidden = !enabled;
    stageRerigBtn.hidden = !enabled;
    calBanner.hidden = !enabled;
    workshopToggleCalBtn.textContent = enabled
      ? '退出舞台关节拖拽校准'
      : '开启舞台关节拖拽校准';
    statusBadge.textContent = enabled ? '骨骼校准锁定中' : '待机呼吸中';
    appendLog(`avatar.setCalibrationMode(${enabled});`);
  };

  calToggle.addEventListener('change', () => syncCalibrationUI(calToggle.checked));
  workshopToggleCalBtn.addEventListener('click', () => syncCalibrationUI(!calToggle.checked));

  mirrorToggle.addEventListener('change', () => {
    avatar.mirrorCalibration = mirrorToggle.checked;
  });

  const saveLandmarksOnly = async () => {
    const manifest = avatar.getManifest();
    const landmarks = avatar.getLandmarks();
    if (!manifest || !landmarks) return;
    const res = await fetch(`/api/avatars/${manifest.id}/landmarks`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks }),
    });
    if (res.ok) {
      calReadout.textContent = `已保存「${manifest.name}」的关节轴心！可点击挥手/踢腿测试动作效果。`;
      importStatus.textContent = `已保存「${manifest.name}」的 22 个关键点配置。`;
      appendLog(`// 已保存 ${manifest.id} 的骨骼关键点校准配置`);
    }
  };

  saveLmBtn.addEventListener('click', () => void saveLandmarksOnly());
  document.getElementById('btn-workshop-save-lm')!.addEventListener('click', () => {
    void saveLandmarksOnly();
  });

  const rerigCurrentCharacter = async () => {
    const manifest = avatar.getManifest();
    const landmarks = avatar.getLandmarks();
    if (!manifest || !landmarks) return;

    importStatus.classList.add('busy');
    importStatus.textContent = `正在根据当前校准点与下装模式重新切分「${manifest.name}」的 6 图层与五官补丁...`;
    calReadout.textContent = '正在重新运行服务端胶囊关节分割，请稍候（约 3 秒）...';

    try {
      const res = await fetch(`/api/avatars/${manifest.id}/rerig`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          outfitMode: rigOutfitSelect.value,
          landmarks,
        }),
      });
      if (!res.ok) {
        const errData = (await res.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(errData?.detail || (await res.text()));
      }
      const data = (await res.json()) as { manifestUrl: string; outfitMode: string };
      const reloaded = await avatar.loadBundle(data.manifestUrl);
      updateWorkshopForManifest(reloaded, data.manifestUrl);
      importStatus.classList.remove('busy');
      importStatus.textContent = `重切分完成！「${reloaded.name}」已更新为 ${
        data.outfitMode === 'skirt' ? '裙装模式' : '裤装模式'
      } 6 层纹理。`;
      calReadout.textContent = '重切分已完成并热重载至舞台！';
      appendLog(`// 已按校准点重新切分 ${manifest.id} (${data.outfitMode})`);
    } catch (err) {
      importStatus.classList.remove('busy');
      importStatus.textContent = `重切分失败：${err instanceof Error ? err.message : String(err)}`;
    }
  };

  stageRerigBtn.addEventListener('click', () => void rerigCurrentCharacter());
  document.getElementById('btn-workshop-rerig')!.addEventListener('click', () => {
    void rerigCurrentCharacter();
  });

  const exportCurrentBundle = () => {
    const manifest = avatar.getManifest();
    if (!manifest) return;
    const link = document.createElement('a');
    link.href = `/api/avatars/${encodeURIComponent(manifest.id)}/export`;
    link.download = `${manifest.id}-avatar-bundle.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    appendLog(`// 导出标准 6 层角色包: /api/avatars/${manifest.id}/export`);
  };

  document.getElementById('btn-header-export')!.addEventListener('click', exportCurrentBundle);
  document.getElementById('btn-workshop-export')!.addEventListener('click', exportCurrentBundle);

  deleteBtn.addEventListener('click', async () => {
    const manifest = avatar.getManifest();
    if (!manifest || BUILTIN_IDS.has(manifest.id)) return;
    if (!window.confirm(`确定要删除自定义角色「${manifest.name}」吗？`)) return;

    const res = await fetch(`/api/avatars/${encodeURIComponent(manifest.id)}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      await refreshAvatarDropdown('/avatars/sunny/manifest.json');
      avatarSelect.value = '/avatars/sunny/manifest.json';
      const sunny = await avatar.loadBundle('/avatars/sunny/manifest.json');
      updateWorkshopForManifest(sunny, '/avatars/sunny/manifest.json');
      importStatus.textContent = `已删除自定义角色「${manifest.name}」，已切回默认角色。`;
      appendLog(`// 已删除自定义角色 ${manifest.id}`);
    }
  });

  // ==================== Actions, Sliders, Speech & Script Console ====================
  const jumpHeightSlider = document.getElementById('slider-jump-height') as HTMLInputElement;
  const jumpHeightOut = document.getElementById('out-jump-height')!;
  jumpHeightSlider.addEventListener('input', () => {
    jumpHeightOut.textContent = `${jumpHeightSlider.value}px`;
  });

  const flashBtn = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add('triggered');
    window.setTimeout(() => el.classList.remove('triggered'), 140);
  };

  const bindAction = (id: string, fn: () => void) => {
    document.getElementById(id)!.addEventListener('click', () => {
      flashBtn(id);
      fn();
    });
  };

  bindAction('act-wave', () => {
    void avatar.wave({ hand: 'right', times: 3 });
    appendLog("await avatar.wave({ hand: 'right', times: 3 });");
  });

  bindAction('act-cheer', () => {
    void avatar.cheer({ times: 3 });
    appendLog('await avatar.cheer({ times: 3 });');
  });

  bindAction('act-walk', () => {
    void avatar.walk({ steps: 4 });
    appendLog('await avatar.walk({ steps: 4 });');
  });

  bindAction('act-kick', () => {
    void avatar.kick({ leg: 'right' });
    appendLog("await avatar.kick({ leg: 'right' });");
  });

  bindAction('act-blink', () => {
    void avatar.blink();
    appendLog('await avatar.blink();');
  });

  bindAction('act-smile', () => {
    void avatar.smile({ duration: 2200 });
    appendLog('await avatar.smile({ duration: 2200 });');
  });

  bindAction('act-laugh', () => {
    void avatar.laugh({ duration: 2400 });
    appendLog('await avatar.laugh({ duration: 2400 });');
  });

  bindAction('act-nod', () => {
    void avatar.nod({ times: 2 });
    appendLog('await avatar.nod({ times: 2 });');
  });

  bindAction('act-shake', () => {
    void avatar.shakeHead({ times: 2 });
    appendLog('await avatar.shakeHead({ times: 2 });');
  });

  bindAction('act-turn-left', () => {
    updateXYPad(-0.72, -0.06);
    appendLog('avatar.turnHead(-0.72, -0.06);');
  });

  bindAction('act-turn-right', () => {
    updateXYPad(0.72, -0.06);
    appendLog('avatar.turnHead(0.72, -0.06);');
  });

  bindAction('act-jump', () => {
    const h = Number(jumpHeightSlider.value) || 130;
    void avatar.jump({ height: h });
    appendLog(`await avatar.jump({ height: ${h} });`);
  });

  document.getElementById('act-combo-joyjump')!.addEventListener('click', () => {
    const h = Number(jumpHeightSlider.value) || 135;
    void Promise.all([avatar.jump({ height: h }), avatar.laugh({ duration: 1800 })]);
    appendLog(`await Promise.all([avatar.jump({ height: ${h} }), avatar.laugh()]);`);
  });

  document.getElementById('act-combo-greet')!.addEventListener('click', async () => {
    appendLog('// 执行挥手打招呼 + 全身互动演出序列');
    const mode = (document.getElementById('select-speech-mode') as HTMLSelectElement)
      .value as SpeechMode;
    const curName = avatar.getManifest()?.name.split(' ')[0] ?? '大家好';
    void avatar.speak(`嗨！我是${curName}，很高兴见到你！`, { mode });
    await avatar.wave({ hand: 'right', times: 3 });
    await avatar.walk({ steps: 4 });
    await Promise.all([avatar.jump({ height: 135 }), avatar.cheer({ times: 2 })]);
  });

  // 2.5D XY Pad & Fine Sliders
  const xyPad = document.getElementById('head-xy-pad')!;
  const xyKnob = document.getElementById('head-xy-knob')!;
  const rollSlider = document.getElementById('slider-roll') as HTMLInputElement;
  const rollOut = document.getElementById('out-roll')!;
  const smileSlider = document.getElementById('slider-smile') as HTMLInputElement;
  const smileOut = document.getElementById('out-smile')!;

  let currentPadYaw = 0;
  let currentPadPitch = 0;

  const updateXYPad = (yaw: number, pitch: number) => {
    currentPadYaw = Math.max(-1, Math.min(1, yaw));
    currentPadPitch = Math.max(-1, Math.min(1, pitch));
    xyKnob.style.left = `${(currentPadYaw * 0.5 + 0.5) * 100}%`;
    xyKnob.style.top = `${(currentPadPitch * 0.5 + 0.5) * 100}%`;
    avatar.turnHead(currentPadYaw, currentPadPitch, Number(rollSlider.value));
  };

  let padDragging = false;
  const setPadFromPointer = (e: PointerEvent) => {
    const rect = xyPad.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    updateXYPad(nx, ny);
  };

  xyPad.addEventListener('pointerdown', (e) => {
    padDragging = true;
    xyPad.setPointerCapture(e.pointerId);
    setPadFromPointer(e);
  });
  xyPad.addEventListener('pointermove', (e) => {
    if (padDragging) setPadFromPointer(e);
  });
  xyPad.addEventListener('pointerup', () => {
    padDragging = false;
  });

  rollSlider.addEventListener('input', () => {
    const r = Number(rollSlider.value);
    rollOut.textContent = r.toFixed(2);
    avatar.turnHead(currentPadYaw, currentPadPitch, r);
  });

  smileSlider.addEventListener('input', () => {
    const s = Number(smileSlider.value);
    smileOut.textContent = s.toFixed(2);
    avatar.actions.setSmileLevel(s);
  });

  document.getElementById('btn-reset-pose')!.addEventListener('click', () => {
    rollSlider.value = '0';
    rollOut.textContent = '0.00';
    smileSlider.value = '0';
    smileOut.textContent = '0.00';
    updateXYPad(0, 0);
    avatar.reset();
    appendLog('avatar.reset();');
  });

  // Speech & Lip-Sync Controls
  const speechInput = document.getElementById('input-speech-text') as HTMLTextAreaElement;
  const speechModeSelect = document.getElementById('select-speech-mode') as HTMLSelectElement;

  document.querySelectorAll<HTMLButtonElement>('.chip-btn').forEach((chip) => {
    chip.addEventListener('click', () => {
      if (chip.dataset.text) {
        speechInput.value = chip.dataset.text;
      }
    });
  });

  document.getElementById('btn-speak')!.addEventListener('click', () => {
    const text = speechInput.value.trim();
    if (!text) return;
    const mode = speechModeSelect.value as SpeechMode;
    void avatar.speak(text, { mode });
    appendLog(`await avatar.speak(${JSON.stringify(text)}, { mode: '${mode}' });`);
  });

  document.getElementById('btn-stop-speak')!.addEventListener('click', () => {
    avatar.stopSpeaking();
    appendLog('avatar.stopSpeaking();');
  });

  // Microphone toggle
  const micBtn = document.getElementById('btn-mic-toggle')!;
  micBtn.addEventListener('click', async () => {
    try {
      const running = await avatar.toggleMicrophone();
      micBtn.classList.toggle('recording', running);
      micBtn.textContent = running ? '关闭麦克风跟读 (收音中...)' : '开启麦克风实时跟读';
      appendLog(`await avatar.toggleMicrophone(); // active=${running}`);
    } catch (err) {
      alert(`无法开启麦克风，请检查浏览器麦克风权限：${err instanceof Error ? err.message : String(err)}`);
    }
  });

  // External audio upload
  const audioInput = document.getElementById('input-audio-file') as HTMLInputElement;
  audioInput.addEventListener('change', async () => {
    const file = audioInput.files?.[0];
    if (!file) return;
    await avatar.speakAudio(file);
    appendLog(`await avatar.speakAudio(File('${file.name}'));`);
    audioInput.value = '';
  });

  // Live JS SDK Script Runner
  const codeEditor = document.getElementById('code-editor') as HTMLTextAreaElement;
  const scriptStatus = document.getElementById('script-run-status')!;

  document.getElementById('script-tpl-1')!.addEventListener('click', () => {
    codeEditor.value = SCRIPT_TEMPLATES.tpl1;
  });
  document.getElementById('script-tpl-2')!.addEventListener('click', () => {
    codeEditor.value = SCRIPT_TEMPLATES.tpl2;
  });
  document.getElementById('script-tpl-3')!.addEventListener('click', () => {
    codeEditor.value = SCRIPT_TEMPLATES.tpl3;
  });

  document.getElementById('btn-run-script')!.addEventListener('click', async () => {
    const src = codeEditor.value;
    scriptStatus.textContent = '运行中...';
    appendLog('// 开始运行自定义脚本');
    const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
    try {
      const AsyncFn = Object.getPrototypeOf(async function () {}).constructor as new (
        ...args: string[]
      ) => (av: CartoonAvatar, sl: typeof sleep) => Promise<void>;
      const runner = new AsyncFn('avatar', 'sleep', src);
      await runner(avatar, sleep);
      const curM = avatar.getManifest();
      if (curM) {
        const curUrl = `/avatars/${curM.id}/manifest.json`;
        avatarSelect.value = curUrl;
        updateWorkshopForManifest(curM, curUrl);
      }
      scriptStatus.textContent = '执行完成';
    } catch (err) {
      scriptStatus.textContent = `错误: ${err instanceof Error ? err.message : String(err)}`;
    }
  });

  document.getElementById('btn-clear-log')!.addEventListener('click', () => {
    logBox.textContent = '// 日志已清空';
  });

  // Keyboard shortcuts (when not typing inside textarea/input!)
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return;

    switch (e.key.toLowerCase()) {
      case 'w':
        document.getElementById('act-wave')!.click();
        break;
      case 'e':
        document.getElementById('act-cheer')!.click();
        break;
      case 'r':
        document.getElementById('act-walk')!.click();
        break;
      case 'k':
        document.getElementById('act-kick')!.click();
        break;
      case '1':
        document.getElementById('act-blink')!.click();
        break;
      case '2':
        document.getElementById('act-smile')!.click();
        break;
      case '3':
        document.getElementById('act-laugh')!.click();
        break;
      case '4':
        document.getElementById('act-nod')!.click();
        break;
      case '5':
        document.getElementById('act-shake')!.click();
        break;
      case '6':
        document.getElementById('act-turn-left')!.click();
        break;
      case '7':
        document.getElementById('act-turn-right')!.click();
        break;
      case '8':
      case ' ':
        e.preventDefault();
        document.getElementById('act-jump')!.click();
        break;
    }
  });
}

void bootstrap();
