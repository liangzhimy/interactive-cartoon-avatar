# Interactive Cartoon Avatar Engine & Rigging Studio (`interactive-cartoon-avatar`)

把**任意单张全身正面 / 微 A-Pose 卡通立绘**（少年、成人男女、教师、不同职业、戴眼镜、裤装或裙装均可）转化为**网页端 60fps 实时可编程 2.5D + 四肢骨骼交互角色**：

- **手脚四肢骨骼动作**：`avatar.wave()` 挥手打招呼、`avatar.cheer()` 双手欢呼、`avatar.walk()` 踏步走路、`avatar.kick()` 开心踢腿、`avatar.setLimbPose()` 12 自由度关节角控制（基于 2D 双四元数无折痕蒙皮）
- **2.5D 无畸变立体转头**：`avatar.turnHead(yaw, pitch, roll)` / `avatar.lookAt(x, y)`，跨头身比自适应三维椭球透视网格 + 鼠标视线跟随 + 发梢弹簧惯性
- **眨眼、微笑与开怀大笑**：`avatar.blink()`、`avatar.smile()`、`avatar.laugh()`，支持自适应肤色融合与戴眼镜角色镜框保护
- **点头、摇头与四阶段物理跳跃**：`avatar.nod()`、`avatar.shakeHead()`、`avatar.jump()`（蓄力下蹲挤压 → 腾空拉伸展臂 → 落地缓冲回弹 + 动态地面阴影）
- **开口说话与实时对口型**：`avatar.speak()` 支持浏览器自然人声 TTS、Edge 神经语音、Web Audio 频谱共振峰合成、**麦克风实时收音驱动**与外部音频驱动
- **通用单图换人工坊（角色工坊）**：拖拽上传单张全身卡通图 → 自动去背景、裤装/裙装自适应六层胶囊骨骼切分（`head` / `torso` / `leftArm` / `rightArm` / `leftLeg` / `rightLeg`）+ 7 张五官表情贴片 → 舞台中文骨骼链对称拖拽校准 → 一键重切分与 `.zip` 独立资产包导出

---

## 内置预设角色（开箱即用）

- `public/avatars/sunny/`：**小阳（活力少年 · 短裤）** — Q 版少年裤装骨骼
- `public/avatars/male_teacher/`：**陈老师（男教师 · 裤装）** — 成人头身比 + 护镜表情 + 西裤骨骼
- `public/avatars/female_teacher/`：**林老师（女教师 · 裙装）** — 成人头身比 + 护镜表情 + 百褶裙下双腿独立迈步踢腿骨骼

---

## 目录结构

```text
interactive-cartoon-avatar/
├── public/avatars/                 # 角色资产包目录（内置 sunny / male_teacher / female_teacher）
│   └── <avatar_id>/
│       ├── source.png              # 原始单张平面全身图
│       ├── head.png / torso.png    # 解耦后的头部与延展颈部躯干/裙摆图层
│       ├── left_arm.png / right_arm.png / left_leg.png / right_leg.png
│       ├── patches/                # 羽化对齐的闭眼、笑眼、微笑、大笑、A/O/E 口型贴片
│       └── manifest.json           # 22 个骨骼关键点与变形网格配置
├── server/
│   ├── app.py                      # FastAPI 服务：单图导入绑骨、重切分、.zip 导出、TTS 合成
│   ├── requirements.txt            # Python 依赖清单
│   └── tools/
│       ├── universal_rig.py        # 通用单图去背景 + 六层胶囊骨骼切分 + 表情合成管线
│       ├── build_avatar_bundle.py  # 多图 AI 变体构建脚本
│       └── segment_limbs.py        # 四肢胶囊关节分割工具
└── src/
    ├── engine/
    │   ├── CartoonAvatar.ts        # 核心可编程 SDK 主类
    │   ├── MeshRig.ts              # 2.5D 椭球头部网格 + 6 层双四元数蒙皮骨骼网格 + 中文校准层
    │   ├── ActionController.ts     # 全身动作状态机与阻尼弹簧叠加器
    │   ├── LipSyncEngine.ts        # Web Audio FFT 频谱口型解算器 + TTS / 麦克风驱动
    │   └── SpringPhysics.ts        # 二阶阻尼谐振子物理引擎
    ├── main.ts                     # 网页交互工作台与“角色工坊 (换人)”控制逻辑
    └── style.css                   # 动画工作台样式系统
```

---

## 本地克隆与快速启动

```bash
git clone https://github.com/liangzhimy/interactive-cartoon-avatar.git
cd interactive-cartoon-avatar
```

### 1. 安装并启动 Python 绑骨与语音后端 (`127.0.0.1:8765`)

```bash
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt
./.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

### 2. 安装并启动前端工作台 (`localhost:5173`)

```bash
npm install
npm run dev
```

---

## 可编程 SDK 调用示例

```ts
import { CartoonAvatar } from './src/engine/CartoonAvatar';

const avatar = new CartoonAvatar({
  container: document.getElementById('avatar-stage')!,
  manifestUrl: '/avatars/female_teacher/manifest.json',
});

await avatar.init();

await avatar.wave({ hand: 'right', times: 3 });
await avatar.speak('大家好！我是林老师，现在手脚四肢和表情都能通过代码自由操控啦！');
await avatar.walk({ steps: 4 });
await avatar.kick({ leg: 'right' });
await Promise.all([
  avatar.jump({ height: 140 }),
  avatar.cheer({ times: 2 }),
]);
```
