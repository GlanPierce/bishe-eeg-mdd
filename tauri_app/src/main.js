import * as echarts from 'echarts';
import * as THREE from '@rave-ieeg/three-brain/node_modules/three/build/three.module.js';
import { ViewerCanvas } from '@rave-ieeg/three-brain/src/js/core/ViewerCanvas.js';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import './styles.css';
import lhPialUrl from './assets/brain/N27/surf/lh.pial?url';
import rhPialUrl from './assets/brain/N27/surf/rh.pial?url';

const modelOptions = [
  { key: 'clean', label: 'ExplicitRegionTemporalWeightedStarGNN' },
  { key: 'targeted_repair', label: 'ExplicitRegionTemporalWeightedStarGNN + ArtifactRepair' },
  { key: 'taskonly61_multistate_arch', label: 'MultistateExplicitRegionTemporalNodeGNN TASK-only' },
  { key: 'gcn_unavailable', label: 'DynamicsGraphConvGNN GCN', disabled: true },
  { key: 'custom', label: '+ 导入模型' }
];

const electrodeCoords = {
  Fp1: [-0.42, 0.72, -1.02], Fp2: [0.42, 0.72, -1.02],
  F7: [-1.05, 0.28, -0.72], F3: [-0.48, 0.82, -0.62], Fz: [0, 0.9, -0.62], F4: [0.48, 0.82, -0.62], F8: [1.05, 0.28, -0.72],
  T3: [-1.16, 0.05, -0.04], C3: [-0.58, 0.92, -0.02], Cz: [0, 1.02, -0.02], C4: [0.58, 0.92, -0.02], T4: [1.16, 0.05, -0.04],
  T5: [-0.98, 0.2, 0.62], P3: [-0.46, 0.82, 0.58], Pz: [0, 0.9, 0.58], P4: [0.46, 0.82, 0.58], T6: [0.98, 0.2, 0.62],
  O1: [-0.36, 0.56, 1.08], O2: [0.36, 0.56, 1.08],
  A1: [-1.22, -0.08, 0.02], A2: [1.22, -0.08, 0.02],
  '23A': [-0.18, 0.78, 1.18], '24A': [0.18, 0.78, 1.18]
};

const defaultTemplateEdges = [
  { source: 'Fp1', target: 'F3', value: 0.42, score: 0.42 },
  { source: 'Fp2', target: 'F4', value: 0.38, score: 0.38 },
  { source: 'F3', target: 'C3', value: 0.66, score: 0.66 },
  { source: 'F4', target: 'C4', value: 0.58, score: 0.58 },
  { source: 'C3', target: 'P3', value: 0.82, score: 0.82 },
  { source: 'C4', target: 'P4', value: 0.76, score: 0.76 },
  { source: 'P3', target: 'O1', value: 0.50, score: 0.50 },
  { source: 'P4', target: 'O2', value: 0.46, score: 0.46 },
  { source: 'Fz', target: 'Cz', value: 0.92, score: 0.92 },
  { source: 'Cz', target: 'Pz', value: 0.70, score: 0.70 },
  { source: 'T3', target: 'C3', value: 0.36, score: 0.36 },
  { source: 'T4', target: 'C4', value: 0.34, score: 0.34 }
];

const app = document.getElementById('app');

app.innerHTML = `
  <div id="appShell" class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">EEG</div>
        <div>
          <h1>抑郁风险分析</h1>
          <p>Tauri 轻量预览版</p>
        </div>
      </div>

      <section class="risk-box">
        <span class="label">风险等级</span>
        <div id="riskLabel" class="risk-label muted">待推理</div>
        <div class="prob-line">
          <span>MDD 概率</span>
          <strong id="probValue">--</strong>
        </div>
        <div class="meter"><div id="probBar" class="meter-fill"></div></div>
      </section>

      <section class="info-box">
        <span class="label">数据导入</span>
        <button id="pickFile" class="primary-button">导入 EDF</button>
        <span class="label file-label">当前文件</span>
        <strong id="fileName">未选择</strong>
        <span class="label file-label">数据来源</span>
        <small id="fileSource" class="path-text">未选择</small>
        <small id="fileMeta">等待导入 EDF 脑电文件</small>
      </section>

      <section class="info-box model-run-box">
        <span class="label">模型选择</span>
        <div class="model-action-row">
          <div id="modelSelect" class="model-select" role="button" tabindex="0">
            <div class="model-selected-viewport">
              <div id="modelSelectedText" class="model-selected-text"></div>
            </div>
            <span class="model-chevron">⌄</span>
            <div id="modelMenu" class="model-menu hidden">
              <div class="model-menu-title">模型选择</div>
              ${modelOptions.map((item) => `
                <button class="model-menu-item ${item.disabled ? 'disabled' : ''}" data-model="${item.key}" ${item.disabled ? 'disabled' : ''}>
                  ${item.key === 'custom' ? '+ 导入模型' : item.label}
                </button>
              `).join('')}
            </div>
          </div>
          <button id="runInference" class="icon-button primary-icon" title="开始推理">推理</button>
        </div>
        <div id="progressWrap" class="inference-progress hidden">
          <div class="progress-meta">
            <span id="progressText">准备推理</span>
            <strong id="progressValue">0%</strong>
          </div>
          <div class="progress-track"><div id="progressBar" class="progress-fill"></div></div>
        </div>
        <small id="customModelPath" class="path-text"></small>
      </section>
    </aside>

    <button id="sidebarToggle" class="sidebar-toggle" title="收起侧边栏">‹</button>

    <main class="workspace">
      <nav class="tabs" aria-label="结果视图">
        <button class="tab-button active" data-tab="network">脑网络连接图</button>
        <button class="tab-button" data-tab="contrib">脑区、半球和时间贡献数据</button>
        <button class="tab-button" data-tab="edges">连接强度</button>
      </nav>

      <section id="tab-network" class="tab-page active">
        <section class="panel graph-panel">
          <div class="panel-head graph-panel-head">
            <h3>脑网络连接图</h3>
            <span>标准 10-20 EEG 位置叠加 PCC Top 连接，点击节点查看信息</span>
          </div>
          <div id="networkChart" class="brain-scene">
            <div id="nodeTooltip" class="node-tooltip hidden"></div>
          </div>
        </section>
      </section>

      <section id="tab-contrib" class="tab-page">
        <div class="contrib-grid">
          <section class="panel">
            <div class="panel-head">
              <h3>脑区贡献</h3>
              <span>正值推高 MDD 风险，负值偏向正常</span>
            </div>
            <div id="regionChart" class="chart"></div>
          </section>
          <section class="panel">
            <div class="panel-head">
              <h3>左右半球</h3>
              <span>成对电极影响度对比</span>
            </div>
            <div id="asymChart" class="chart"></div>
          </section>
          <section class="panel">
            <div class="panel-head">
              <h3>时间贡献</h3>
              <span>mean / std / delta 时间摘要</span>
            </div>
            <div id="temporalChart" class="chart"></div>
          </section>
        </div>
      </section>

      <section id="tab-edges" class="tab-page">
        <section class="table-panel">
          <div class="panel-head">
            <h3>连接强度</h3>
            <span>按连接强度和通道影响度综合排序</span>
          </div>
          <table>
            <thead>
              <tr><th>#</th><th>连接</th><th>连接值</th><th>综合分</th></tr>
            </thead>
            <tbody id="edgeRows"><tr><td colspan="4">暂无数据</td></tr></tbody>
          </table>
        </section>
      </section>
    </main>
  </div>

  <section id="consolePanel" class="console-panel console-overlay hidden">
    <div id="consoleHead" class="console-head">
      <span>Console</span>
      <button id="consoleClose" class="console-close" title="关闭">×</button>
    </div>
    <div id="consoleOutput" class="console-output"></div>
  </section>
`;

const $ = (id) => document.getElementById(id);

const els = {
  appShell: $('appShell'),
  sidebarToggle: $('sidebarToggle'),
  modelSelect: $('modelSelect'),
  modelSelectedText: $('modelSelectedText'),
  modelMenu: $('modelMenu'),
  customModelPath: $('customModelPath'),
  pickFile: $('pickFile'),
  runInference: $('runInference'),
  progressWrap: $('progressWrap'),
  progressText: $('progressText'),
  progressValue: $('progressValue'),
  progressBar: $('progressBar'),
  fileName: $('fileName'),
  fileSource: $('fileSource'),
  fileMeta: $('fileMeta'),
  riskLabel: $('riskLabel'),
  probValue: $('probValue'),
  probBar: $('probBar'),
  consolePanel: $('consolePanel'),
  consoleHead: $('consoleHead'),
  consoleClose: $('consoleClose'),
  consoleOutput: $('consoleOutput'),
  edgeRows: $('edgeRows'),
  networkChart: $('networkChart'),
  nodeTooltip: $('nodeTooltip')
};

let customModel = '';
let selectedEdf = '';
let selectedModel = 'clean';
let progressTimer = null;
let consoleDrag = null;

const charts = {
  region: echarts.init($('regionChart')),
  asym: echarts.init($('asymChart')),
  temporal: echarts.init($('temporalChart'))
};

const brain = {
  nodes: [],
  edges: [],
  canvas: null,
  animationId: null,
  raycaster: null,
  pointer: null,
  nodeObjects: [],
  edgeObjects: [],
  surfaceObjects: [],
  surfaceData: [],
  labels: [],
  nodeGlowObjects: [],
  edgeJumpLabels: [],
  drag: null,
  dotTexture: null,
  glowTexture: null,
  focusAnimation: null,
  activeTooltipNode: null,
  selectedGlowNode: null,
  selectedGlowParticles: null,
  focusedNodeName: null,
  highlightParticles: [],
  regionGlowSurfaces: [],
  signalObjects: []
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(digits);
}

function pushConsole(message, type = 'system') {
  const row = document.createElement('div');
  row.className = `console-line ${type}`;
  row.innerHTML = `<span class="console-time">[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}]</span><span class="console-type">${type.toUpperCase()}</span><span class="console-message"></span>`;
  row.querySelector('.console-message').textContent = message;
  els.consoleOutput.appendChild(row);
  els.consoleOutput.scrollTop = els.consoleOutput.scrollHeight;
}

function toggleConsole(force) {
  const show = typeof force === 'boolean' ? force : els.consolePanel.classList.contains('hidden');
  els.consolePanel.classList.toggle('hidden', !show);
}

function setProgress(value, text) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  els.progressWrap.classList.remove('hidden');
  els.progressBar.style.width = `${pct}%`;
  els.progressValue.textContent = `${pct}%`;
  if (text) els.progressText.textContent = text;
}

function startProgress() {
  clearInterval(progressTimer);
  let value = 8;
  setProgress(value, '读取 EDF');
  progressTimer = setInterval(() => {
    value = Math.min(88, value + Math.max(1, Math.round((90 - value) * 0.12)));
    setProgress(value, value < 35 ? '读取 EDF' : value < 65 ? '提取特征' : '模型推理');
  }, 450);
}

function finishProgress(ok) {
  clearInterval(progressTimer);
  progressTimer = null;
  setProgress(100, ok ? '推理完成' : '推理失败');
  if (!ok) els.progressBar.classList.add('failed');
  setTimeout(() => {
    els.progressBar.classList.remove('failed');
    if (ok) els.progressWrap.classList.add('hidden');
  }, ok ? 900 : 1600);
}

function setBusy(isBusy) {
  els.pickFile.disabled = isBusy;
  els.runInference.disabled = isBusy;
  els.modelSelect.classList.toggle('disabled', isBusy);
}

function updateModelUi() {
  const option = modelOptions.find((item) => item.key === selectedModel) || modelOptions[0];
  els.modelSelectedText.textContent = option.key === 'custom' ? '' : option.label;
  els.modelSelectedText.classList.toggle('empty', option.key === 'custom');
  els.customModelPath.textContent = option.key === 'custom' ? customModel : '';
  requestAnimationFrame(() => {
    const parent = els.modelSelectedText.parentElement;
    const distance = parent ? Math.max(0, els.modelSelectedText.scrollWidth - parent.clientWidth) : 0;
    els.modelSelectedText.style.setProperty('--scroll-distance', `${distance}px`);
    els.modelSelectedText.classList.toggle('scrolling', distance > 4);
  });
}

function resizeCharts() {
  requestAnimationFrame(() => {
    Object.values(charts).forEach((chart) => chart.resize());
    resizeBrainScene();
  });
}



function shortChannel(name) {
  return String(name).split('-')[0];
}

function classifyChannel(name) {
  const label = shortChannel(name);
  const first = label[0]?.toUpperCase();
  const hemisphere = /[1357]$|A1$/.test(label) ? 'left' : /[2468]$|A2$/.test(label) ? 'right' : 'midline';
  const region = first === 'F' ? 'frontal'
    : first === 'C' ? 'central'
      : first === 'P' ? 'parietal'
        : first === 'O' ? 'occipital'
          : first === 'T' || /^A[12]$/.test(label) ? 'temporal'
            : 'auxiliary';
  return { hemisphere, region };
}

function scalpZ(x, y) {
  const xr = x / 128;
  const yr = y / 142;
  const dome = Math.max(0, 1 - xr * xr - yr * yr);
  return 16 + 84 * Math.sqrt(dome);
}

function fixedNodePosition(node) {
  const raw = electrodeCoords[node.name] || electrodeCoords[shortChannel(node.name)];
  if (raw) {
    const x = raw[0] * 70;
    const y = -raw[2] * 82;
    return { x, y, z: scalpZ(x, y) + 8 };
  }
  const side = node.hemisphere === 'left' ? -1 : node.hemisphere === 'right' ? 1 : 0;
  const y = { frontal: 68, central: 8, temporal: 0, parietal: -48, occipital: -82 }[node.region] ?? 0;
  const x = side * 62;
  return { x, y, z: scalpZ(x, y) + 8 };
}

function createTemplateNodes() {
  return Object.keys(electrodeCoords).map((name) => {
    const meta = classifyChannel(name);
    const pos = fixedNodePosition({ name, ...meta });
    return {
      name,
      ...meta,
      ...pos,
      influence: 0,
      size: 7,
      color: meta.hemisphere === 'left' ? '#2563eb' : meta.hemisphere === 'right' ? '#dc2626' : '#64748b'
    };
  });
}

function readFsInt24(view, offset) {
  return (view.getUint8(offset) << 16) | (view.getUint8(offset + 1) << 8) | view.getUint8(offset + 2);
}

function skipFsLine(view, offset) {
  let cursor = offset;
  while (cursor < view.byteLength && view.getUint8(cursor) !== 10) cursor += 1;
  return cursor + 1;
}

async function loadFreeSurferSurface(url) {
  const buffer = await fetch(url).then((res) => {
    if (!res.ok) throw new Error(`Failed to load brain surface: ${res.status}`);
    return res.arrayBuffer();
  });
  const view = new DataView(buffer);
  let offset = 0;
  const magic = readFsInt24(view, offset);
  offset += 3;
  if (magic !== 16777214) throw new Error(`Unsupported FreeSurfer surface magic: ${magic}`);
  offset = skipFsLine(view, offset);
  offset = skipFsLine(view, offset);
  const nVertices = view.getInt32(offset, false);
  offset += 4;
  const nFaces = view.getInt32(offset, false);
  offset += 4;
  const position = new Float32Array(nVertices * 3);
  for (let i = 0; i < position.length; i += 1) {
    position[i] = view.getFloat32(offset, false);
    offset += 4;
  }
  const index = new Uint32Array(nFaces * 3);
  for (let i = 0; i < index.length; i += 1) {
    index[i] = view.getInt32(offset, false);
    offset += 4;
  }
  return { isSurfaceMesh: true, nVertices, position, index };
}

function resetBrainObjects() {
  brain.edgeObjects.forEach((object) => {
    object.geometry?.dispose?.();
    object.material?.dispose?.();
    object.removeFromParent();
  });
  brain.signalObjects.forEach((object) => {
    object.traverse?.((child) => {
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    });
    object.removeFromParent();
  });
  brain.edgeJumpLabels.forEach((label) => label.remove());
  brain.edgeJumpLabels = [];
  brain.nodeObjects.forEach((object) => object.userData?.dispose?.());
  brain.nodeGlowObjects.forEach((object) => object.userData?.dispose?.());
  brain.labels.forEach((object) => {
    object.material?.map?.dispose?.();
    object.material?.dispose?.();
    object.removeFromParent();
  });
  brain.edgeObjects = [];
  brain.signalObjects = [];
  brain.nodeObjects = [];
  brain.nodeGlowObjects = [];
  brain.labels = [];
  brain.selectedGlowNode = null;
  brain.activeTooltipNode = null;
  els.nodeTooltip.classList.add('hidden');
  clearAllHighlightEffects();
}

function getNodeDotTexture() {
  if (brain.dotTexture) return brain.dotTexture;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 64, 64);
  ctx.beginPath();
  ctx.arc(32, 32, 7, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  brain.dotTexture = new THREE.CanvasTexture(canvas);
  return brain.dotTexture;
}

function getGlowTexture() {
  if (brain.glowTexture) return brain.glowTexture;
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 60);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.18, 'rgba(255,255,255,0.82)');
  gradient.addColorStop(0.48, 'rgba(210,210,210,0.2)');
  gradient.addColorStop(1, 'rgba(180,180,180,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  brain.glowTexture = new THREE.CanvasTexture(canvas);
  brain.glowTexture.minFilter = THREE.LinearFilter;
  brain.glowTexture.magFilter = THREE.LinearFilter;
  brain.glowTexture.generateMipmaps = false;
  return brain.glowTexture;
}

function regionId(region) {
  return {
    frontal: 1,
    central: 2,
    parietal: 3,
    occipital: 4,
    temporal: 5,
    auxiliary: 6
  }[region] || 0;
}

function hemisphereId(hemisphere) {
  return hemisphere === 'left' ? -1 : hemisphere === 'right' ? 1 : 0;
}

function regionMaskExpression(prefix = 'vBrainPosition') {
  return `
float highlightRegionMask(vec3 p, int region, int hemi, float centerX) {
  float hemisphereMask = 1.0;
  if (hemi < 0) {
    hemisphereMask = 1.0 - smoothstep(-7.0, 8.0, p.x);
  } else if (hemi > 0) {
    hemisphereMask = smoothstep(-8.0, 7.0, p.x);
  } else {
    hemisphereMask = 1.0 - smoothstep(12.0, 24.0, abs(p.x));
  }
  float x = abs(p.x);
  float lateralX = abs(p.x - centerX);
  float mask = 0.0;
  if (region == 1) {
    mask = smoothstep(12.0, 48.0, p.y) * smoothstep(-26.0, 10.0, p.z);
  } else if (region == 2) {
    mask = (1.0 - smoothstep(12.0, 30.0, abs(p.y - 4.0))) * smoothstep(20.0, 50.0, p.z) * (1.0 - smoothstep(62.0, 82.0, x));
  } else if (region == 3) {
    mask = smoothstep(-52.0, -22.0, p.y) * (1.0 - smoothstep(-12.0, 8.0, p.y)) * smoothstep(22.0, 52.0, p.z) * (1.0 - smoothstep(62.0, 84.0, x));
  } else if (region == 4) {
    mask = 1.0 - smoothstep(-82.0, -54.0, p.y);
  } else if (region == 5) {
    mask = smoothstep(16.0, 42.0, lateralX) * (1.0 - smoothstep(48.0, 78.0, p.z)) * (1.0 - smoothstep(48.0, 76.0, abs(p.y)));
  } else if (region == 6) {
    mask = smoothstep(26.0, 56.0, lateralX) * (1.0 - smoothstep(34.0, 68.0, abs(p.y))) * (1.0 - smoothstep(48.0, 76.0, p.z));
  }
  return clamp(mask * hemisphereMask, 0.0, 1.0);
}`;
}

function setBrainRegionHighlight(node) {
  clearBrainRegionHighlight();
  if (!node || !brain.canvas || !brain.surfaceData.length) return;
  const targetHemisphere = hemisphereId(node.hemisphere);
  const center = new THREE.Vector3(node.x, node.y, node.z);
  const radius = node.region === 'temporal' || node.region === 'auxiliary' ? 27 : 31;
  addRegionGlowSurface(node, radius);
  const glowPositions = [];
  const particlePositions = [];
  const normals = [];
  const seeds = [];

  brain.surfaceData.forEach((surface) => {
    if (targetHemisphere < 0 && surface.hemisphere !== 'left') return;
    if (targetHemisphere > 0 && surface.hemisphere !== 'right') return;
    const sourcePositions = surface.positions;
    const sourceNormals = surface.normals;
    const stride = Math.max(1, Math.floor(sourcePositions.length / 3 / 2400));
    for (let i = 0; i < sourcePositions.length; i += 3 * stride) {
      const x = sourcePositions[i];
      const y = sourcePositions[i + 1];
      const z = sourcePositions[i + 2];
      const distance = center.distanceTo(new THREE.Vector3(x, y, z));
      if (distance > radius) continue;
      const nx = sourceNormals[i] || 0;
      const ny = sourceNormals[i + 1] || 0;
      const nz = sourceNormals[i + 2] || 1;
      const jitter = 0.3 + Math.random() * 0.7;
      glowPositions.push(x + nx * 0.35, y + ny * 0.35, z + nz * 0.35);
      if (Math.random() <= 0.24 || particlePositions.length < 36) {
        particlePositions.push(x + nx * jitter, y + ny * jitter, z + nz * jitter);
        normals.push(nx, ny, nz);
        seeds.push(Math.random() * Math.PI * 2);
      }
      if (glowPositions.length / 3 >= 900) break;
    }
  });

  if (!glowPositions.length) return;
  const glowGeometry = new THREE.BufferGeometry();
  glowGeometry.setAttribute('position', new THREE.Float32BufferAttribute(glowPositions, 3));
  const glowSeeds = new Float32Array(glowPositions.length / 3);
  for (let i = 0; i < glowSeeds.length; i += 1) glowSeeds[i] = Math.random();
  glowGeometry.setAttribute('seed', new THREE.BufferAttribute(glowSeeds, 1));
  const glowMaterial = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      color: { value: new THREE.Color('#ffffff') }
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float seed;
      uniform float time;
      varying float vAlpha;
      void main() {
        float heightOpacity = smoothstep(-4.0, 78.0, position.z);
        float bottomFade = smoothstep(-28.0, -2.0, position.z);
        float brainOpacity = mix(0.03, 0.48, heightOpacity) * bottomFade;
        float shimmer = 0.72 + 0.28 * sin(time * 1.7 + seed * 13.0);
        vAlpha = brainOpacity * shimmer;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = (8.2 + 4.2 * shimmer) * (360.0 / max(160.0, -mvPosition.z));
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      varying float vAlpha;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float d = length(p);
        float glow = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(color, glow * vAlpha * 0.24);
      }
    `
  });
  const glow = new THREE.Points(glowGeometry, glowMaterial);
  glow.renderOrder = 7;
  glow.userData.startedAt = performance.now();
  brain.canvas.add_to_scene(glow);
  brain.highlightParticles.push(glow);

  if (!particlePositions.length) {
    if (brain.canvas) brain.canvas.needsUpdate = true;
    return;
  }
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.Float32BufferAttribute(particlePositions, 3));
  particleGeometry.setAttribute('normalLift', new THREE.Float32BufferAttribute(normals, 3));
  particleGeometry.setAttribute('seed', new THREE.Float32BufferAttribute(seeds, 1));
  const particleMaterial = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      color: { value: new THREE.Color('#ffffff') }
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute vec3 normalLift;
      attribute float seed;
      uniform float time;
      varying float vAlpha;
      void main() {
        float progress = fract(seed + time * (0.16 + seed * 0.018));
        float fadeIn = smoothstep(0.0, 0.16, progress);
        float fadeOut = 1.0 - smoothstep(0.34, 1.0, progress);
        vAlpha = fadeIn * fadeOut;
        vec3 displaced = position + normalize(normalLift) * (0.35 + progress * 7.4);
        vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = (3.8 + 2.4 * (1.0 - progress)) * (360.0 / max(160.0, -mvPosition.z));
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      varying float vAlpha;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float d = length(p);
        float core = smoothstep(0.42, 0.0, d);
        float halo = smoothstep(0.5, 0.0, d) * 0.42;
        gl_FragColor = vec4(color, (core + halo) * vAlpha * 0.36);
      }
    `
  });
  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.renderOrder = 9;
  particles.userData.startedAt = performance.now();
  brain.canvas.add_to_scene(particles);
  brain.highlightParticles.push(particles);
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function addRegionGlowSurface(node, radius) {
  const targetHemisphere = hemisphereId(node.hemisphere);
  const center = new THREE.Vector3(node.x, node.y, node.z);
  brain.surfaceObjects.forEach((surface) => {
    const hemisphere = surface.userData?.hemisphere
      || (surface.name?.includes('left') ? 'left' : surface.name?.includes('right') ? 'right' : null);
    if (targetHemisphere < 0 && hemisphere !== 'left') return;
    if (targetHemisphere > 0 && hemisphere !== 'right') return;
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        highlightCenter: { value: center },
        highlightRadius: { value: radius },
        highlightHemisphere: { value: targetHemisphere }
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      vertexShader: `
        varying vec3 vBrainPosition;
        void main() {
          vBrainPosition = position;
          vec3 displaced = position + normal * 0.42;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 highlightCenter;
        uniform float highlightRadius;
        uniform int highlightHemisphere;
        varying vec3 vBrainPosition;
        void main() {
          float heightOpacity = smoothstep(-4.0, 78.0, vBrainPosition.z);
          float bottomFade = smoothstep(-28.0, -2.0, vBrainPosition.z);
          float brainOpacity = mix(0.03, 0.48, heightOpacity) * bottomFade;
          float distanceMask = 1.0 - smoothstep(highlightRadius * 0.42, highlightRadius, distance(vBrainPosition, highlightCenter));
          float breathe = 0.82 + 0.18 * sin(time * 1.25);
          float alpha = distanceMask * brainOpacity * breathe * 0.34;
          gl_FragColor = vec4(vec3(1.0), alpha);
        }
      `
    });
    const mesh = new THREE.Mesh(surface.geometry.clone(), material);
    mesh.renderOrder = 6;
    brain.canvas.add_to_scene(mesh);
    brain.regionGlowSurfaces.push(mesh);
  });
}

function clearBrainRegionHighlight() {
  brain.highlightParticles.forEach((particles) => {
    particles.geometry?.dispose?.();
    particles.material?.dispose?.();
    particles.removeFromParent();
  });
  brain.highlightParticles = [];
  brain.regionGlowSurfaces.forEach((surface) => {
    surface.geometry?.dispose?.();
    surface.material?.dispose?.();
    surface.removeFromParent();
  });
  brain.regionGlowSurfaces = [];
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function clearAllHighlightEffects() {
  clearBrainRegionHighlight();
  clearSelectedGlowParticles();
  setSelectedNodeGlow(null);
  clearEdgeJumpLabels();
}

function isInHighlightedRegion(point, node, surfaceHemisphere) {
  const targetHemisphere = hemisphereId(node.hemisphere);
  const isMidline = targetHemisphere === 0 || node.hemisphere === 'midline';
  if (isMidline && Math.abs(point.x) > 18) return false;
  const x = Math.abs(point.x);
  const lateralX = surfaceHemisphere === 'left' ? Math.abs(point.x + 42) : Math.abs(point.x - 42);
  switch (node.region) {
    case 'frontal':
      return point.y > 22 && point.z > -18;
    case 'central':
      return point.y > -16 && point.y < 26 && point.z > 18 && x < 70;
    case 'parietal':
      return point.y > -58 && point.y < -18 && point.z > 18 && x < 74;
    case 'occipital':
      return point.y < -58 && point.z > -14;
    case 'temporal':
      return lateralX > 18 && point.z > -30 && point.z < 48 && point.y > -46 && point.y < 42;
    case 'auxiliary':
      return lateralX > 28 && point.z > -24 && point.y > -34 && point.y < 34;
    default:
      return isMidline ? point.z > 2 : false;
  }
}

function updateBrainRegionParticles() {
  if (!brain.highlightParticles.length && !brain.regionGlowSurfaces.length) return;
  const now = performance.now();
  brain.regionGlowSurfaces.forEach((surface) => {
    if (surface.material?.uniforms?.time) {
      surface.material.uniforms.time.value = now / 1000;
    }
  });
  brain.highlightParticles.forEach((particles) => {
    if (particles.material?.uniforms?.time) {
      particles.material.uniforms.time.value = (now - particles.userData.startedAt) / 1000;
    } else if (particles.material) {
      particles.material.opacity = 0.32 + 0.075 * Math.sin(now * 0.0022);
    }
  });
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function applyBrainOpacityGradient(material, hemisphere) {
  const centerX = hemisphere === 'left' ? -33 : 33;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.opacityCenterX = { value: centerX };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vBrainPosition;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvBrainPosition = transformed;');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
uniform float opacityCenterX;
varying vec3 vBrainPosition;`
      )
      .replace(
        '#include <dithering_fragment>',
        `float heightOpacity = smoothstep(-4.0, 78.0, vBrainPosition.z);
float bottomFade = smoothstep(-28.0, -2.0, vBrainPosition.z);
float brainOpacity = mix(0.03, 0.48, heightOpacity) * bottomFade;
gl_FragColor.a *= brainOpacity;
#include <dithering_fragment>`
      );
  };
  material.needsUpdate = true;
}

async function addBrainSurface() {
  brain.surfaceObjects.forEach((object) => object.userData?.dispose?.());
  brain.surfaceData = [];
  const surfaces = [
    ['left', await loadFreeSurferSurface(lhPialUrl)],
    ['right', await loadFreeSurferSurface(rhPialUrl)]
  ];
  brain.surfaceObjects = surfaces.map(([hemisphere, mesh]) => {
    const inst = brain.canvas.add_object({
      type: 'free',
      ...mesh,
      subject_code: 'EEG_TEMPLATE',
      hemisphere,
      fileName: 'pial'
    });
    inst.object.geometry.computeVertexNormals();
    brain.surfaceData.push({
      hemisphere,
      positions: inst.object.geometry.attributes.position.array,
      normals: inst.object.geometry.attributes.normal.array
    });
    inst.object.material.color.set('#ffffff');
    inst.object.material.opacity = 1;
    inst.object.material.transparent = true;
    inst.object.material.depthWrite = false;
    inst.object.material.side = THREE.DoubleSide;
    applyBrainOpacityGradient(inst.object.material, hemisphere);
    inst.object.userData.hemisphere = hemisphere;
    inst.object.renderOrder = -10;
    return inst.object;
  });
}

function projectNodeToSurface(node) {
  const candidates = node.hemisphere === 'left'
    ? brain.surfaceData.filter((surface) => surface.hemisphere === 'left')
    : node.hemisphere === 'right'
      ? brain.surfaceData.filter((surface) => surface.hemisphere === 'right')
      : brain.surfaceData;
  let bestSurface = null;
  let bestIndex = 0;
  let bestDistance = Infinity;
  candidates.forEach((surface) => {
    const positions = surface.positions;
    for (let i = 0; i < positions.length; i += 3) {
      const dx = positions[i] - node.x;
      const dy = positions[i + 1] - node.y;
      const z = positions[i + 2];
      const dist = dx * dx + dy * dy + Math.max(0, 70 - z) * 55;
      if (dist < bestDistance) {
        bestDistance = dist;
        bestSurface = surface;
        bestIndex = i;
      }
    }
  });
  if (!bestSurface) return node;
  const positions = bestSurface.positions;
  const normals = bestSurface.normals;
  const cx = bestSurface.hemisphere === 'left' ? -33 : 33;
  const radial = new THREE.Vector3(
    positions[bestIndex] - cx,
    positions[bestIndex + 1],
    positions[bestIndex + 2] - 10
  ).normalize();
  let nx = normals[bestIndex] || radial.x;
  let ny = normals[bestIndex + 1] || radial.y;
  let nz = normals[bestIndex + 2] || radial.z;
  if (nz < 0.18) {
    nx = radial.x;
    ny = radial.y;
    nz = Math.max(0.28, radial.z);
  }
  const normal = new THREE.Vector3(nx, ny, nz).normalize();
  const offset = 1.35;
  return {
    ...node,
    x: positions[bestIndex] + normal.x * offset,
    y: positions[bestIndex + 1] + normal.y * offset,
    z: positions[bestIndex + 2] + normal.z * offset,
    normal: { x: normal.x, y: normal.y, z: normal.z }
  };
}

function createTextSprite(text, position) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 192;
  const ctx = canvas.getContext('2d');
  ctx.font = '700 76px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.fillText(text, 256, 96);
  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  const normal = position.normal ? new THREE.Vector3(position.normal.x, position.normal.y, position.normal.z) : new THREE.Vector3(0, 0, 1);
  sprite.position.set(
    position.x + normal.x * 5.5,
    position.y + normal.y * 5.5,
    position.z + normal.z * 5.5
  );
  sprite.scale.set(18, 7.2, 1);
  sprite.renderOrder = 20;
  sprite.userData.eegNodeName = text;
  sprite.userData.baseOpacity = material.opacity;
  brain.canvas.add_to_scene(sprite);
  brain.labels.push(sprite);
}

function addElectrodeNode(node, index) {
  const normal = node.normal ? new THREE.Vector3(node.normal.x, node.normal.y, node.normal.z) : new THREE.Vector3(0, 0, 1);
  const glowMaterial = new THREE.SpriteMaterial({
    map: getGlowTexture(),
    color: '#ffffff',
    transparent: true,
    opacity: 0.24,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });
  const glow = new THREE.Sprite(glowMaterial);
  glow.name = `EEG node glow ${index + 1} ${node.name}`;
  glow.position.set(
    node.x + normal.x * 0.65,
    node.y + normal.y * 0.65,
    node.z + normal.z * 0.65
  );
  glow.scale.set(6.9, 6.9, 1);
  glow.userData.eegNodeName = node.name;
  glow.userData.baseOpacity = glowMaterial.opacity;
  glow.userData.dispose = () => {
    glowMaterial.dispose();
    glow.removeFromParent();
  };
  glow.renderOrder = 11;
  brain.canvas.add_to_scene(glow);
  brain.nodeGlowObjects.push(glow);

  const material = new THREE.SpriteMaterial({
    map: getNodeDotTexture(),
    color: '#ffffff',
    transparent: true,
    opacity: 0.96,
    depthTest: true,
    depthWrite: false
  });
  const object = new THREE.Sprite(material);
  object.name = `EEG node ${index + 1} ${node.name}`;
  object.position.set(node.x, node.y, node.z);
  object.scale.set(6, 6, 1);
  object.userData.eegNode = node;
  object.userData.baseOpacity = material.opacity;
  object.userData.dispose = () => {
    material.dispose();
    object.removeFromParent();
  };
  object.renderOrder = 10;
  brain.canvas.add_to_scene(object);
  brain.nodeObjects.push(object);
  createTextSprite(node.name, node);
}

function makeConnectionCurve(source, target, strength, curveOffset = 0) {
  const sourceNormal = source.normal
    ? new THREE.Vector3(source.normal.x, source.normal.y, source.normal.z).normalize()
    : new THREE.Vector3(source.x, source.y, source.z - 18).normalize();
  const targetNormal = target.normal
    ? new THREE.Vector3(target.normal.x, target.normal.y, target.normal.z).normalize()
    : new THREE.Vector3(target.x, target.y, target.z - 18).normalize();
  const start = new THREE.Vector3(source.x, source.y, source.z);
  const end = new THREE.Vector3(target.x, target.y, target.z);
  const mid = start.clone().add(end).multiplyScalar(0.5);
  const outward = sourceNormal.clone().add(targetNormal).normalize();
  if (!Number.isFinite(outward.x)) outward.copy(mid.clone().sub(new THREE.Vector3(0, 0, 18)).normalize());
  const side = new THREE.Vector3().subVectors(end, start).cross(outward).normalize();
  if (!Number.isFinite(side.x)) side.set(0, 1, 0);
  const distance = start.distanceTo(end);
  mid.add(outward.multiplyScalar(Math.min(7.2, 0.032 * distance + strength * 1.25 + 1.25)));
  mid.add(side.multiplyScalar(curveOffset * 0.45));
  return new THREE.QuadraticBezierCurve3(start, mid, end);
}

function makeConnectionGeometry(curve, strength) {
  return new THREE.TubeGeometry(curve, 40, 0.045 + strength * 0.22, 5, false);
}

function addEdgeObject(edge, nodeById, curveOffset = 0) {
  const source = nodeById.get(edgeEndpointName(edge.source));
  const target = nodeById.get(edgeEndpointName(edge.target));
  if (!source || !target) return;
  const strength = Math.max(0.08, Math.min(1, Number(edge.score || edge.strength || Math.abs(edge.value) || 0.2)));
  const curve = makeConnectionCurve(source, target, strength, curveOffset);
  const material = new THREE.MeshBasicMaterial({
    color: '#ffffff',
    transparent: true,
    opacity: 0.34 + strength * 0.42,
    depthTest: true,
    depthWrite: false
  });
  const mesh = new THREE.Mesh(makeConnectionGeometry(curve, strength), material);
  mesh.userData.edge = edge;
  mesh.userData.sourceName = source.name;
  mesh.userData.targetName = target.name;
  mesh.userData.baseOpacity = material.opacity;
  mesh.renderOrder = 5;
  brain.canvas.add_to_scene(mesh);
  brain.edgeObjects.push(mesh);
  addConnectionSignalObject(curve, strength, source.name, target.name);
}

function addConnectionSignalObject(curve, strength, sourceName, targetName) {
  const segments = 96;
  const positions = new Float32Array(segments * 2 * 3);
  const pathT = new Float32Array(segments * 2);
  for (let i = 0; i < segments; i += 1) {
    const t0 = i / segments;
    const t1 = (i + 1) / segments;
    const p0 = curve.getPoint(t0);
    const p1 = curve.getPoint(t1);
    positions[i * 6] = p0.x;
    positions[i * 6 + 1] = p0.y;
    positions[i * 6 + 2] = p0.z;
    positions[i * 6 + 3] = p1.x;
    positions[i * 6 + 4] = p1.y;
    positions[i * 6 + 5] = p1.z;
    pathT[i * 2] = t0;
    pathT[i * 2 + 1] = t1;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('pathT', new THREE.BufferAttribute(pathT, 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      strength: { value: strength },
      fade: { value: 1 },
      color: { value: new THREE.Color('#ffffff') }
    },
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float pathT;
      uniform float time;
      uniform float strength;
      uniform float fade;
      varying float vAlpha;
      void main() {
        float phase = fract(time * (0.24 + strength * 0.26));
        float d = min(abs(pathT - phase), 1.0 - abs(pathT - phase));
        float head = exp(-d * d / (0.0022 + strength * 0.0018));
        float tailDistance = mod(phase - pathT + 1.0, 1.0);
        float tail = exp(-tailDistance * tailDistance / 0.0065) * 0.28;
        vAlpha = clamp((head + tail) * (0.72 + strength * 0.58) * fade, 0.0, 0.82);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      varying float vAlpha;
      void main() {
        gl_FragColor = vec4(color, vAlpha);
      }
    `
  });
  const signal = new THREE.LineSegments(geometry, material);
  signal.renderOrder = 22;
  signal.userData.startedAt = performance.now();
  signal.userData.sourceName = sourceName;
  signal.userData.targetName = targetName;
  signal.userData.baseOpacity = 1;
  signal.userData.focusFade = 1;
  brain.canvas.add_to_scene(signal);
  brain.signalObjects.push(signal);
  const haloMaterial = material.clone();
  haloMaterial.uniforms = {
    time: { value: 0 },
    strength: { value: strength },
    fade: { value: 0.46 },
    color: { value: new THREE.Color('#ffffff') }
  };
  haloMaterial.blending = THREE.AdditiveBlending;
  const halo = new THREE.LineSegments(geometry.clone(), haloMaterial);
  halo.renderOrder = 21;
  halo.scale.set(1.003, 1.003, 1.003);
  halo.userData.startedAt = performance.now();
  halo.userData.isSignalHalo = true;
  halo.userData.sourceName = sourceName;
  halo.userData.targetName = targetName;
  halo.userData.baseOpacity = 1;
  halo.userData.focusFade = 1;
  brain.canvas.add_to_scene(halo);
  brain.signalObjects.push(halo);
}

function updateConnectionSignals() {
  if (!brain.signalObjects.length) return;
  const now = performance.now();
  brain.signalObjects.forEach((signal) => {
    if (signal.material?.uniforms?.time) {
      signal.material.uniforms.time.value = (now - signal.userData.startedAt) / 1000;
      const baseFade = signal.userData.isSignalHalo ? 0.46 : 1;
      signal.material.uniforms.fade.value = baseFade * (signal.userData.focusFade ?? 1);
    }
  });
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function drawBrainNetwork(nodes, edges = []) {
  if (!brain.canvas) return;
  resetBrainObjects();
  const projectedNodes = nodes.map((node) => projectNodeToSurface(node));
  const nodeById = new Map(projectedNodes.map((node) => [node.name, node]));
  const edgeCounts = new Map();
  edges.forEach((edge) => {
    [edgeEndpointName(edge.source), edgeEndpointName(edge.target)].forEach((name) => {
      edgeCounts.set(name, (edgeCounts.get(name) || 0) + 1);
    });
  });
  const edgeSlots = new Map();
  edges.forEach((edge) => {
    const sourceName = edgeEndpointName(edge.source);
    const targetName = edgeEndpointName(edge.target);
    const key = edgeCounts.get(sourceName) >= edgeCounts.get(targetName) ? sourceName : targetName;
    const slot = edgeSlots.get(key) || 0;
    const total = Math.max(1, edgeCounts.get(key) || 1);
    edgeSlots.set(key, slot + 1);
    const centeredSlot = slot - (total - 1) / 2;
    addEdgeObject(edge, nodeById, centeredSlot * 1.2);
  });
  projectedNodes.forEach((node, index) => addElectrodeNode(node, index));
  brain.nodes = projectedNodes;
  brain.edges = edges;
  brain.canvas.needsUpdate = true;
}

function setBrainCamera() {
  const camera = brain.canvas.mainCamera;
  camera.position.set(0, 0, 500);
  camera.up.set(0, 1, 0);
  camera.lookAt(0, 0, 42);
  camera.zoom = 1.02;
  camera.updateProjectionMatrix();
  brain.canvas.trackball?.lookAt?.({ x: 0, y: 0, z: 42 });
  brain.canvas.trackball?.update?.();
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2;
}

function keepModelTopReadable(quaternion) {
  const topVector = new THREE.Vector3(0, 0, 1).applyQuaternion(quaternion);
  topVector.z = 0;
  if (topVector.lengthSq() < 0.015) {
    topVector.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion));
    topVector.z = 0;
  }
  if (topVector.lengthSq() < 0.015) return quaternion;
  topVector.normalize();
  if (topVector.y >= 0.22) return quaternion;

  const targetY = 0.46;
  const targetX = Math.sign(topVector.x || 1) * Math.sqrt(1 - targetY * targetY);
  const target = new THREE.Vector3(targetX, targetY, 0).normalize();
  const angle = Math.atan2(
    topVector.x * target.y - topVector.y * target.x,
    topVector.dot(target)
  );
  return new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle)
    .multiply(quaternion);
}

function keepHemispheresReadable(quaternion) {
  const rightVector = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion);
  rightVector.z = 0;
  if (rightVector.lengthSq() < 0.015) return quaternion;
  rightVector.normalize();
  if (rightVector.x >= 0.18) return quaternion;

  const targetX = 0.42;
  const targetY = Math.sign(rightVector.y || 1) * Math.sqrt(1 - targetX * targetX);
  const target = new THREE.Vector3(targetX, targetY, 0).normalize();
  const angle = Math.atan2(
    rightVector.x * target.y - rightVector.y * target.x,
    rightVector.dot(target)
  );
  return new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle)
    .multiply(quaternion);
}

function keepBrainOrientationReadable(quaternion) {
  return keepModelTopReadable(keepHemispheresReadable(keepModelTopReadable(keepHemispheresReadable(quaternion))));
}

function updateBrainFocusAnimation() {
  if (!brain.focusAnimation || !brain.canvas) return;
  const now = performance.now();
  const progress = Math.min(1, (now - brain.focusAnimation.startedAt) / brain.focusAnimation.duration);
  const eased = easeInOutCubic(progress);
  brain.canvas.origin.quaternion.copy(brain.focusAnimation.fromQuaternion).slerp(brain.focusAnimation.toQuaternion, eased);
  brain.canvas.origin.position.lerpVectors(brain.focusAnimation.fromPosition, brain.focusAnimation.toPosition, eased);
  brain.canvas.mainCamera.zoom = THREE.MathUtils.lerp(brain.focusAnimation.fromZoom, brain.focusAnimation.toZoom, eased);
  brain.canvas.mainCamera.updateProjectionMatrix();
  brain.canvas.needsUpdate = true;
  if (progress >= 1) {
    const onComplete = brain.focusAnimation.onComplete;
    brain.focusAnimation = null;
    onComplete?.();
  }
}

function findRenderedNode(name) {
  return brain.nodes.find((node) => node.name === name);
}

function setNodeLabelsVisible(visible) {
  brain.labels.forEach((label) => {
    label.visible = visible;
  });
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function setNodeLabelVisible(name, visible) {
  brain.labels.forEach((label) => {
    if (label.userData.eegNodeName === name) label.visible = visible;
  });
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function setSelectedNodeGlow(name) {
  brain.selectedGlowNode = name;
  clearSelectedGlowParticles();
  brain.nodeGlowObjects.forEach((glow) => {
    const selected = glow.userData.eegNodeName === name;
    glow.material.opacity = selected ? 0.46 : 0.24;
    glow.scale.setScalar(selected ? 9.2 : 6.9);
    glow.renderOrder = selected ? 19 : 11;
  });
  if (name) createSelectedGlowParticles(name);
  applyFocusFade(name);
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function focusFadeForNode(name) {
  if (!brain.focusedNodeName) return 1;
  if (name === brain.focusedNodeName) return 1;
  const focused = findRenderedNode(brain.focusedNodeName);
  const node = findRenderedNode(name);
  if (!focused || !node) return 1;
  const distance = new THREE.Vector3(focused.x, focused.y, focused.z).distanceTo(new THREE.Vector3(node.x, node.y, node.z));
  return THREE.MathUtils.clamp(1 - distance / 98, 0.08, 0.58);
}

function applyFocusFade(name) {
  brain.focusedNodeName = name;
  brain.nodeObjects.forEach((object) => {
    const nodeName = object.userData.eegNode?.name;
    object.material.opacity = (object.userData.baseOpacity ?? 0.96) * focusFadeForNode(nodeName);
  });
  brain.labels.forEach((label) => {
    label.material.opacity = (label.userData.baseOpacity ?? 1) * focusFadeForNode(label.userData.eegNodeName);
  });
  brain.nodeGlowObjects.forEach((glow) => {
    if (glow.userData.eegNodeName === brain.selectedGlowNode) {
      glow.material.opacity = 0.5;
      glow.scale.setScalar(9.6);
      glow.renderOrder = 19;
    } else {
      glow.material.opacity = (glow.userData.baseOpacity ?? 0.24) * focusFadeForNode(glow.userData.eegNodeName);
      glow.scale.setScalar(6.9);
      glow.renderOrder = 11;
    }
  });
  brain.edgeObjects.forEach((edge) => {
    const sourceFade = focusFadeForNode(edge.userData.sourceName);
    const targetFade = focusFadeForNode(edge.userData.targetName);
    edge.material.opacity = (edge.userData.baseOpacity ?? 0.42) * Math.max(sourceFade, targetFade);
  });
  brain.signalObjects.forEach((signal) => {
    const sourceFade = focusFadeForNode(signal.userData.sourceName);
    const targetFade = focusFadeForNode(signal.userData.targetName);
    signal.userData.focusFade = Math.max(sourceFade, targetFade);
  });
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function clearSelectedGlowParticles() {
  if (!brain.selectedGlowParticles) return;
  brain.selectedGlowParticles.geometry?.dispose?.();
  brain.selectedGlowParticles.material?.dispose?.();
  brain.selectedGlowParticles.removeFromParent();
  brain.selectedGlowParticles = null;
}

function createSelectedGlowParticles(name) {
  const node = findRenderedNode(name);
  if (!node || !brain.canvas) return;
  const count = 56;
  const positions = new Float32Array(count * 3);
  const dirs = new Float32Array(count * 3);
  const seeds = new Float32Array(count);
  const normal = node.normal ? new THREE.Vector3(node.normal.x, node.normal.y, node.normal.z).normalize() : new THREE.Vector3(0, 0, 1);
  for (let i = 0; i < count; i += 1) {
    const angle = Math.random() * Math.PI * 2;
    const tangent = new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0).normalize();
    const dir = normal.clone().multiplyScalar(0.78 + Math.random() * 0.35).addScaledVector(tangent, (Math.random() - 0.5) * 0.55).normalize();
    positions[i * 3] = node.x;
    positions[i * 3 + 1] = node.y;
    positions[i * 3 + 2] = node.z;
    dirs[i * 3] = dir.x;
    dirs[i * 3 + 1] = dir.y;
    dirs[i * 3 + 2] = dir.z;
    seeds[i] = Math.random();
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('direction', new THREE.BufferAttribute(dirs, 3));
  geometry.setAttribute('seed', new THREE.BufferAttribute(seeds, 1));
  const material = new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 }, color: { value: new THREE.Color('#ffffff') } },
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute vec3 direction;
      attribute float seed;
      uniform float time;
      varying float vAlpha;
      void main() {
        float p = fract(time * (0.12 + seed * 0.05) + seed);
        float fade = smoothstep(0.0, 0.18, p) * (1.0 - smoothstep(0.62, 1.0, p));
        vAlpha = fade * 0.42;
        vec3 displaced = position + direction * (p * 15.0);
        vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = (2.0 + 2.2 * (1.0 - p)) * (360.0 / max(160.0, -mvPosition.z));
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      varying float vAlpha;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float d = length(p);
        float alpha = smoothstep(0.5, 0.0, d) * vAlpha;
        gl_FragColor = vec4(color, alpha);
      }
    `
  });
  const particles = new THREE.Points(geometry, material);
  particles.renderOrder = 23;
  particles.userData.nodeName = name;
  particles.userData.startedAt = performance.now();
  brain.canvas.add_to_scene(particles);
  brain.selectedGlowParticles = particles;
}

function updateSelectedNodeGlow() {
  if (!brain.selectedGlowNode) return;
  const now = performance.now();
  const pulse = 0.5 + 0.5 * Math.sin(now * 0.0048);
  brain.nodeGlowObjects.forEach((glow) => {
    if (glow.userData.eegNodeName !== brain.selectedGlowNode) return;
    glow.material.opacity = 0.46 + pulse * 0.2;
    glow.scale.setScalar(9.2 + pulse * 1.8);
  });
  if (brain.selectedGlowParticles?.material?.uniforms?.time) {
    brain.selectedGlowParticles.material.uniforms.time.value = (now - brain.selectedGlowParticles.userData.startedAt) / 1000;
  }
  if (brain.canvas) brain.canvas.needsUpdate = true;
}

function clearEdgeJumpLabels() {
  brain.edgeJumpLabels.forEach((label) => label.remove());
  brain.edgeJumpLabels = [];
}

function createEdgeJumpLabels(node) {
  clearEdgeJumpLabels();
  if (!node) return;
  const links = brain.edges
    .filter((edge) => edgeEndpointName(edge.source) === node.name || edgeEndpointName(edge.target) === node.name)
    .slice(0, 10);
  links.forEach((edge) => {
    const sourceName = edgeEndpointName(edge.source);
    const targetName = edgeEndpointName(edge.target);
    const otherName = sourceName === node.name ? targetName : sourceName;
    const otherNode = findRenderedNode(otherName);
    if (!otherNode) return;
    const label = document.createElement('button');
    label.className = 'edge-jump-label';
    label.textContent = otherName;
    label.type = 'button';
    label.userData = { from: node.name, to: otherName };
    label.addEventListener('mousedown', (event) => event.stopPropagation());
    label.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const target = findRenderedNode(otherName);
      if (!target) return;
      focusNode(target, () => showNodeTooltip(target, event));
    });
    els.networkChart.appendChild(label);
    brain.edgeJumpLabels.push(label);
  });
  updateEdgeJumpLabels();
}

function updateEdgeJumpLabels() {
  if (!brain.edgeJumpLabels.length || !brain.canvas) return;
  brain.canvas.origin.updateMatrixWorld(true);
  brain.canvas.mainCamera.updateMatrixWorld(true);
  const tooltipHidden = els.nodeTooltip.classList.contains('hidden');
  const tooltipRect = tooltipHidden ? null : els.nodeTooltip.getBoundingClientRect();
  const chartRect = els.networkChart.getBoundingClientRect();
  brain.edgeJumpLabels.forEach((label, index) => {
    const from = findRenderedNode(label.userData.from);
    const to = findRenderedNode(label.userData.to);
    if (!from || !to || tooltipHidden) {
      label.classList.add('hidden');
      return;
    }
    const fromScreen = new THREE.Vector3(from.x, from.y, from.z)
      .applyMatrix4(brain.canvas.origin.matrixWorld)
      .project(brain.canvas.mainCamera);
    const toScreen = new THREE.Vector3(to.x, to.y, to.z)
      .applyMatrix4(brain.canvas.origin.matrixWorld)
      .project(brain.canvas.mainCamera);
    if (fromScreen.z < -1 || fromScreen.z > 1 || toScreen.z < -1 || toScreen.z > 1) {
      label.classList.add('hidden');
      return;
    }
    const fromX = (fromScreen.x * 0.5 + 0.5) * els.networkChart.clientWidth;
    const fromY = (-fromScreen.y * 0.5 + 0.5) * els.networkChart.clientHeight;
    const toX = (toScreen.x * 0.5 + 0.5) * els.networkChart.clientWidth;
    const toY = (-toScreen.y * 0.5 + 0.5) * els.networkChart.clientHeight;
    const dx = toX - fromX;
    const dy = toY - fromY;
    const length = Math.max(1, Math.hypot(dx, dy));
    let nx = -dy / length;
    let ny = dx / length;
    if (ny > 0) {
      nx *= -1;
      ny *= -1;
    }
    const t = 0.24 + (index % 3) * 0.06;
    let x = fromX + dx * t + nx * 14;
    let y = fromY + dy * t + ny * 14;
    if (tooltipRect) {
      const tooltipLeft = tooltipRect.left - chartRect.left - 12;
      const tooltipRight = tooltipRect.right - chartRect.left + 12;
      const tooltipTop = tooltipRect.top - chartRect.top - 10;
      const tooltipBottom = tooltipRect.bottom - chartRect.top + 10;
      if (x > tooltipLeft && x < tooltipRight && y > tooltipTop && y < tooltipBottom) {
        const aboveY = tooltipTop - 14;
        const belowY = tooltipBottom + 14;
        y = Math.abs(y - aboveY) <= Math.abs(y - belowY) ? aboveY : belowY;
        x = Math.max(tooltipLeft - 42, Math.min(tooltipRight + 42, x + nx * 34));
      }
    }
    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
    if (angle > 90 || angle < -90) angle += 180;
    label.textContent = `${toX < fromX ? '<' : '>'} ${label.userData.to}`;
    label.style.left = `${Math.max(8, Math.min(els.networkChart.clientWidth - 96, x))}px`;
    label.style.top = `${Math.max(8, Math.min(els.networkChart.clientHeight - 26, y))}px`;
    label.style.transform = `translate(-50%, -50%) rotate(${angle}deg)`;
    label.classList.remove('hidden');
  });
}

function updateActiveTooltipPosition() {
  if (!brain.canvas || !brain.activeTooltipNode || els.nodeTooltip.classList.contains('hidden')) return;
  const node = findRenderedNode(brain.activeTooltipNode);
  if (!node) return;
  brain.canvas.origin.updateMatrixWorld(true);
  brain.canvas.mainCamera.updateMatrixWorld(true);
  const worldPosition = new THREE.Vector3(node.x, node.y, node.z).applyMatrix4(brain.canvas.origin.matrixWorld);
  const screenPosition = worldPosition.clone().project(brain.canvas.mainCamera);
  const x = (screenPosition.x * 0.5 + 0.5) * els.networkChart.clientWidth;
  const y = (-screenPosition.y * 0.5 + 0.5) * els.networkChart.clientHeight;
  if (screenPosition.z < -1 || screenPosition.z > 1) {
    els.nodeTooltip.classList.add('hidden');
    setNodeLabelVisible(brain.activeTooltipNode, true);
    brain.activeTooltipNode = null;
    clearBrainRegionHighlight();
    setSelectedNodeGlow(null);
    clearEdgeJumpLabels();
    return;
  }
  const tooltipWidth = 285;
  const preferLeft = x > els.networkChart.clientWidth - tooltipWidth - 120;
  const tooltipX = preferLeft ? x - tooltipWidth - 14 : x + 14;
  els.nodeTooltip.style.left = `${Math.max(12, Math.min(tooltipX, els.networkChart.clientWidth - tooltipWidth - 12))}px`;
  els.nodeTooltip.style.top = `${Math.max(12, Math.min(y + 14, els.networkChart.clientHeight - 190))}px`;
}

function cancelBrainFocusAnimation() {
  brain.focusAnimation = null;
}

function focusNode(node, onComplete = null) {
  if (!brain.canvas || !node) return;
  const normal = node.normal
    ? new THREE.Vector3(node.normal.x, node.normal.y, node.normal.z).normalize()
    : new THREE.Vector3(node.x, node.y, node.z - 18).normalize();
  const fromQuaternion = brain.canvas.origin.quaternion.clone();
  const currentNormal = normal.clone().applyQuaternion(fromQuaternion).normalize();
  const focusRotation = new THREE.Quaternion().setFromUnitVectors(currentNormal, new THREE.Vector3(0, 0, 1));
  const toQuaternion = keepBrainOrientationReadable(focusRotation.multiply(fromQuaternion));
  const rotatedNode = new THREE.Vector3(node.x, node.y, node.z).applyQuaternion(toQuaternion);
  const leftOffset = -20;
  const toPosition = new THREE.Vector3(leftOffset - rotatedNode.x, -rotatedNode.y, 42 - rotatedNode.z);
  brain.focusAnimation = {
    startedAt: performance.now(),
    duration: 760,
    fromQuaternion,
    toQuaternion,
    fromPosition: brain.canvas.origin.position.clone(),
    toPosition,
    fromZoom: brain.canvas.mainCamera.zoom,
    toZoom: 2.78,
    onComplete
  };
}

function arcballVector(event) {
  const rect = brain.canvas.main_canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  const lengthSq = x * x + y * y;
  if (lengthSq > 1) {
    return new THREE.Vector3(x, y, 0).normalize();
  }
  return new THREE.Vector3(x, y, Math.sqrt(1 - lengthSq)).normalize();
}

async function initBrainScene() {
  const tooltip = els.nodeTooltip;
  els.networkChart.innerHTML = '';
  els.networkChart.appendChild(tooltip);
  brain.raycaster = new THREE.Raycaster();
  brain.pointer = new THREE.Vector2();
  brain.canvas = new ViewerCanvas(
    els.networkChart,
    els.networkChart.clientWidth || 960,
    els.networkChart.clientHeight || 640,
    0,
    false,
    false,
    true
  );
  brain.canvas.set_state('target_subject', 'EEG_TEMPLATE');
  brain.canvas.set_state('surface_type', 'pial');
  brain.canvas.set_state('material_type_left', 'normal');
  brain.canvas.set_state('material_type_right', 'normal');
  brain.canvas.set_state('surface_opacity_left', 0.48);
  brain.canvas.set_state('surface_opacity_right', 0.48);
  brain.canvas.init_subject('EEG_TEMPLATE');
  brain.canvas.shared_data.set('EEG_TEMPLATE', {
    matrices: {
      tkrRAS_Scanner: new THREE.Matrix4().identity(),
      tkrRAS_MNI305: new THREE.Matrix4().identity()
    }
  });
  if (brain.canvas.compass?.container) brain.canvas.compass.container.visible = false;
  brain.canvas.setBackground({ color: '#000000' });
  try {
    await addBrainSurface();
    setBrainCamera();
    const renderLoop = () => {
      updateBrainFocusAnimation();
      updateActiveTooltipPosition();
      updateBrainRegionParticles();
      updateSelectedNodeGlow();
      updateConnectionSignals();
      updateEdgeJumpLabels();
      brain.canvas?.render();
      brain.animationId = requestAnimationFrame(renderLoop);
    };
    renderLoop();
    brain.canvas.trackball.enabled = false;
    brain.canvas.trackball.handleResize();
    brain.canvas.activated = true;
    setupBrainInteraction();
    drawBrainNetwork(createTemplateNodes(), defaultTemplateEdges);
    pushConsole('threeBrain N27 pial template initialized.');
  } catch (error) {
    pushConsole(String(error), 'error');
    toggleConsole(true);
  }
}

function resizeBrainScene() {
  if (!brain.canvas) return;
  brain.canvas.handle_resize(els.networkChart.clientWidth, els.networkChart.clientHeight, true, true);
  setBrainCamera();
  updateActiveTooltipPosition();
}

function setupBrainInteraction() {
  const dom = brain.canvas.main_canvas;
  const startDrag = (event) => {
    if (![0, 1, 2].includes(event.button)) return;
    cancelBrainFocusAnimation();
    const isPan = event.button === 1 || event.button === 2 || event.shiftKey;
    brain.drag = {
      x: event.clientX,
      y: event.clientY,
      moved: false,
      mode: isPan ? 'pan' : 'rotate',
      vector: isPan ? null : arcballVector(event)
    };
    event.preventDefault();
  };
  const moveDrag = (event) => {
    if (!brain.drag) return;
    const dx = event.clientX - brain.drag.x;
    const dy = event.clientY - brain.drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) brain.drag.moved = true;
    brain.drag.x = event.clientX;
    brain.drag.y = event.clientY;
    if (brain.drag.mode === 'pan') {
      const panScale = 0.72 / Math.max(0.35, brain.canvas.mainCamera.zoom);
      brain.canvas.origin.position.x += dx * panScale;
      brain.canvas.origin.position.y -= dy * panScale;
    } else {
      const currentVector = arcballVector(event);
      const rotation = new THREE.Quaternion().setFromUnitVectors(brain.drag.vector, currentVector);
      brain.canvas.origin.quaternion.premultiply(rotation);
      brain.drag.vector = currentVector;
    }
    brain.canvas.needsUpdate = true;
    event.preventDefault();
  };
  const endDrag = (event) => {
    const wasClick = brain.drag && !brain.drag.moved;
    brain.drag = null;
    if (wasClick) handleBrainClick(event);
  };
  dom.addEventListener('mousedown', startDrag);
  dom.addEventListener('contextmenu', (event) => event.preventDefault());
  window.addEventListener('mousemove', moveDrag);
  window.addEventListener('mouseup', endDrag);
  dom.addEventListener('wheel', (event) => {
    event.preventDefault();
    cancelBrainFocusAnimation();
    const camera = brain.canvas.mainCamera;
    camera.zoom = Math.max(0.45, Math.min(3.45, camera.zoom * (1 - event.deltaY * 0.001)));
    camera.updateProjectionMatrix();
    brain.canvas.needsUpdate = true;
  }, { passive: false });
}

function handleBrainClick(event) {
  if (!brain.canvas || !brain.nodeObjects.length) return;
  const rect = brain.canvas.main_renderer.domElement.getBoundingClientRect();
  brain.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  brain.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  brain.raycaster.setFromCamera(brain.pointer, brain.canvas.mainCamera);
  const hits = brain.raycaster.intersectObjects(brain.nodeObjects, false);
  if (hits.length) {
    const node = hits[0].object.userData.eegNode;
    if (brain.activeTooltipNode) setNodeLabelVisible(brain.activeTooltipNode, true);
    brain.activeTooltipNode = null;
    els.nodeTooltip.classList.add('hidden');
    clearEdgeJumpLabels();
    setSelectedNodeGlow(node.name);
    setBrainRegionHighlight(node);
    focusNode(node, () => showNodeTooltip(node, event));
  } else {
    if (brain.activeTooltipNode) setNodeLabelVisible(brain.activeTooltipNode, true);
    brain.activeTooltipNode = null;
    els.nodeTooltip.classList.add('hidden');
    clearBrainRegionHighlight();
    setSelectedNodeGlow(null);
    clearEdgeJumpLabels();
  }
}

function renderNetwork(payload) {
  const graph = payload.visualization.graph;
  const maxInfluence = Math.max(...graph.nodes.map((node) => Number(node.influence || 0)), 1e-9);
  const nodes = graph.nodes.map((node) => {
    const meta = classifyChannel(node.name);
    const pos = fixedNodePosition({ ...node, ...meta });
    const influenceRatio = Math.sqrt(Number(node.influence || 0) / maxInfluence);
    return {
      ...node,
      hemisphere: node.hemisphere || meta.hemisphere,
      region: node.region || meta.region,
      ...pos,
      size: 7 + 13 * influenceRatio,
      color: Number(node.contribution || node.value || 0) < 0 ? '#2563eb' : '#dc2626'
    };
  });
  const nodeById = new Map(nodes.map((node) => [node.name, node]));
  const edges = graph.edges.filter((edge) => nodeById.has(edgeEndpointName(edge.source)) && nodeById.has(edgeEndpointName(edge.target)));
  drawBrainNetwork(nodes, edges);
  pushConsole(`Network rendered with threeBrain: ${nodes.length} nodes, ${edges.length} links.`);
}

function edgeEndpointName(endpoint) {
  if (typeof endpoint === 'string') return endpoint;
  return endpoint?.name || endpoint?.id || '';
}

function showNodeTooltip(node, event) {
  const links = brain.edges
    .filter((edge) => edgeEndpointName(edge.source) === node.name || edgeEndpointName(edge.target) === node.name)
    .slice(0, 6);
  if (brain.activeTooltipNode && brain.activeTooltipNode !== node.name) {
    setNodeLabelVisible(brain.activeTooltipNode, true);
  }
  brain.activeTooltipNode = node.name;
  setNodeLabelVisible(node.name, false);
  setBrainRegionHighlight(node);
  setSelectedNodeGlow(node.name);
  createEdgeJumpLabels(node);
  els.nodeTooltip.innerHTML = `
    <div class="node-tooltip-head">
      <strong>${node.name}</strong>
    </div>
    <div class="node-tooltip-grid">
      <span>Hemisphere</span><b>${node.hemisphere}</b>
      <span>Region</span><b>${node.region}</b>
      <span>Influence</span><b>${fmt(node.influence, 4)}</b>
    </div>
    <div class="node-tooltip-title">Top Links</div>
    <div class="node-tooltip-links">
      ${links.length ? links.map((edge) => `<span>${edgeEndpointName(edge.source)} - ${edgeEndpointName(edge.target)}</span><b>${fmt(edge.value, 4)}</b>`).join('') : '<span>No top links</span><b>--</b>'}
    </div>
  `;
  els.nodeTooltip.classList.remove('hidden');
  updateActiveTooltipPosition();
}

function renderRisk(payload) {
  const prob = payload.prediction.probMdd;
  els.riskLabel.textContent = payload.prediction.risk.label;
  els.riskLabel.className = `risk-label ${payload.prediction.risk.code}`;
  els.probValue.textContent = `${(prob * 100).toFixed(1)}%`;
  els.probBar.style.width = `${Math.max(0, Math.min(100, prob * 100))}%`;
  els.probBar.style.background =
    payload.prediction.risk.code === 'normal' ? '#16803a' : payload.prediction.risk.code === 'mild' ? '#b86e00' : '#b42318';
}

function renderMeta(payload) {
  const { file, features } = payload;
  els.fileName.textContent = file.name;
  const sourcePath = file.path || selectedEdf || '未选择';
  els.fileSource.textContent = sourcePath;
  els.fileSource.title = sourcePath;
  const channelText = file.modelChannels && file.modelChannels !== file.nChannels ? `${file.nChannels} / ${file.modelChannels}` : file.nChannels;
  els.fileMeta.textContent = `${channelText} 通道 · ${fmt(file.durationSeconds, 1)} 秒 · ${fmt(file.sfreq, 1)} Hz · ${features.temporal.n_windows} 窗`;
  file.warnings.forEach((warning) => pushConsole(warning, 'warn'));
  pushConsole(payload.prediction.note);
}

function renderRegion(payload) {
  const rows = [...payload.visualization.regions].sort((a, b) => a.contribution - b.contribution);
  charts.region.setOption({
    grid: { left: 86, right: 28, top: 18, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: rows.map((row) => row.name) },
    series: [{
      type: 'bar',
      data: rows.map((row) => ({ value: row.contribution, itemStyle: { color: row.contribution >= 0 ? '#b86e00' : '#11756b' } })),
      label: { show: true, position: 'right', formatter: (item) => fmt(item.value, 3) }
    }]
  });
}

function renderAsymmetry(payload) {
  const rows = payload.visualization.asymmetry;
  charts.asym.setOption({
    grid: { left: 70, right: 20, top: 24, bottom: 42 },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, data: ['左侧', '右侧'] },
    xAxis: { type: 'category', data: rows.map((row) => row.region) },
    yAxis: { type: 'value' },
    series: [
      { name: '左侧', type: 'bar', data: rows.map((row) => row.left), itemStyle: { color: '#2563eb' } },
      { name: '右侧', type: 'bar', data: rows.map((row) => row.right), itemStyle: { color: '#dc2626' } }
    ]
  });
}

function renderTemporal(payload) {
  const rows = payload.visualization.temporalGroups;
  charts.temporal.setOption({
    grid: { left: 58, right: 24, top: 18, bottom: 36 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: rows.map((row) => row.name) },
    yAxis: { type: 'value' },
    series: [{
      type: 'bar',
      data: rows.map((row) => ({ value: row.contribution, itemStyle: { color: row.contribution >= 0 ? '#b86e00' : '#11756b' } })),
      label: { show: true, position: 'top', formatter: (item) => fmt(item.value, 3) }
    }]
  });
}

function renderEdges(payload) {
  const rows = payload.visualization.graph.edges.slice(0, 20);
  els.edgeRows.innerHTML = rows.map((edge) => `
    <tr><td>${edge.rank}</td><td>${edge.source} - ${edge.target}</td><td>${fmt(edge.value, 4)}</td><td>${fmt(edge.score, 4)}</td></tr>
  `).join('');
}

function render(payload) {
  renderMeta(payload);
  renderRisk(payload);
  renderNetwork(payload);
  renderRegion(payload);
  renderAsymmetry(payload);
  renderTemporal(payload);
  renderEdges(payload);
  resizeCharts();
}

function closeModelMenu() {
  els.modelMenu.classList.add('hidden');
}

function toggleModelMenu() {
  if (els.modelSelect.classList.contains('disabled')) return;
  els.modelMenu.classList.toggle('hidden');
}

async function importCustomModel() {
  const selected = await open({
    title: '选择兼容 AppModel pickle',
    multiple: false,
    filters: [{ name: 'Pickle model', extensions: ['pkl'] }]
  });
  if (selected) {
    customModel = selected;
    selectedModel = 'custom';
    pushConsole(`自定义模型已导入：${selected}`);
  } else {
    selectedModel = 'clean';
  }
  updateModelUi();
}

els.modelSelect.addEventListener('click', toggleModelMenu);
els.modelSelect.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    toggleModelMenu();
  }
  if (event.key === 'Escape') closeModelMenu();
});

els.modelMenu.addEventListener('click', async (event) => {
  event.stopPropagation();
  const button = event.target.closest('.model-menu-item');
  if (!button || button.disabled) return;
  closeModelMenu();
  const key = button.dataset.model;
  if (key === 'custom') {
    await importCustomModel();
    return;
  }
  selectedModel = key;
  updateModelUi();
});

document.addEventListener('click', (event) => {
  if (!els.modelSelect.contains(event.target)) closeModelMenu();
});

document.addEventListener('keydown', (event) => {
  if (event.code === 'Backquote') {
    event.preventDefault();
    toggleConsole();
  }
  if (event.key === 'Escape') {
    toggleConsole(false);
    if (brain.activeTooltipNode) setNodeLabelVisible(brain.activeTooltipNode, true);
    brain.activeTooltipNode = null;
    els.nodeTooltip.classList.add('hidden');
    clearBrainRegionHighlight();
    setSelectedNodeGlow(null);
    clearEdgeJumpLabels();
  }
});

document.querySelectorAll('.tab-button').forEach((button) => {
  button.addEventListener('click', () => {
    const tab = button.dataset.tab;
    document.querySelectorAll('.tab-button').forEach((item) => item.classList.toggle('active', item === button));
    document.querySelectorAll('.tab-page').forEach((page) => page.classList.toggle('active', page.id === `tab-${tab}`));
    resizeCharts();
  });
});

els.sidebarToggle.addEventListener('click', () => {
  const collapsed = els.appShell.classList.toggle('sidebar-collapsed');
  els.sidebarToggle.textContent = collapsed ? '›' : '‹';
  els.sidebarToggle.title = collapsed ? '展开侧边栏' : '收起侧边栏';
  resizeCharts();
});

els.consoleClose.addEventListener('click', () => toggleConsole(false));
els.consoleHead.addEventListener('pointerdown', (event) => {
  if (event.target === els.consoleClose) return;
  const rect = els.consolePanel.getBoundingClientRect();
  consoleDrag = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  els.consoleHead.setPointerCapture(event.pointerId);
});
els.consoleHead.addEventListener('pointermove', (event) => {
  if (!consoleDrag) return;
  const left = Math.max(8, Math.min(window.innerWidth - els.consolePanel.offsetWidth - 8, event.clientX - consoleDrag.x));
  const top = Math.max(8, Math.min(window.innerHeight - els.consolePanel.offsetHeight - 8, event.clientY - consoleDrag.y));
  els.consolePanel.style.left = `${left}px`;
  els.consolePanel.style.top = `${top}px`;
  els.consolePanel.style.right = 'auto';
  els.consolePanel.style.bottom = 'auto';
  els.consolePanel.style.transform = 'none';
});
els.consoleHead.addEventListener('pointerup', () => {
  consoleDrag = null;
});

els.pickFile.addEventListener('click', async () => {
  const selected = await open({
    title: '选择 EDF 脑电文件',
    multiple: false,
    filters: [{ name: 'EDF EEG', extensions: ['edf'] }]
  });
  if (!selected) return;

  selectedEdf = selected;
  els.fileName.textContent = selected.split(/[\\/]/).pop();
  els.fileSource.textContent = selected;
  els.fileSource.title = selected;
  els.fileMeta.textContent = '等待推理文件信息';
  els.riskLabel.textContent = '待推理';
  els.riskLabel.className = 'risk-label muted';
  els.probValue.textContent = '--';
  els.probBar.style.width = '0%';
  els.progressWrap.classList.add('hidden');
  pushConsole(`EDF 已导入：${selected}`);
});

els.runInference.addEventListener('click', async () => {
  if (!selectedEdf) {
    pushConsole('请先导入 EDF 文件。', 'error');
    toggleConsole(true);
    return;
  }
  if (selectedModel === 'custom' && !customModel) {
    pushConsole('请先导入兼容的 AppModel .pkl 文件。', 'error');
    toggleConsole(true);
    return;
  }

  pushConsole('正在读取 EDF、提取图特征并调用所选模型。首次运行某个模型配置会建立缓存。');
  setBusy(true);
  startProgress();

  try {
    const payload = await invoke('run_inference', {
      edfPath: selectedEdf,
      modelProfile: selectedModel === 'custom' ? 'clean' : selectedModel,
      customModelPath: selectedModel === 'custom' ? customModel : null
    });
    render(payload);
    finishProgress(true);
    setBusy(false);
    pushConsole('推理完成，结果已刷新。');
  } catch (error) {
    pushConsole(String(error), 'error');
    toggleConsole(true);
    els.riskLabel.textContent = '失败';
    els.riskLabel.className = 'risk-label severe';
    finishProgress(false);
    setBusy(false);
  }
});

window.addEventListener('resize', resizeCharts);

initBrainScene();
updateModelUi();
pushConsole('控制台初始化完成。按 ` 打开或关闭控制台。');
