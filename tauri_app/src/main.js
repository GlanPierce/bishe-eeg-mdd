import './styles.css';
import { mountShell, queryElements } from './ui/shell.js';
import { createStartupLoader } from './ui/startup.js';
import { createCharts, baseChartStyle, chartAxis, chartPalette, valueColor } from './charts/charts.js';
import { createConsole } from './console/console.js';
import { BrainScene } from './brain/index.js';
import { fmt, signedFmt } from './utils/format.js';
import { invoke, open } from './services/inference.js';

const app = document.getElementById('app');
let modelOptions = [{ key: 'custom', label: '+ 导入模型' }];
mountShell(app, modelOptions);

const els = queryElements();
const startup = createStartupLoader({
  loader: document.getElementById('startupLoader'),
  percent: document.getElementById('startupPercent'),
  fill: document.getElementById('startupFill')
});
const charts = createCharts(els);
const appConsole = createConsole(els);
const brainScene = new BrainScene({ els, logger: appConsole, startup });

const tabOrder = ['history', 'network', 'contrib', 'edges'];
const validTabs = [...tabOrder];
const multistateModelKey = 'taskonly61_multistate_arch';
const stateOrder = ['TASK', 'EC', 'EO'];
const historyStorageKey = 'eeg-mdd-inference-history-v1';
const maxHistoryItems = 20;
let activeTab = 'network';
let customModel = '';
let selectedEdf = '';
let selectedEdfStates = {};
let selectedModel = 'clean';
let progressTimer = null;
let lastNonHistoryTab = 'network';

function normalizeModelEntry(item) {
  return {
    key: item.key,
    label: item.label,
    folder: item.folder,
    path: item.path,
    consoleOnly: Boolean(item.consoleOnly ?? item.console_only),
    disabled: Boolean(item.disabled),
    badge: item.badge
  };
}

function customModelOption() {
  return { key: 'custom', label: '+ 导入模型' };
}

function renderModelMenu() {
  const items = modelOptions
    .filter((item) => !item.consoleOnly)
    .map((item) => `
      <button class="model-menu-item ${item.disabled ? 'disabled' : ''}" data-model="${item.key}" ${item.disabled ? 'disabled' : ''}>
        ${item.key === 'custom' ? '+ 导入模型' : item.label}
        ${item.badge ? `<small>${item.badge}</small>` : ''}
      </button>
    `)
    .join('');
  els.modelMenu.innerHTML = `
    <div class="model-menu-title">模型选择</div>
    ${items}
    <button class="model-menu-item model-folder-item" data-action="open-model-folder" type="button">打开模型文件夹</button>
  `;
}

async function refreshModelOptions() {
  try {
    const entries = await invoke('list_model_folder');
    const diskModels = entries.map(normalizeModelEntry);
    modelOptions = [...diskModels, customModelOption()];
    if (!modelOptions.some((item) => item.key === selectedModel)) {
      selectedModel = modelOptions.find((item) => !item.consoleOnly && item.key !== 'custom')?.key || 'custom';
    }
    renderModelMenu();
    updateModelUi();
  } catch (error) {
    appConsole.log(`模型目录读取失败：${error}`, 'error');
    modelOptions = [customModelOption()];
    selectedModel = 'custom';
    renderModelMenu();
    updateModelUi();
  }
}

function isMultistateSelected() {
  return selectedModel === multistateModelKey;
}

function classifyEdfState(path, fallbackIndex = 0) {
  const name = String(path).split(/[\\/]/).pop().toUpperCase();
  if (/(^|[_\-\s])TASK([_\-\s.]|$)/.test(name)) return 'TASK';
  if (/(^|[_\-\s])EC([_\-\s.]|$)|EYES?CLOSED/.test(name)) return 'EC';
  if (/(^|[_\-\s])EO([_\-\s.]|$)|EYES?OPEN/.test(name)) return 'EO';
  return stateOrder[fallbackIndex] || null;
}

function describeSelectedEdfs() {
  if (!isMultistateSelected()) return selectedEdf || '';
  return stateOrder
    .filter((state) => selectedEdfStates[state])
    .map((state) => `${state}=${selectedEdfStates[state]}`)
    .join(' | ');
}

function selectedEdfFileName(path) {
  return String(path || '').split(/[\\/]/).pop();
}

function fileNameWithoutExtension(nameOrPath) {
  const name = selectedEdfFileName(nameOrPath);
  return name.replace(/\.[^.\\/]+$/, '');
}

function updateOverflowTooltips(root = document) {
  const targets = root.querySelectorAll([
    '#fileName',
    '#fileSource',
    '.edf-state-button strong',
    '.history-card-meta b',
    '.history-card-meta small',
    '.history-card-meta em',
    '.history-card-meta i'
  ].join(','));
  targets.forEach((element) => {
    element.classList.remove('overflow-scroll');
    element.style.removeProperty('--scroll-distance');
    const text = element.getAttribute('title') || element.dataset.tooltip || element.textContent || '';
    element.dataset.tooltip = text;
    element.removeAttribute('title');
    element.classList.toggle('has-text-tooltip', Boolean(text));
  });
}

function showTextTooltip(element, event) {
  const text = element?.dataset?.tooltip;
  if (!text || !els.textTooltip) return;
  els.textTooltip.textContent = text;
  els.textTooltip.classList.remove('hidden');
  moveTextTooltip(event);
}

function moveTextTooltip(event) {
  if (!els.textTooltip || els.textTooltip.classList.contains('hidden')) return;
  const margin = 14;
  const rect = els.textTooltip.getBoundingClientRect();
  const left = Math.max(8, Math.min(window.innerWidth - rect.width - 8, event.clientX + margin));
  const top = Math.max(8, Math.min(window.innerHeight - rect.height - 8, event.clientY + margin));
  els.textTooltip.style.left = `${left}px`;
  els.textTooltip.style.top = `${top}px`;
}

function hideTextTooltip() {
  els.textTooltip?.classList.add('hidden');
}

function showNetworkLegend() {
  els.networkLegend.classList.remove('hidden', 'closing');
}

function hideNetworkLegend() {
  els.networkLegend.classList.add('closing');
  window.setTimeout(() => {
    if (els.networkLegend.classList.contains('closing')) {
      els.networkLegend.classList.add('hidden');
      els.networkLegend.classList.remove('closing');
    }
  }, 240);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[char]);
}

function readHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(historyStorageKey) || '[]');
    return Array.isArray(parsed) ? parsed.filter((item) => item && item.payload) : [];
  } catch (error) {
    appConsole.log(`History read failed: ${error}`, 'warn');
    return [];
  }
}

function writeHistory(items) {
  let next = items.slice(0, maxHistoryItems);
  while (next.length >= 0) {
    try {
      localStorage.setItem(historyStorageKey, JSON.stringify(next));
      return true;
    } catch (error) {
      if (next.length === 0) {
        appConsole.log(`History save failed: ${error}`, 'warn');
        return false;
      }
      next = next.slice(0, -1);
    }
  }
  return false;
}

function historyKeyForCurrentSelection(payload = null) {
  const source = isMultistateSelected()
    ? stateOrder.map((state) => `${state}:${selectedEdfStates[state] || ''}`).join('|')
    : selectedEdf || payload?.file?.path || payload?.file?.name || '';
  const modelPath = selectedModel === 'custom' ? customModel : '';
  return `${selectedModel}|${modelPath}|${source}`;
}

function historyTitle(entry) {
  return fileNameWithoutExtension(entry.fileName || entry.payload?.file?.name || 'EDF result');
}

function captureDefaultCanvasThumbnail() {
  const canvas = brainScene.brain.canvas;
  const rendererCanvas = canvas?.main_renderer?.domElement;
  if (!canvas || !rendererCanvas) return '';
  const origin = canvas.origin;
  const camera = canvas.mainCamera;
  const previous = {
    quaternion: origin.quaternion.clone(),
    position: origin.position.clone(),
    zoom: camera.zoom,
    focusAnimation: brainScene.brain.focusAnimation
  };
  try {
    brainScene.brain.focusAnimation = null;
    origin.quaternion.identity();
    origin.position.set(0, 0, 0);
    brainScene.setBrainCamera();
    canvas.render();
    const target = document.createElement('canvas');
    target.width = 432;
    target.height = 304;
    const ctx = target.getContext('2d');
    const sourceWidth = rendererCanvas.width || rendererCanvas.clientWidth;
    const sourceHeight = rendererCanvas.height || rendererCanvas.clientHeight;
    if (!ctx || !sourceWidth || !sourceHeight) return '';
    const sourceRatio = sourceWidth / sourceHeight;
    const targetRatio = target.width / target.height;
    const cropWidth = sourceRatio > targetRatio ? sourceHeight * targetRatio : sourceWidth;
    const cropHeight = sourceRatio > targetRatio ? sourceHeight : sourceWidth / targetRatio;
    const cropX = (sourceWidth - cropWidth) / 2;
    const cropY = (sourceHeight - cropHeight) / 2;
    ctx.drawImage(rendererCanvas, cropX, cropY, cropWidth, cropHeight, 0, 0, target.width, target.height);
    return target.toDataURL('image/jpeg', 0.72);
  } catch (error) {
    appConsole.log(`Thumbnail capture failed: ${error}`, 'warn');
    return '';
  } finally {
    origin.quaternion.copy(previous.quaternion);
    origin.position.copy(previous.position);
    camera.zoom = previous.zoom;
    camera.updateProjectionMatrix();
    brainScene.brain.focusAnimation = previous.focusAnimation;
    canvas.needsUpdate = true;
    canvas.render();
  }
}

function renderHistoryPage() {
  const items = readHistory();
  if (!items.length) {
    els.historyList.innerHTML = '<div class="history-empty">暂无已保存的推理结果</div>';
    return;
  }
  els.historyList.innerHTML = items.map((entry) => `
    <button class="history-card" type="button" data-history-id="${escapeHtml(entry.id)}">
      <div class="history-card-meta">
        <strong style="color:${riskGradientColor(entry.payload?.prediction?.probMdd)}">${escapeHtml(entry.payload?.prediction?.risk?.label || '--')}</strong>
        <span>${Number.isFinite(Number(entry.payload?.prediction?.probMdd)) ? `${(Number(entry.payload.prediction.probMdd) * 100).toFixed(1)}%` : '--'}</span>
        <i title="${escapeHtml(entry.modelLabel || entry.modelKey || '--')}">${escapeHtml(entry.modelLabel || entry.modelKey || '--')}</i>
        <b title="${escapeHtml(historyTitle(entry))}">${escapeHtml(historyTitle(entry))}</b>
        <small title="${escapeHtml(entry.source || '')}">${escapeHtml(entry.source || '--')}</small>
        <em>${escapeHtml(new Date(entry.savedAt).toLocaleString())}</em>
      </div>
      <div class="history-card-preview">${entry.thumbnail ? `<img src="${escapeHtml(entry.thumbnail)}" alt="" />` : '<span>暂无缩略图</span>'}</div>
    </button>
  `).join('');
  requestAnimationFrame(() => updateOverflowTooltips(els.historyList));
}

function saveHistory(payload) {
  const option = modelOptions.find((item) => item.key === selectedModel) || modelOptions[0];
  const source = describeSelectedEdfs() || payload?.file?.path || '';
  const key = historyKeyForCurrentSelection(payload);
  const entry = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    key,
    savedAt: new Date().toISOString(),
    source,
    fileName: fileNameWithoutExtension(payload?.file?.name || selectedEdf || source),
    modelKey: selectedModel,
    modelLabel: option.label,
    customModelPath: selectedModel === 'custom' ? customModel : '',
    edfPath: selectedEdf,
    edfStates: { ...selectedEdfStates },
    thumbnail: captureDefaultCanvasThumbnail(),
    payload
  };
  const items = readHistory().filter((item) => item.key !== key);
  const saved = writeHistory([entry, ...items]);
  if (saved) renderHistoryPage();
}

function restoreHistory(id) {
  const entry = readHistory().find((item) => item.id === id);
  if (!entry) {
    appConsole.log('History item not found.', 'error');
    return;
  }
  selectedModel = entry.modelKey || 'clean';
  customModel = entry.customModelPath || '';
  selectedEdf = entry.edfPath || entry.payload?.file?.path || '';
  selectedEdfStates = { ...(entry.edfStates || {}) };
  updateModelUi();
  render(entry.payload);
  appConsole.log(`History restored: ${historyTitle(entry)}`);
}

function resetInferenceDisplay() {
  els.fileMeta.textContent = '';
  els.riskLabel.textContent = '未推理';
  els.riskLabel.className = 'risk-label muted';
  els.riskLabel.style.color = '';
  els.probValue.textContent = '--';
  els.probBar.style.width = '0%';
  els.progressWrap.classList.add('hidden');
}

function updateEdfSlotUi() {
  els.edfStateSlots.classList.toggle('hidden', !isMultistateSelected());
  els.pickFile.classList.toggle('hidden', isMultistateSelected());
  els.edfStateSlots.querySelectorAll('.edf-state-button').forEach((button) => {
    const state = button.dataset.state;
    const path = selectedEdfStates[state];
    const label = path ? fileNameWithoutExtension(path) : '未导入';
    button.classList.toggle('filled', Boolean(path));
    button.querySelector('strong').textContent = label;
    button.querySelector('strong').title = path || label;
    button.title = path || `导入 ${state} EDF`;
  });
  updateOverflowTooltips(els.edfStateSlots);
}

function updateFileSummaryAfterSelection() {
  if (isMultistateSelected()) {
    const filled = stateOrder.filter((state) => selectedEdfStates[state]);
    selectedEdf = selectedEdfStates.TASK || filled.map((state) => selectedEdfStates[state])[0] || '';
    els.fileName.textContent = filled.length ? `已导入 ${filled.length}/3：${filled.join(' + ')}` : '待导入 TASK / EC / EO';
    els.fileName.title = describeSelectedEdfs() || els.fileName.textContent;
    els.fileSource.textContent = describeSelectedEdfs();
    els.fileSource.title = describeSelectedEdfs();
  } else {
    els.fileName.textContent = selectedEdf ? fileNameWithoutExtension(selectedEdf) : '';
    els.fileName.title = selectedEdf || els.fileName.textContent;
    els.fileSource.textContent = selectedEdf;
    els.fileSource.title = selectedEdf;
  }
  updateEdfSlotUi();
  requestAnimationFrame(() => updateOverflowTooltips(els.fileName.parentElement));
}

function riskGradientColor(value) {
  const clamp = Math.max(0, Math.min(1, Number(value) || 0));
  const stops = clamp < 0.5
    ? ['#2fbf71', '#f7f7f2', clamp / 0.5]
    : ['#f7f7f2', '#d93d30', (clamp - 0.5) / 0.5];
  const parse = (hex) => hex.match(/\w\w/g).map((part) => parseInt(part, 16));
  const [ar, ag, ab] = parse(stops[0]);
  const [br, bg, bb] = parse(stops[1]);
  const mix = (a, b) => Math.round(a + (b - a) * stops[2]).toString(16).padStart(2, '0');
  return `#${mix(ar, br)}${mix(ag, bg)}${mix(ab, bb)}`;
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
  setTimeout(() => {
    els.progressBar.classList.remove('failed');
    if (ok) els.progressWrap.classList.add('hidden');
  }, ok ? 900 : 1600);
}

function setBusy(isBusy) {
  els.pickFile.disabled = isBusy;
  els.historyButton.disabled = isBusy;
  els.edfStateSlots.querySelectorAll('button').forEach((button) => {
    button.disabled = isBusy;
  });
  els.runInference.disabled = isBusy;
  els.modelSelect.classList.toggle('disabled', isBusy);
}

function updateModelUi() {
  const option = modelOptions.find((item) => item.key === selectedModel) || modelOptions[0] || customModelOption();
  els.modelSelectedText.textContent = option.key === 'custom' ? '' : option.label;
  els.modelSelectedText.classList.toggle('empty', option.key === 'custom');
  els.customModelPath.textContent = option.key === 'custom' ? customModel : '';
  els.pickFile.textContent = isMultistateSelected() ? '导入 3 个 EDF' : '导入 EDF';
  updateFileSummaryAfterSelection();
  requestAnimationFrame(() => {
    const parent = els.modelSelectedText.parentElement;
    const distance = parent ? Math.max(0, els.modelSelectedText.scrollWidth - parent.clientWidth) : 0;
    els.modelSelectedText.style.setProperty('--scroll-distance', `${distance}px`);
    els.modelSelectedText.classList.toggle('scrolling', distance > 4);
    updateOverflowTooltips(els.appShell);
  });
}

function resizeCharts() {
  requestAnimationFrame(() => {
    Object.values(charts).forEach((chart) => chart.resize());
    brainScene.resizeBrainScene();
  });
}

function defaultSelectableModelKey() {
  return modelOptions.find((item) => !item.consoleOnly && item.key !== 'custom')?.key || 'custom';
}

function resetAppToInitialState() {
  window.clearInterval(progressTimer);
  progressTimer = null;
  customModel = '';
  selectedEdf = '';
  selectedEdfStates = {};
  selectedModel = defaultSelectableModelKey();
  lastNonHistoryTab = 'network';
  closeModelMenu();
  setBusy(false);
  resetInferenceDisplay();
  updateModelUi();
  els.fileName.textContent = '';
  els.fileName.title = '';
  els.fileSource.textContent = '';
  els.fileSource.title = '';
  els.fileMeta.textContent = '';
  els.edgeRows.innerHTML = '<tr><td colspan="4">暂无数据</td></tr>';
  Object.values(charts).forEach((chart) => chart.clear());
  showNetworkLegend();
  if (activeTab !== 'network') switchTab('network');
  if (brainScene.brain.canvas) {
    brainScene.hideNodeTooltip();
    brainScene.clearBrainRegionHighlight();
    brainScene.setSelectedNodeGlow(null);
    brainScene.clearEdgeJumpLabels();
    brainScene.drawBrainNetwork(brainScene.createTemplateNodes(), brainScene.defaultTemplateEdges);
    brainScene.resetBrainView();
  }
  appConsole.log('App reset to initial state.');
}

function renderRisk(payload) {
  const prob = payload.prediction.probMdd;
  els.riskLabel.textContent = payload.prediction.risk.label;
  els.riskLabel.className = `risk-label ${payload.prediction.risk.code}`;
  els.riskLabel.style.color = riskGradientColor(prob);
  els.probValue.textContent = `${(prob * 100).toFixed(1)}%`;
  els.probBar.style.width = `${Math.max(0, Math.min(100, prob * 100))}%`;
  els.probBar.style.background = 'linear-gradient(90deg, #2fbf71 0%, #f7f7f2 50%, #d93d30 100%)';
}

function renderMeta(payload) {
  const { file, features } = payload;
  const sourcePath = describeSelectedEdfs() || file.path || '';
  els.fileName.textContent = fileNameWithoutExtension(file.name);
  els.fileName.title = sourcePath || file.name;
  els.fileSource.textContent = sourcePath;
  els.fileSource.title = sourcePath;
  const channelText = file.modelChannels && file.modelChannels !== file.nChannels ? `${file.nChannels} / ${file.modelChannels}` : file.nChannels;
  els.fileMeta.innerHTML = `
    <span>${channelText} 通道</span>
    <span>${fmt(file.durationSeconds, 1)} 秒</span>
    <span>${fmt(file.sfreq, 1)} Hz</span>
    <span>${features.temporal.n_windows} 窗</span>
  `;
  file.warnings.forEach((warning) => appConsole.log(warning, 'warn'));
  appConsole.log(payload.prediction.note);
  requestAnimationFrame(() => updateOverflowTooltips(els.fileName.parentElement));
}

function renderEdges(payload) {
  const rows = payload.visualization.graph.edges.slice(0, 20);
  els.edgeRows.innerHTML = rows.map((edge) => `
    <tr><td>${edge.rank}</td><td>${edge.source} - ${edge.target}</td><td>${signedFmt(edge.value, 4)}</td><td>${fmt(edge.score, 4)}</td></tr>
  `).join('');
}

function renderUnifiedRegion(payload) {
  const rows = [...payload.visualization.regions].sort((a, b) => a.contribution - b.contribution);
  charts.region.setOption({
    ...baseChartStyle,
    grid: { left: 86, right: 28, top: 18, bottom: 28 },
    xAxis: chartAxis('value'),
    yAxis: chartAxis('category', rows.map((row) => row.name)),
    series: [{
      type: 'bar',
      barWidth: 14,
      data: rows.map((row) => ({
        value: row.contribution,
        itemStyle: { color: valueColor(row.contribution), borderRadius: row.contribution >= 0 ? [0, 5, 5, 0] : [5, 0, 0, 5] }
      })),
      label: { show: true, position: 'right', color: '#f4f7f8', formatter: (item) => fmt(item.value, 3) }
    }]
  });
}

function renderUnifiedTemporal(payload) {
  const rows = payload.visualization.temporalGroups;
  const xAxis = chartAxis('category', rows.map((row) => row.name));
  charts.temporal.setOption({
    ...baseChartStyle,
    grid: { left: 58, right: 24, top: 18, bottom: 36 },
    xAxis: { ...xAxis, axisLabel: { ...xAxis.axisLabel, interval: 0 } },
    yAxis: chartAxis('value'),
    series: [{
      type: 'bar',
      barWidth: 18,
      data: rows.map((row) => ({
        value: row.contribution,
        itemStyle: { color: valueColor(row.contribution), borderRadius: row.contribution >= 0 ? [5, 5, 0, 0] : [0, 0, 5, 5] }
      })),
      label: { show: true, position: 'top', color: '#f4f7f8', formatter: (item) => fmt(item.value, 3) }
    }]
  });
}

function renderUnifiedAsymmetryFixed(payload) {
  const rows = payload.visualization.asymmetry;
  charts.asym.setOption({
    ...baseChartStyle,
    color: [chartPalette.left, chartPalette.right],
    grid: { left: 70, right: 20, top: 24, bottom: 42 },
    legend: {
      bottom: 0,
      data: [
        { name: '左侧', icon: 'rect', itemStyle: { color: chartPalette.left, borderColor: chartPalette.left } },
        { name: '右侧', icon: 'rect', itemStyle: { color: chartPalette.right, borderColor: chartPalette.right } }
      ],
      textStyle: { color: chartPalette.text },
      inactiveColor: '#4c545b',
      inactiveBorderColor: '#4c545b'
    },
    xAxis: chartAxis('category', rows.map((row) => row.region)),
    yAxis: chartAxis('value'),
    series: [
      { name: '左侧', type: 'bar', barWidth: 13, data: rows.map((row) => ({ value: row.left, itemStyle: { color: chartPalette.left, borderRadius: row.left >= 0 ? [5, 5, 0, 0] : [0, 0, 5, 5] } })) },
      { name: '右侧', type: 'bar', barWidth: 13, data: rows.map((row) => ({ value: row.right, itemStyle: { color: chartPalette.right, borderRadius: row.right >= 0 ? [5, 5, 0, 0] : [0, 0, 5, 5] } })) }
    ]
  });
}

function render(payload) {
  renderMeta(payload);
  renderRisk(payload);
  brainScene.renderNetwork(payload);
  renderUnifiedRegion(payload);
  renderUnifiedAsymmetryFixed(payload);
  renderUnifiedTemporal(payload);
  renderEdges(payload);
  resizeCharts();
}

function closeModelMenu() {
  els.modelMenu.classList.add('hidden');
}

async function toggleModelMenu() {
  if (els.modelSelect.classList.contains('disabled')) return;
  await refreshModelOptions();
  els.modelMenu.classList.toggle('hidden');
}

function switchTab(tab) {
  if (!validTabs.includes(tab) || tab === activeTab) return;
  const previousTab = activeTab;
  const previousIsDataView = previousTab !== 'network';
  const nextIsDataView = tab !== 'network';
  const previousPage = document.getElementById(`tab-${previousTab}`);
  if (tab === 'history') renderHistoryPage();
  if (previousTab !== 'history') lastNonHistoryTab = previousTab;
  if (previousTab !== 'network') {
    previousPage?.classList.add('leaving');
    previousPage?.classList.toggle('data-crossfade', previousIsDataView && nextIsDataView);
    window.setTimeout(() => {
      previousPage?.classList.remove('leaving');
      previousPage?.classList.remove('data-crossfade');
      if (activeTab !== previousTab) previousPage?.classList.remove('active');
    }, 320);
  }
  activeTab = tab;
  if (tab !== 'history') lastNonHistoryTab = tab;
  els.appShell.classList.toggle('data-view-active', nextIsDataView);
  document.querySelectorAll('.tab-button').forEach((item) => item.classList.toggle('active', item.dataset.tab === tab));
  document.querySelectorAll('.tab-page').forEach((page) => {
    if (page.id !== `tab-${previousTab}` || previousTab === 'network') page.classList.toggle('active', page.id === `tab-${tab}`);
  });
  if (tab !== 'network' || previousTab === 'network') previousPage?.classList.remove('active');
  resizeCharts();
}

function stepTab(direction) {
  const index = Math.max(0, tabOrder.indexOf(activeTab));
  switchTab(tabOrder[(index + direction + tabOrder.length) % tabOrder.length]);
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
    appConsole.log(`自定义模型已导入：${selected}`);
  } else {
    selectedModel = 'clean';
  }
  updateModelUi();
}

async function pickEdf() {
  if (isMultistateSelected()) {
    const firstMissing = stateOrder.find((state) => !selectedEdfStates[state]) || 'TASK';
    await pickEdfState(firstMissing);
    return;
  }
  const selected = await open({
    title: '选择 EDF 脑电文件',
    multiple: false,
    filters: [{ name: 'EDF EEG', extensions: ['edf'] }]
  });
  if (!selected) return;
  selectedEdf = Array.isArray(selected) ? selected[0] : selected;
  selectedEdfStates = {};
  updateFileSummaryAfterSelection();
  resetInferenceDisplay();
  appConsole.log(`EDF 已导入：${describeSelectedEdfs()}`);
}

async function pickEdfState(state) {
  const selected = await open({
    title: `选择 ${state} EDF 脑电文件`,
    multiple: false,
    filters: [{ name: 'EDF EEG', extensions: ['edf'] }]
  });
  if (!selected) return;
  selectedEdfStates = { ...selectedEdfStates, [state]: Array.isArray(selected) ? selected[0] : selected };
  updateFileSummaryAfterSelection();
  resetInferenceDisplay();
  appConsole.log(`${state} EDF 已导入：${selectedEdfStates[state]}`);
}

async function runInference() {
  const hasRequiredEdf = isMultistateSelected()
    ? stateOrder.every((state) => selectedEdfStates[state])
    : Boolean(selectedEdf);
  if (!hasRequiredEdf) {
    appConsole.log(isMultistateSelected() ? '请先导入 TASK、EC、EO 三个 EDF 文件。' : '请先导入 EDF 文件。', 'error');
    appConsole.toggle(true);
    return;
  }
  if (selectedModel === 'custom' && !customModel) {
    appConsole.log('请先导入兼容的 AppModel .pkl 文件。', 'error');
    appConsole.toggle(true);
    return;
  }
  appConsole.log('正在读取 EDF、提取图特征并调用所选模型。');
  setBusy(true);
  startProgress();
  try {
    const payload = await invoke('run_inference', {
      edfPath: selectedEdf,
      modelProfile: selectedModel === 'custom' ? 'clean' : selectedModel,
      customModelPath: selectedModel === 'custom' ? customModel : null,
      edfStates: isMultistateSelected() ? selectedEdfStates : null
    });
    render(payload);
    saveHistory(payload);
    finishProgress(true);
    appConsole.log('推理完成，结果已刷新。');
  } catch (error) {
    appConsole.log(String(error), 'error');
    appConsole.toggle(true);
    els.riskLabel.textContent = '失败';
    els.riskLabel.className = 'risk-label severe';
    els.riskLabel.style.color = riskGradientColor(1);
    finishProgress(false);
  } finally {
    setBusy(false);
  }
}

async function openModelsFolder() {
  try {
    const path = await invoke('open_model_folder');
    appConsole.log(`模型文件夹已打开：${path}`);
    await refreshModelOptions();
  } catch (error) {
    appConsole.log(String(error), 'error');
    appConsole.toggle(true);
  }
}

function registerConsoleCommands() {
  const modelKeys = () => modelOptions
    .filter((item) => !item.disabled || item.consoleOnly)
    .map((item) => item.key);
  const completeFrom = (values) => ({ token }) => values.filter((item) => item.startsWith(token));
  const setLegendVisible = (mode = 'toggle') => {
    const show = mode === 'show'
      ? true
      : mode === 'hide'
        ? false
        : els.networkLegend.classList.contains('hidden');
    if (show) showNetworkLegend();
    else hideNetworkLegend();
    appConsole.log(`Network legend ${show ? 'shown' : 'hidden'}.`);
  };

  appConsole.register('help', () => appConsole.log([...appConsole.commands.keys()].sort().join('\n')));
  appConsole.register('status', () => {
    appConsole.log(`EDF: ${describeSelectedEdfs() || 'none'}\nModel: ${selectedModel}\nTab: ${activeTab}\nNodes: ${brainScene.brain.nodes.length}\nEdges: ${brainScene.brain.edges.length}`);
  });
  appConsole.register('pick-edf', pickEdf);
  appConsole.register('run', runInference);
  appConsole.register('reset-view', resetAppToInitialState);
  appConsole.register(
    'console',
    ([mode]) => appConsole.toggle(mode === 'show' ? true : mode === 'hide' ? false : undefined),
    { complete: completeFrom(['show', 'hide']) }
  );
  appConsole.register(
    'tab',
    ([tab]) => switchTab(tab),
    { complete: completeFrom(tabOrder) }
  );
  appConsole.register('models-folder', openModelsFolder);
  appConsole.register(
    'legend',
    ([mode]) => setLegendVisible(mode),
    { complete: completeFrom(['show', 'hide', 'toggle']) }
  );
  appConsole.register(
    'text-block',
    ([mode]) => setLegendVisible(mode),
    { complete: completeFrom(['show', 'hide', 'toggle']) }
  );
  appConsole.register('model', async ([action, value]) => {
    if (action === 'list' || !action) {
      appConsole.log(modelOptions.map((item) => {
        const flags = [
          item.disabled && !item.consoleOnly ? 'disabled' : '',
          item.consoleOnly ? '测试用/console-only' : ''
        ].filter(Boolean);
        return `${item.key}${flags.length ? ` (${flags.join(', ')})` : ''}: ${item.label}`;
      }).join('\n'));
      return;
    }
    if (action === 'use') {
      const option = modelOptions.find((item) => item.key === value);
      if (!option || (option.disabled && !option.consoleOnly)) throw new Error(`Model unavailable: ${value}`);
      selectedModel = option.key;
      updateModelUi();
      if (option.consoleOnly) appConsole.log(`${option.label} 已通过 Console 开启。该模型标记为测试用。`, 'warn');
      return;
    }
    if (action === 'custom') await importCustomModel();
  }, {
    complete: ({ argsBefore }) => {
      if (argsBefore.length === 0) return ['list', 'use', 'custom'];
      if (argsBefore.length === 1 && argsBefore[0] === 'use') return modelKeys();
      return [];
    }
  });
}

function bindEvents() {
  els.modelSelect.addEventListener('click', () => toggleModelMenu());
  els.modelSelect.addEventListener('keydown', async (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      await toggleModelMenu();
    }
    if (event.key === 'Escape') closeModelMenu();
  });
  els.modelMenu.addEventListener('click', async (event) => {
    event.stopPropagation();
    const button = event.target.closest('.model-menu-item');
    if (!button || button.disabled) return;
    closeModelMenu();
    if (button.dataset.action === 'open-model-folder') await openModelsFolder();
    else if (button.dataset.model === 'custom') await importCustomModel();
    else {
      selectedModel = button.dataset.model;
      updateModelUi();
    }
  });
  document.addEventListener('click', (event) => {
    if (!els.modelSelect.contains(event.target)) closeModelMenu();
  });
  document.addEventListener('mouseover', (event) => {
    const target = event.target.closest('.has-text-tooltip');
    if (target) showTextTooltip(target, event);
  });
  document.addEventListener('mousemove', moveTextTooltip);
  document.addEventListener('mouseout', (event) => {
    if (event.target.closest('.has-text-tooltip')) hideTextTooltip();
  });
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Backquote') {
      event.preventDefault();
      appConsole.toggle();
    }
    if (event.key === 'Escape') {
      appConsole.toggle(false);
      brainScene.hideNodeTooltip();
      brainScene.clearBrainRegionHighlight();
      brainScene.setSelectedNodeGlow(null);
    }
    if (event.key === 'ArrowLeft') stepTab(-1);
    if (event.key === 'ArrowRight') stepTab(1);
  });
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => {
      const tab = button.dataset.tab;
      const nextTab = tab === 'history' && activeTab === 'history'
        ? lastNonHistoryTab
        : (tab === 'contrib' || tab === 'edges') && activeTab === tab
          ? 'network'
          : tab;
      switchTab(nextTab);
    });
  });
  els.prevTab.addEventListener('click', () => stepTab(-1));
  els.nextTab.addEventListener('click', () => stepTab(1));
  els.resetView.addEventListener('click', (event) => {
    event.stopPropagation();
    resetAppToInitialState();
  });
  els.networkLegend.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    hideNetworkLegend();
  });
  els.consoleClose.addEventListener('click', () => appConsole.toggle(false));
  els.pickFile.addEventListener('click', pickEdf);
  els.historyList.addEventListener('click', (event) => {
    const button = event.target.closest('.history-card');
    if (button) restoreHistory(button.dataset.historyId);
  });
  els.edfStateSlots.querySelectorAll('.edf-state-button').forEach((button) => {
    button.addEventListener('click', () => pickEdfState(button.dataset.state));
  });
  els.runInference.addEventListener('click', runInference);
  window.addEventListener('resize', resizeCharts);
}

registerConsoleCommands();
bindEvents();
refreshModelOptions();
brainScene.initBrainScene().finally(() => startup.hide());
updateModelUi();
appConsole.log('Console ready. Press ` to toggle.');
