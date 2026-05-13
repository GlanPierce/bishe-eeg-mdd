import * as echarts from 'echarts';

export function createCharts(els) {
  const init = (element) => echarts.init(element, null, {
    width: element.clientWidth || 420,
    height: element.clientHeight || 292
  });
  return {
    waveform: init(els.waveformChart),
    psd: init(els.psdChart),
    pccHeatmap: init(els.pccHeatmapChart),
    region: init(els.regionChart),
    asym: init(els.asymChart),
    temporal: init(els.temporalChart)
  };
}

export const chartPalette = {
  positive: '#d8948d',
  negative: '#93c69b',
  neutral: '#f7f7f2',
  left: '#8f9aa6',
  right: '#ffffff',
  grid: 'rgba(255, 255, 255, 0.08)',
  axis: 'rgba(255, 255, 255, 0.36)',
  text: 'rgba(244, 247, 248, 0.78)'
};

export const baseChartStyle = {
  backgroundColor: 'transparent',
  textStyle: { color: chartPalette.text, fontFamily: '"Microsoft YaHei", "Segoe UI", Arial, sans-serif' },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(8, 11, 12, 0.94)',
    borderColor: 'rgba(255, 255, 255, 0.16)',
    textStyle: { color: '#f4f7f8' },
    axisPointer: { lineStyle: { color: 'rgba(255, 255, 255, 0.26)' } }
  }
};

export function valueColor(value) {
  return Number(value) >= 0 ? chartPalette.positive : chartPalette.negative;
}

export function mixHexColor(base, target, ratio) {
  const parse = (hex) => hex.match(/\w\w/g).map((part) => parseInt(part, 16));
  const [ar, ag, ab] = parse(base);
  const [br, bg, bb] = parse(target);
  const clamped = Math.max(0, Math.min(1, Number(ratio) || 0));
  const mix = (a, b) => Math.round(a + (b - a) * clamped).toString(16).padStart(2, '0');
  return `#${mix(ar, br)}${mix(ag, bg)}${mix(ab, bb)}`;
}

export function contributionColor(value, maxAbs = 1) {
  const contribution = Number(value) || 0;
  if (!contribution) return chartPalette.neutral;
  const ratio = Math.pow(Math.min(1, Math.abs(contribution) / Math.max(1e-9, Number(maxAbs) || 0)), 0.72);
  return contribution >= 0
    ? mixHexColor(chartPalette.neutral, chartPalette.positive, ratio)
    : mixHexColor(chartPalette.neutral, chartPalette.negative, ratio);
}

export function chartAxis(type, data = null) {
  return {
    type,
    data,
    axisLine: { lineStyle: { color: chartPalette.axis } },
    axisTick: { lineStyle: { color: chartPalette.axis } },
    axisLabel: { color: chartPalette.text },
    splitLine: { lineStyle: { color: chartPalette.grid } }
  };
}
