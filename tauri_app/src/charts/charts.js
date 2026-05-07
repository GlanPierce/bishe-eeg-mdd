import * as echarts from 'echarts';

export function createCharts(els) {
  const init = (element) => echarts.init(element, null, {
    width: element.clientWidth || 420,
    height: element.clientHeight || 292
  });
  return {
    region: init(els.regionChart),
    asym: init(els.asymChart),
    temporal: init(els.temporalChart)
  };
}

export const chartPalette = {
  positive: '#ffffff',
  negative: '#59636f',
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
