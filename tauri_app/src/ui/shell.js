export function mountShell(app, modelOptions) {
  const items = modelOptions.filter((item) => !item.consoleOnly).map((item) => `
    <button class="model-menu-item ${item.disabled ? 'disabled' : ''} ${item.consoleOnly ? 'console-only' : ''}" data-model="${item.key}" ${item.disabled ? 'disabled' : ''}>
      ${item.key === 'custom' ? '+ 导入模型' : item.label}
      ${item.badge ? `<small>${item.badge}</small>` : ''}
    </button>
  `).join('');
  app.innerHTML = `
  <div id="appShell" class="app-shell">
    <div class="top-overlay">
      <section class="top-import glass-strip">
        <span class="label">数据导入</span>
        <button id="pickFile" class="primary-button">导入 EDF</button>
        <div id="edfStateSlots" class="edf-state-slots hidden">
          <button class="edf-state-button" data-state="TASK" type="button"><span>TASK</span><strong>未导入</strong></button>
          <button class="edf-state-button" data-state="EC" type="button"><span>EC</span><strong>未导入</strong></button>
          <button class="edf-state-button" data-state="EO" type="button"><span>EO</span><strong>未导入</strong></button>
        </div>
        <strong id="fileName"></strong>
        <small id="fileSource" class="path-text"></small>
        <small id="fileMeta"></small>
      </section>
      <nav class="tabs view-tabs" aria-label="结果视图">
        <button id="prevTab" class="tab-arrow" type="button" title="上一页">&lt;</button>
        <div class="tab-track">
          <button id="historyButton" class="tab-button" data-tab="history">历史记录</button><span class="tab-separator">丨</span>
          <button class="tab-button active" data-tab="network">脑网络连接图</button><span class="tab-separator">丨</span>
          <button class="tab-button" data-tab="signals">信号特征</button><span class="tab-separator">丨</span>
          <button class="tab-button" data-tab="contrib">贡献数据</button><span class="tab-separator">丨</span>
          <button class="tab-button" data-tab="edges">连接强度</button>
        </div>
        <button id="nextTab" class="tab-arrow" type="button" title="下一页">&gt;</button>
      </nav>
      <section class="top-infer glass-strip">
        <div class="top-risk">
          <span class="label">风险等级</span><div id="riskLabel" class="risk-label muted">未推理</div>
          <div class="prob-line"><span>MDD 概率</span><strong id="probValue">--</strong></div>
          <div class="meter"><div id="probBar" class="meter-fill"></div></div>
        </div>
        <div class="model-action-row">
          <div id="modelSelect" class="model-select" role="button" tabindex="0">
            <div class="model-selected-viewport"><div id="modelSelectedText" class="model-selected-text"></div></div>
            <span class="model-chevron">⌄</span>
            <div id="modelMenu" class="model-menu hidden">
              <div class="model-menu-title">模型选择</div>
              ${items}
              <button class="model-menu-item model-folder-item" data-action="open-model-folder" type="button">打开模型文件夹</button>
            </div>
          </div>
          <button id="runInference" class="icon-button primary-icon" title="开始推理">推理</button>
        </div>
        <div id="progressWrap" class="inference-progress hidden"><div class="progress-meta"><span id="progressText">准备推理</span><strong id="progressValue">0%</strong></div><div class="progress-track"><div id="progressBar" class="progress-fill"></div></div></div>
        <small id="customModelPath" class="path-text"></small>
      </section>
    </div>
    <main class="workspace">
      <section id="tab-network" class="tab-page active"><section class="panel graph-panel"><div id="networkChart" class="brain-scene"><button id="resetView" class="reset-view-button" type="button" title="重置视角">重置</button><div id="networkLegend" class="network-legend"><b>图例</b><span><i class="legend-node red"></i>节点红色：通道特征偏向推高 MDD</span><span><i class="legend-node green"></i>节点绿色：通道特征偏向正常</span><span><i class="legend-edge red"></i>边红色：连接贡献偏向推高 MDD</span><span><i class="legend-edge green"></i>边绿色：连接贡献偏向正常</span><small>边粗细/亮度表示综合强度；连接值正负仅表示 PCC 相关方向。右键关闭图例。</small></div><div id="nodeTooltip" class="node-tooltip hidden"></div></div></section></section>
      <section id="tab-signals" class="tab-page"><div class="content-page"><div class="signal-grid">
        <section class="panel signal-panel-wide"><div class="panel-head"><h3>EEG时域波形</h3><span>推理输入的前8秒多通道波形，单位μV</span></div><div id="waveformChart" class="chart signal-chart"></div></section>
        <section class="panel"><div class="panel-head"><h3>功率谱密度</h3><span>Welch估计，0.5-45 Hz</span></div><div id="psdChart" class="chart signal-chart"></div></section>
        <section class="panel"><div class="panel-head"><h3>PCC热力图</h3><span>通道间皮尔逊相关系数</span></div><div id="pccHeatmapChart" class="chart signal-chart"></div></section>
      </div></div></section>
      <section id="tab-contrib" class="tab-page"><div class="content-page"><div class="contrib-grid">
        <section class="panel"><div class="panel-head"><h3>脑区贡献</h3><span>各脑区特征对模型输出的方向和强度</span></div><div id="regionChart" class="chart"></div></section>
        <section class="panel"><div class="panel-head"><h3>左右半球</h3><span>同一脑区内左右侧通道影响度对比</span></div><div id="asymChart" class="chart"></div></section>
        <section class="panel"><div class="panel-head"><h3>时间贡献</h3><span>时间窗口统计量对模型输出的影响</span></div><div id="temporalChart" class="chart"></div></section>
        <section class="panel explain-panel"><div class="panel-head"><h3>指标说明</h3><span>贡献值不是临床诊断，只解释当前模型如何得到输出</span></div><div class="explain-list"><p><b>脑区贡献</b>：按脑区汇总标准化特征值与模型权重的乘积。正值推向 MDD，负值推向正常；绝对值越大，影响越强。</p><p><b>左右半球</b>：同一脑区内分别累加左侧、右侧成对电极的通道影响度，只表示模型依赖程度，不表示病灶位置。</p><p><b>时间贡献</b>：来自滑动时间窗的均值、标准差和变化量等摘要，方向表示推向 MDD 或正常，大小表示被模型使用的程度。</p></div></section>
      </div></div></section>
      <section id="tab-edges" class="tab-page"><div class="content-page"><section class="table-panel"><div class="panel-head"><h3>连接强度</h3><span>按连接本身和相关通道影响度综合排序</span></div><div class="table-explain"><p><b>连接值</b>：当前主图使用 PCC 相关连接；正负号只表示两个通道的相关方向，绝对值越大表示同步变化越强。</p><p><b>主图颜色</b>：边颜色按模型贡献方向显示，红色偏向推高 MDD，绿色偏向推向正常；它和 PCC 正负号不是同一个含义。</p><p><b>综合分</b>：排序用分数，约等于连接强度的绝对值乘以两端通道影响度的加权项，用来把连接强且相关通道重要的边排到前面。</p></div><table><thead><tr><th>#</th><th>连接</th><th>连接值</th><th>综合分</th></tr></thead><tbody id="edgeRows"><tr><td colspan="4">暂无数据</td></tr></tbody></table></section></div></section>
      <section id="tab-history" class="tab-page history-tab"><div class="content-page history-page"><section class="history-panel"><div class="panel-head"><h3>历史记录</h3><span>选择已保存的推理结果，直接恢复视图</span></div><div id="historyList" class="history-card-grid"></div></section></div></section>
    </main>
  </div>
  <section id="consolePanel" class="console-panel console-overlay hidden"><div id="consoleHead" class="console-head"><span>Console</span><button id="consoleClose" class="console-close" title="关闭">×</button></div><div id="consoleOutput" class="console-output"></div><form id="consoleForm" class="console-form"><span>&gt;</span><input id="consoleInput" autocomplete="off" spellcheck="false" /></form></section>`;
  app.insertAdjacentHTML('beforeend', '<div id="textTooltip" class="text-tooltip hidden"></div>');
}

export function queryElements() {
  const $ = (id) => document.getElementById(id);
  const els = {
    appShell: $('appShell'), prevTab: $('prevTab'), nextTab: $('nextTab'), modelSelect: $('modelSelect'), modelSelectedText: $('modelSelectedText'), modelMenu: $('modelMenu'), customModelPath: $('customModelPath'), pickFile: $('pickFile'), historyButton: $('historyButton'), historyList: $('historyList'), edfStateSlots: $('edfStateSlots'), runInference: $('runInference'), progressWrap: $('progressWrap'), progressText: $('progressText'), progressValue: $('progressValue'), progressBar: $('progressBar'), fileName: $('fileName'), fileSource: $('fileSource'), fileMeta: $('fileMeta'), riskLabel: $('riskLabel'), probValue: $('probValue'), probBar: $('probBar'), consolePanel: $('consolePanel'), consoleHead: $('consoleHead'), consoleClose: $('consoleClose'), consoleOutput: $('consoleOutput'), consoleForm: $('consoleForm'), consoleInput: $('consoleInput'), textTooltip: $('textTooltip'), edgeRows: $('edgeRows'), networkChart: $('networkChart'), resetView: $('resetView'), networkLegend: $('networkLegend'), nodeTooltip: $('nodeTooltip'), waveformChart: $('waveformChart'), psdChart: $('psdChart'), pccHeatmapChart: $('pccHeatmapChart'), regionChart: $('regionChart'), asymChart: $('asymChart'), temporalChart: $('temporalChart')
  };
  const dataSummary = document.createElement('div');
  dataSummary.className = 'data-summary';
  els.fileSource.parentElement.insertBefore(dataSummary, els.fileSource);
  dataSummary.append(els.fileSource, els.fileMeta);
  return els;
}
