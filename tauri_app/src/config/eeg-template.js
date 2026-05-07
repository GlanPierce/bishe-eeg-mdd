export const modelOptions = [
  { key: 'clean', label: 'ExplicitRegionTemporalWeightedStarGNN' },
  { key: 'targeted_repair', label: 'ExplicitRegionTemporalWeightedStarGNN + targeted artifact repair' },
  {
    key: 'taskonly61_multistate_arch',
    label: 'Multistate explicit region-temporal weighted-star model',
    disabled: true,
    consoleOnly: true,
    badge: '测试用 / Console only'
  },
  { key: 'gcn_unavailable', label: 'DynamicsGraphConvGNN GCN', disabled: true },
  { key: 'custom', label: '+ 导入模型' }
];

export const electrodeCoords = {
  Fp1: [-0.42, 0.72, -1.02], Fp2: [0.42, 0.72, -1.02],
  F7: [-1.05, 0.28, -0.72], F3: [-0.48, 0.82, -0.62], Fz: [0, 0.9, -0.62], F4: [0.48, 0.82, -0.62], F8: [1.05, 0.28, -0.72],
  T3: [-1.16, 0.05, -0.04], C3: [-0.58, 0.92, -0.02], Cz: [0, 1.02, -0.02], C4: [0.58, 0.92, -0.02], T4: [1.16, 0.05, -0.04],
  T5: [-0.98, 0.2, 0.62], P3: [-0.46, 0.82, 0.58], Pz: [0, 0.9, 0.58], P4: [0.46, 0.82, 0.58], T6: [0.98, 0.2, 0.62],
  O1: [-0.36, 0.56, 1.08], O2: [0.36, 0.56, 1.08],
  A1: [-1.22, -0.08, 0.02], A2: [1.22, -0.08, 0.02],
  '23A': [-0.18, 0.78, 1.18], '24A': [0.18, 0.78, 1.18]
};

export const defaultTemplateEdges = [
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

export function shortChannel(name) {
  return String(name).split('-')[0];
}

export function classifyChannel(name) {
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

export function scalpZ(x, y) {
  const xr = x / 128;
  const yr = y / 142;
  const dome = Math.max(0, 1 - xr * xr - yr * yr);
  return 16 + 84 * Math.sqrt(dome);
}

export function fixedNodePosition(node) {
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

export function createTemplateNodes() {
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
