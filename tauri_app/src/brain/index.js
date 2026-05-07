import * as THREE from '@rave-ieeg/three-brain/node_modules/three/build/three.module.js';
import { ViewerCanvas } from '@rave-ieeg/three-brain/src/js/core/ViewerCanvas.js';
import lhPialUrl from '../assets/brain/N27/surf/lh.pial?url';
import rhPialUrl from '../assets/brain/N27/surf/rh.pial?url';
import {
  classifyChannel,
  createTemplateNodes,
  defaultTemplateEdges,
  fixedNodePosition,
  shortChannel
} from '../config/eeg-template.js';
import { installSurfaceBase } from './surfaces-base.js';
import { installSurfaceEffects } from './surfaces-effects.js';
import { installObjects } from './objects.js';
import { installFocusCamera } from './focus-camera.js';
import { installFocusSelection } from './focus-selection.js';
import { installInteraction } from './interaction.js';
import { fmt } from '../utils/format.js';

export class BrainScene {
  constructor({ els, logger, startup }) {
    this.THREE = THREE;
    this.ViewerCanvas = ViewerCanvas;
    this.lhPialUrl = lhPialUrl;
    this.rhPialUrl = rhPialUrl;
    this.els = els;
    this.logger = logger;
    this.startup = startup;
    this.shortChannel = shortChannel;
    this.classifyChannel = classifyChannel;
    this.fixedNodePosition = fixedNodePosition;
    this.createTemplateNodes = createTemplateNodes;
    this.defaultTemplateEdges = defaultTemplateEdges;
    this.fmt = fmt;
    this.brain = {
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
      inertia: null,
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
  }
}

installSurfaceBase(BrainScene);
installSurfaceEffects(BrainScene);
installObjects(BrainScene);
installFocusCamera(BrainScene);
installFocusSelection(BrainScene);
installInteraction(BrainScene);

Object.assign(BrainScene.prototype, {
  readFsInt24(view, offset) {
    return (view.getUint8(offset) << 16) | (view.getUint8(offset + 1) << 8) | view.getUint8(offset + 2);
  },

  skipFsLine(view, offset) {
    let cursor = offset;
    while (cursor < view.byteLength && view.getUint8(cursor) !== 10) cursor += 1;
    return cursor + 1;
  },

  async loadFreeSurferSurface(url) {
    const buffer = await fetch(url).then((res) => {
      if (!res.ok) throw new Error(`Failed to load brain surface: ${res.status}`);
      return res.arrayBuffer();
    });
    const view = new DataView(buffer);
    let offset = 0;
    const magic = this.readFsInt24(view, offset);
    offset += 3;
    if (magic !== 16777214) throw new Error(`Unsupported FreeSurfer surface magic: ${magic}`);
    offset = this.skipFsLine(view, offset);
    offset = this.skipFsLine(view, offset);
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
});
