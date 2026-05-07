export function installInteraction(BrainScene) {
  Object.assign(BrainScene.prototype, {
    arcballVector(event) {
      const rect = this.brain.canvas.main_canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      const lengthSq = x * x + y * y;
      if (lengthSq > 1) {
        return new this.THREE.Vector3(x, y, 0).normalize();
      }
      return new this.THREE.Vector3(x, y, Math.sqrt(1 - lengthSq)).normalize();
    },
    
    async initBrainScene() {
      this.startup.set(18);
      const tooltip = this.els.nodeTooltip;
      this.els.networkChart.innerHTML = '';
      this.els.networkChart.appendChild(this.els.resetView);
      if (this.els.networkLegend) {
        this.els.networkLegend.classList.remove('hidden', 'closing');
        this.els.networkChart.appendChild(this.els.networkLegend);
      }
      this.els.networkChart.appendChild(tooltip);
      this.brain.raycaster = new this.THREE.Raycaster();
      this.brain.pointer = new this.THREE.Vector2();
      this.brain.canvas = new this.ViewerCanvas(
        this.els.networkChart,
        this.els.networkChart.clientWidth || 960,
        this.els.networkChart.clientHeight || 640,
        0,
        false,
        false,
        true
      );
      this.brain.canvas.set_state('target_subject', 'EEG_TEMPLATE');
      this.brain.canvas.set_state('surface_type', 'pial');
      this.brain.canvas.set_state('material_type_left', 'normal');
      this.brain.canvas.set_state('material_type_right', 'normal');
      this.brain.canvas.set_state('surface_opacity_left', 0.48);
      this.brain.canvas.set_state('surface_opacity_right', 0.48);
      this.brain.canvas.init_subject('EEG_TEMPLATE');
      this.brain.canvas.shared_data.set('EEG_TEMPLATE', {
        matrices: {
          tkrRAS_Scanner: new this.THREE.Matrix4().identity(),
          tkrRAS_MNI305: new this.THREE.Matrix4().identity()
        }
      });
      if (this.brain.canvas.compass?.container) this.brain.canvas.compass.container.visible = false;
      this.brain.canvas.setBackground({ color: '#000000' });
      try {
        this.startup.set(42);
        await this.addBrainSurface();
        this.startup.set(72);
        this.setBrainCamera();
        const renderLoop = () => {
          this.updateBrainFocusAnimation();
          this.updateDragInertia();
          this.updateActiveTooltipPosition();
          this.updateBrainRegionParticles();
          this.updateSelectedNodeGlow();
          this.updateConnectionSignals();
          this.updateEdgeJumpLabels();
          this.brain.canvas?.render();
          this.brain.animationId = requestAnimationFrame(renderLoop);
        };
        renderLoop();
        this.brain.canvas.trackball.enabled = false;
        this.brain.canvas.trackball.handleResize();
        this.brain.canvas.activated = true;
        this.setupBrainInteraction();
        this.drawBrainNetwork(this.createTemplateNodes(), this.defaultTemplateEdges);
        requestAnimationFrame(() => this.resizeBrainScene());
        setTimeout(() => this.resizeBrainScene(), 120);
        this.startup.set(94);
        this.logger.log('threeBrain N27 pial template initialized.');
      } catch (error) {
        this.logger.log(String(error), 'error');
        this.logger.toggle(true);
      }
    },
    
    resizeBrainScene() {
      if (!this.brain.canvas) return;
      const rect = this.els.networkChart.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width || window.innerWidth));
      const height = Math.max(1, Math.round(rect.height || window.innerHeight));
      this.brain.canvas.handle_resize(width, height, true, true);
      this.brain.canvas.main_renderer?.setSize?.(width, height, false);
      this.setBrainCamera();
      this.updateActiveTooltipPosition();
    },
    
    setupBrainInteraction() {
      const dom = this.brain.canvas.main_canvas;
      const rendererDom = this.brain.canvas.main_renderer?.domElement;
      const startDrag = (event) => {
        if (![0, 1, 2].includes(event.button)) return;
        this.cancelBrainFocusAnimation();
        this.brain.inertia = null;
        const isPan = event.button === 1 || event.button === 2 || event.shiftKey;
        this.brain.drag = {
          x: event.clientX,
          y: event.clientY,
          dx: 0,
          dy: 0,
          button: event.button,
          moved: false,
          mode: isPan ? 'pan' : 'rotate',
          vector: isPan ? null : this.arcballVector(event)
        };
        event.preventDefault();
      };
      const moveDrag = (event) => {
        if (!this.brain.drag) return;
        const dx = event.clientX - this.brain.drag.x;
        const dy = event.clientY - this.brain.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) this.brain.drag.moved = true;
        this.brain.drag.x = event.clientX;
        this.brain.drag.y = event.clientY;
        this.brain.drag.dx = dx;
        this.brain.drag.dy = dy;
        if (this.brain.drag.mode === 'pan') {
          this.applyPanDelta(dx, dy);
        } else {
          const currentVector = this.arcballVector(event);
          const rotation = new this.THREE.Quaternion().setFromUnitVectors(this.brain.drag.vector, currentVector);
          this.brain.canvas.origin.quaternion.premultiply(rotation);
          this.brain.drag.vector = currentVector;
        }
        this.brain.canvas.needsUpdate = true;
        event.preventDefault();
      };
      const endDrag = (event) => {
        const drag = this.brain.drag;
        const wasClick = drag && !drag.moved;
        this.brain.drag = null;
        if (!drag) return;
        if (wasClick && drag.button === 1) {
          this.resetBrainView();
          return;
        }
        if (wasClick) {
          this.handleBrainClick(event);
          return;
        }
        this.startDragInertia(drag);
      };
      [dom, rendererDom].filter(Boolean).forEach((target) => {
        target.addEventListener('mousedown', startDrag);
        target.addEventListener('contextmenu', (event) => event.preventDefault());
      });
      window.addEventListener('mousemove', moveDrag);
      window.addEventListener('mouseup', endDrag);
      [dom, rendererDom].filter(Boolean).forEach((target) => target.addEventListener('wheel', (event) => {
        event.preventDefault();
        this.cancelBrainFocusAnimation();
        const camera = this.brain.canvas.mainCamera;
        camera.zoom = Math.max(0.45, Math.min(3.45, camera.zoom * (1 - event.deltaY * 0.001)));
        camera.updateProjectionMatrix();
        this.brain.canvas.needsUpdate = true;
      }, { passive: false }));
    },

    applyPanDelta(dx, dy) {
      const panScale = 0.34 / Math.max(0.35, this.brain.canvas.mainCamera.zoom);
      this.brain.canvas.origin.position.x += dx * panScale;
      this.brain.canvas.origin.position.y -= dy * panScale;
      this.brain.canvas.origin.position.x = this.THREE.MathUtils.clamp(this.brain.canvas.origin.position.x, -150, 150);
      this.brain.canvas.origin.position.y = this.THREE.MathUtils.clamp(this.brain.canvas.origin.position.y, -120, 120);
    },

    applyRotateDelta(dx, dy) {
      const rect = this.brain.canvas.main_canvas.getBoundingClientRect();
      const angle = Math.hypot(dx, dy) / Math.max(260, Math.min(rect.width, rect.height)) * 1.28;
      if (angle < 0.0001) return;
      const axis = new this.THREE.Vector3(dy, dx, 0).normalize();
      this.brain.canvas.origin.quaternion.premultiply(new this.THREE.Quaternion().setFromAxisAngle(axis, angle));
    },

    startDragInertia(drag) {
      const speed = Math.hypot(drag.dx || 0, drag.dy || 0);
      if (speed < 2.5) return;
      this.brain.inertia = {
        mode: drag.mode,
        dx: this.THREE.MathUtils.clamp(drag.dx, -38, 38),
        dy: this.THREE.MathUtils.clamp(drag.dy, -38, 38),
        startedAt: performance.now()
      };
    },

    updateDragInertia() {
      const inertia = this.brain.inertia;
      if (!inertia || !this.brain.canvas || this.brain.drag || this.brain.focusAnimation) return;
      if (inertia.mode === 'pan') this.applyPanDelta(inertia.dx, inertia.dy);
      else this.applyRotateDelta(inertia.dx, inertia.dy);
      inertia.dx *= 0.91;
      inertia.dy *= 0.91;
      const slow = Math.hypot(inertia.dx, inertia.dy) < 0.14;
      if (slow || performance.now() - inertia.startedAt > 1250) this.brain.inertia = null;
      this.brain.canvas.needsUpdate = true;
    },
    
    handleBrainClick(event) {
      if (!this.brain.canvas || !this.brain.nodeObjects.length) return;
      const rect = this.brain.canvas.main_renderer.domElement.getBoundingClientRect();
      this.brain.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.brain.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this.brain.raycaster.setFromCamera(this.brain.pointer, this.brain.canvas.mainCamera);
      const hits = this.brain.raycaster.intersectObjects(this.brain.nodeObjects, false);
      const node = hits[0]?.object.userData.eegNode || this.findNearestScreenNode(event);
      if (node) {
        this.hideNodeTooltip();
        this.setSelectedNodeGlow(node.name);
        this.setBrainRegionHighlight(node);
        this.focusNode(node, () => this.showNodeTooltip(node, event));
      } else {
        this.hideNodeTooltip();
        this.clearBrainRegionHighlight();
        this.setSelectedNodeGlow(null);
      }
    },

    findNearestScreenNode(event) {
      const rect = this.brain.canvas.main_renderer.domElement.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      this.brain.canvas.origin.updateMatrixWorld(true);
      this.brain.canvas.mainCamera.updateMatrixWorld(true);
      let nearest = null;
      let nearestDistance = Infinity;
      this.brain.nodeObjects.forEach((object) => {
        const node = object.userData.eegNode;
        if (!node || object.visible === false) return;
        const worldPosition = new this.THREE.Vector3();
        object.getWorldPosition(worldPosition);
        const screenPosition = worldPosition.project(this.brain.canvas.mainCamera);
        if (screenPosition.z < -1 || screenPosition.z > 1) return;
        const x = rect.left + (screenPosition.x * 0.5 + 0.5) * rect.width;
        const y = rect.top + (-screenPosition.y * 0.5 + 0.5) * rect.height;
        const distance = Math.hypot(event.clientX - x, event.clientY - y);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = node;
        }
      });
      return nearestDistance <= 28 ? nearest : null;
    },
    
    renderNetwork(payload) {
      const graph = payload.visualization.graph;
      const maxInfluence = Math.max(...graph.nodes.map((node) => Number(node.influence || 0)), 1e-9);
      const nodes = graph.nodes.map((node) => {
        const meta = this.classifyChannel(node.name);
        const pos = this.fixedNodePosition({ ...node, ...meta });
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
      const edges = graph.edges.filter((edge) => nodeById.has(this.edgeEndpointName(edge.source)) && nodeById.has(this.edgeEndpointName(edge.target)));
      this.drawBrainNetwork(nodes, edges);
      this.logger.log(`Network rendered with threeBrain: ${nodes.length} nodes, ${edges.length} links.`);
    },
    
    edgeEndpointName(endpoint) {
      if (typeof endpoint === 'string') return endpoint;
      return endpoint?.name || endpoint?.id || '';
    },
    
    showNodeTooltip(node, event) {
      const links = this.brain.edges
        .filter((edge) => this.edgeEndpointName(edge.source) === node.name || this.edgeEndpointName(edge.target) === node.name)
        .slice(0, 6);
      if (this.brain.activeTooltipNode && this.brain.activeTooltipNode !== node.name) {
        this.setNodeLabelVisible(this.brain.activeTooltipNode, true);
      }
      this.brain.activeTooltipNode = node.name;
      this.setNodeLabelVisible(node.name, false);
      this.setBrainRegionHighlight(node);
      this.setSelectedNodeGlow(node.name);
      this.createEdgeJumpLabels(node);
      this.els.nodeTooltip.innerHTML = `
        <div class="node-tooltip-head">
          <strong>${node.name}</strong>
        </div>
        <div class="node-tooltip-grid">
          <span>Hemisphere</span><b>${node.hemisphere}</b>
          <span>Region</span><b>${node.region}</b>
          <span>Influence</span><b>${this.fmt(node.influence, 4)}</b>
        </div>
        <div class="node-tooltip-title">Top Links</div>
        <div class="node-tooltip-links">
          ${links.length ? links.map((edge) => `<span>${this.edgeEndpointName(edge.source)} - ${this.edgeEndpointName(edge.target)}</span><b>${Number(edge.value) >= 0 ? '+' : ''}${this.fmt(edge.value, 4)}</b>`).join('') : '<span>No top links</span><b>--</b>'}
        </div>
      `;
      this.els.nodeTooltip.classList.remove('hidden');
      this.updateActiveTooltipPosition();
    }
  });
}
