export function installFocusSelection(BrainScene) {
  Object.assign(BrainScene.prototype, {
    setSelectedNodeGlow(name) {
      this.brain.selectedGlowNode = name;
      this.clearSelectedGlowParticles();
      this.brain.nodeGlowObjects.forEach((glow) => {
        const selected = glow.userData.eegNodeName === name;
        glow.material.opacity = selected ? 0.46 : 0.24;
        glow.scale.setScalar(selected ? 9.2 : 6.9);
        glow.renderOrder = selected ? 19 : 11;
      });
      if (name) this.createSelectedGlowParticles(name);
      this.applyFocusFade(name);
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    focusFadeForNode(name) {
      if (!this.brain.focusedNodeName) return 1;
      if (name === this.brain.focusedNodeName) return 1;
      const focused = this.findRenderedNode(this.brain.focusedNodeName);
      const node = this.findRenderedNode(name);
      if (!focused || !node) return 1;
      const distance = new this.THREE.Vector3(focused.x, focused.y, focused.z).distanceTo(new this.THREE.Vector3(node.x, node.y, node.z));
      const proximity = this.THREE.MathUtils.clamp(1 - distance / 105, 0, 1);
      return this.THREE.MathUtils.lerp(0.012, 0.95, proximity ** 1.85);
    },
    
    focusFadeForEdge(edge) {
      if (!this.brain.focusedNodeName) return 1;
      const sourceFade = this.focusFadeForNode(edge.userData.sourceName);
      const targetFade = this.focusFadeForNode(edge.userData.targetName);
      return this.THREE.MathUtils.clamp(Math.max(sourceFade, targetFade) * 0.9, 0.025, 0.9);
    },
    
    focusCenterVector() {
      const focused = this.brain.focusedNodeName ? this.findRenderedNode(this.brain.focusedNodeName) : null;
      return focused ? new this.THREE.Vector3(focused.x, focused.y, focused.z) : new this.THREE.Vector3();
    },
    
    hideNodeTooltip() {
      if (this.brain.activeTooltipNode) this.setNodeLabelVisible(this.brain.activeTooltipNode, true);
      this.brain.activeTooltipNode = null;
      this.els.nodeTooltip.classList.add('hidden');
      this.clearEdgeJumpLabels();
    },
    
    applyFocusFade(name) {
      this.brain.focusedNodeName = name;
      const focusCenter = this.focusCenterVector();
      const focusActive = this.brain.focusedNodeName ? 1 : 0;
      this.brain.nodeObjects.forEach((object) => {
        const nodeName = object.userData.eegNode?.name;
        object.material.opacity = nodeName === this.brain.focusedNodeName ? 1 : (object.userData.baseOpacity ?? 0.96) * this.focusFadeForNode(nodeName);
      });
      this.brain.labels.forEach((label) => {
        label.material.opacity = label.userData.eegNodeName === this.brain.focusedNodeName ? 1 : (label.userData.baseOpacity ?? 1) * this.focusFadeForNode(label.userData.eegNodeName);
      });
      this.brain.nodeGlowObjects.forEach((glow) => {
        if (glow.userData.eegNodeName === this.brain.selectedGlowNode) {
          glow.material.opacity = 0.5;
          glow.scale.setScalar(9.6);
          glow.renderOrder = 19;
        } else {
          glow.material.opacity = (glow.userData.baseOpacity ?? 0.24) * this.focusFadeForNode(glow.userData.eegNodeName);
          glow.scale.setScalar(6.9);
          glow.renderOrder = 11;
        }
      });
      this.brain.edgeObjects.forEach((edge) => {
        if (edge.material?.uniforms?.focusCenter) edge.material.uniforms.focusCenter.value.copy(focusCenter);
        if (edge.material?.uniforms?.focusActive) edge.material.uniforms.focusActive.value = focusActive;
        if (edge.material?.uniforms?.baseOpacity) edge.material.uniforms.baseOpacity.value = edge.userData.baseOpacity ?? 0.42;
      });
      this.brain.signalObjects.forEach((signal) => {
        if (signal.material?.uniforms?.focusCenter) signal.material.uniforms.focusCenter.value.copy(focusCenter);
        if (signal.material?.uniforms?.focusActive) signal.material.uniforms.focusActive.value = focusActive;
        signal.userData.focusFade = 1;
      });
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    clearSelectedGlowParticles() {
      if (!this.brain.selectedGlowParticles) return;
      this.brain.selectedGlowParticles.geometry?.dispose?.();
      this.brain.selectedGlowParticles.material?.dispose?.();
      this.brain.selectedGlowParticles.removeFromParent();
      this.brain.selectedGlowParticles = null;
    },
    
    createSelectedGlowParticles(name) {
      const node = this.findRenderedNode(name);
      if (!node || !this.brain.canvas) return;
      const count = 56;
      const positions = new Float32Array(count * 3);
      const dirs = new Float32Array(count * 3);
      const seeds = new Float32Array(count);
      const normal = node.normal ? new this.THREE.Vector3(node.normal.x, node.normal.y, node.normal.z).normalize() : new this.THREE.Vector3(0, 0, 1);
      for (let i = 0; i < count; i += 1) {
        const angle = Math.random() * Math.PI * 2;
        const tangent = new this.THREE.Vector3(Math.cos(angle), Math.sin(angle), 0).normalize();
        const dir = normal.clone().multiplyScalar(0.78 + Math.random() * 0.35).addScaledVector(tangent, (Math.random() - 0.5) * 0.55).normalize();
        positions[i * 3] = node.x;
        positions[i * 3 + 1] = node.y;
        positions[i * 3 + 2] = node.z;
        dirs[i * 3] = dir.x;
        dirs[i * 3 + 1] = dir.y;
        dirs[i * 3 + 2] = dir.z;
        seeds[i] = Math.random();
      }
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('direction', new this.THREE.BufferAttribute(dirs, 3));
      geometry.setAttribute('seed', new this.THREE.BufferAttribute(seeds, 1));
      const material = new this.THREE.ShaderMaterial({
        uniforms: { time: { value: 0 }, color: { value: new this.THREE.Color('#ffffff') } },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: this.THREE.AdditiveBlending,
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
      const particles = new this.THREE.Points(geometry, material);
      particles.renderOrder = 23;
      particles.userData.nodeName = name;
      particles.userData.startedAt = performance.now();
      this.brain.canvas.add_to_scene(particles);
      this.brain.selectedGlowParticles = particles;
    },
    
    updateSelectedNodeGlow() {
      if (!this.brain.selectedGlowNode) return;
      const now = performance.now();
      const pulse = 0.5 + 0.5 * Math.sin(now * 0.0048);
      this.brain.nodeGlowObjects.forEach((glow) => {
        if (glow.userData.eegNodeName !== this.brain.selectedGlowNode) return;
        glow.material.opacity = 0.46 + pulse * 0.2;
        glow.scale.setScalar(9.2 + pulse * 1.8);
      });
      if (this.brain.selectedGlowParticles?.material?.uniforms?.time) {
        this.brain.selectedGlowParticles.material.uniforms.time.value = (now - this.brain.selectedGlowParticles.userData.startedAt) / 1000;
      }
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    clearEdgeJumpLabels() {
      this.brain.edgeJumpLabels.forEach((label) => label.remove());
      this.brain.edgeJumpLabels = [];
    },
    
    createEdgeJumpLabels(node) {
      this.clearEdgeJumpLabels();
      if (!node) return;
      const links = this.brain.edges
        .filter((edge) => this.edgeEndpointName(edge.source) === node.name || this.edgeEndpointName(edge.target) === node.name)
        .slice(0, 10);
      links.forEach((edge) => {
        const sourceName = this.edgeEndpointName(edge.source);
        const targetName = this.edgeEndpointName(edge.target);
        const otherName = sourceName === node.name ? targetName : sourceName;
        const otherNode = this.findRenderedNode(otherName);
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
          const target = this.findRenderedNode(otherName);
          if (!target) return;
          this.focusNode(target, () => this.showNodeTooltip(target, event));
        });
        this.els.networkChart.appendChild(label);
        this.brain.edgeJumpLabels.push(label);
      });
      this.updateEdgeJumpLabels();
    },
    
    updateEdgeJumpLabels() {
      if (!this.brain.edgeJumpLabels.length || !this.brain.canvas) return;
      this.brain.canvas.origin.updateMatrixWorld(true);
      this.brain.canvas.mainCamera.updateMatrixWorld(true);
      const tooltipHidden = this.els.nodeTooltip.classList.contains('hidden');
      const tooltipRect = tooltipHidden ? null : this.els.nodeTooltip.getBoundingClientRect();
      const chartRect = this.els.networkChart.getBoundingClientRect();
      this.brain.edgeJumpLabels.forEach((label, index) => {
        const from = this.findRenderedNode(label.userData.from);
        const to = this.findRenderedNode(label.userData.to);
        if (!from || !to || tooltipHidden) {
          label.classList.add('hidden');
          return;
        }
        const fromScreen = new this.THREE.Vector3(from.x, from.y, from.z)
          .applyMatrix4(this.brain.canvas.origin.matrixWorld)
          .project(this.brain.canvas.mainCamera);
        const toScreen = new this.THREE.Vector3(to.x, to.y, to.z)
          .applyMatrix4(this.brain.canvas.origin.matrixWorld)
          .project(this.brain.canvas.mainCamera);
        if (fromScreen.z < -1 || fromScreen.z > 1 || toScreen.z < -1 || toScreen.z > 1) {
          label.classList.add('hidden');
          return;
        }
        const fromX = (fromScreen.x * 0.5 + 0.5) * this.els.networkChart.clientWidth;
        const fromY = (-fromScreen.y * 0.5 + 0.5) * this.els.networkChart.clientHeight;
        const toX = (toScreen.x * 0.5 + 0.5) * this.els.networkChart.clientWidth;
        const toY = (-toScreen.y * 0.5 + 0.5) * this.els.networkChart.clientHeight;
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
        label.style.left = `${Math.max(8, Math.min(this.els.networkChart.clientWidth - 96, x))}px`;
        label.style.top = `${Math.max(8, Math.min(this.els.networkChart.clientHeight - 26, y))}px`;
        label.style.transform = `translate(-50%, -50%) rotate(${angle}deg)`;
        label.classList.remove('hidden');
      });
    },
    
    updateActiveTooltipPosition() {
      if (!this.brain.canvas || !this.brain.activeTooltipNode || this.els.nodeTooltip.classList.contains('hidden')) return;
      const node = this.findRenderedNode(this.brain.activeTooltipNode);
      if (!node) return;
      this.brain.canvas.origin.updateMatrixWorld(true);
      this.brain.canvas.mainCamera.updateMatrixWorld(true);
      const worldPosition = new this.THREE.Vector3(node.x, node.y, node.z).applyMatrix4(this.brain.canvas.origin.matrixWorld);
      const screenPosition = worldPosition.clone().project(this.brain.canvas.mainCamera);
      const x = (screenPosition.x * 0.5 + 0.5) * this.els.networkChart.clientWidth;
      const y = (-screenPosition.y * 0.5 + 0.5) * this.els.networkChart.clientHeight;
      if (screenPosition.z < -1 || screenPosition.z > 1) {
        this.els.nodeTooltip.classList.add('hidden');
        this.setNodeLabelVisible(this.brain.activeTooltipNode, true);
        this.brain.activeTooltipNode = null;
        this.clearBrainRegionHighlight();
        this.setSelectedNodeGlow(null);
        this.clearEdgeJumpLabels();
        return;
      }
      const tooltipWidth = 285;
      const preferLeft = x > this.els.networkChart.clientWidth - tooltipWidth - 120;
      const tooltipX = preferLeft ? x - tooltipWidth - 14 : x + 14;
      this.els.nodeTooltip.style.left = `${Math.max(12, Math.min(tooltipX, this.els.networkChart.clientWidth - tooltipWidth - 12))}px`;
      this.els.nodeTooltip.style.top = `${Math.max(12, Math.min(y + 14, this.els.networkChart.clientHeight - 190))}px`;
    },
    
    cancelBrainFocusAnimation() {
      this.brain.focusAnimation = null;
    },
    
    focusNode(node, onComplete = null) {
      if (!this.brain.canvas || !node) return;
      this.hideNodeTooltip();
      const normal = node.normal
        ? new this.THREE.Vector3(node.normal.x, node.normal.y, node.normal.z).normalize()
        : new this.THREE.Vector3(node.x, node.y, node.z - 18).normalize();
      const fromQuaternion = this.brain.canvas.origin.quaternion.clone();
      const currentNormal = normal.clone().applyQuaternion(fromQuaternion).normalize();
      const focusRotation = new this.THREE.Quaternion().setFromUnitVectors(currentNormal, new this.THREE.Vector3(0, 0, 1));
      const toQuaternion = this.keepBrainOrientationReadable(focusRotation.multiply(fromQuaternion));
      const rotatedNode = new this.THREE.Vector3(node.x, node.y, node.z).applyQuaternion(toQuaternion);
      const leftOffset = -10;
      const toPosition = new this.THREE.Vector3(leftOffset - rotatedNode.x, -rotatedNode.y, 42 - rotatedNode.z);
      this.brain.focusAnimation = {
        startedAt: performance.now(),
        duration: 760,
        fromQuaternion,
        toQuaternion,
        fromPosition: this.brain.canvas.origin.position.clone(),
        toPosition,
        fromZoom: this.brain.canvas.mainCamera.zoom,
        toZoom: 2.78,
        onComplete
      };
    }
  });
}
