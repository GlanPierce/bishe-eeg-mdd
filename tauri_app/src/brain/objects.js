export function installObjects(BrainScene) {
  Object.assign(BrainScene.prototype, {
    createTextSprite(text, position) {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 192;
      const ctx = canvas.getContext('2d');
      ctx.font = '700 76px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
      ctx.fillText(text, 256, 96);
      const texture = new this.THREE.CanvasTexture(canvas);
      texture.generateMipmaps = false;
      texture.minFilter = this.THREE.LinearFilter;
      texture.magFilter = this.THREE.LinearFilter;
      texture.needsUpdate = true;
      const material = new this.THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
      const sprite = new this.THREE.Sprite(material);
      const normal = position.normal ? new this.THREE.Vector3(position.normal.x, position.normal.y, position.normal.z) : new this.THREE.Vector3(0, 0, 1);
      sprite.position.set(
        position.x + normal.x * 5.5,
        position.y + normal.y * 5.5,
        position.z + normal.z * 5.5
      );
      sprite.scale.set(18, 7.2, 1);
      sprite.renderOrder = 20;
      sprite.userData.eegNodeName = text;
      sprite.userData.baseOpacity = material.opacity;
      this.brain.canvas.add_to_scene(sprite);
      this.brain.labels.push(sprite);
    },
    
    addElectrodeNode(node, index) {
      const normal = node.normal ? new this.THREE.Vector3(node.normal.x, node.normal.y, node.normal.z) : new this.THREE.Vector3(0, 0, 1);
      const nodeColor = this.nodeColor(node);
      const glowMaterial = new this.THREE.SpriteMaterial({
        map: this.getGlowTexture(),
        color: nodeColor,
        transparent: true,
        opacity: 0.26,
        depthTest: true,
        depthWrite: false,
        blending: this.THREE.AdditiveBlending
      });
      const glow = new this.THREE.Sprite(glowMaterial);
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
      this.brain.canvas.add_to_scene(glow);
      this.brain.nodeGlowObjects.push(glow);
    
      const material = new this.THREE.SpriteMaterial({
        map: this.getNodeDotTexture(),
        color: nodeColor,
        transparent: true,
        opacity: 0.96,
        depthTest: true,
        depthWrite: false
      });
      const object = new this.THREE.Sprite(material);
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
      this.brain.canvas.add_to_scene(object);
      this.brain.nodeObjects.push(object);
      this.createTextSprite(node.name, node);
    },
    
    makeConnectionCurve(source, target, strength, curveOffset = 0) {
      const sourceNormal = source.normal
        ? new this.THREE.Vector3(source.normal.x, source.normal.y, source.normal.z).normalize()
        : new this.THREE.Vector3(source.x, source.y, source.z - 18).normalize();
      const targetNormal = target.normal
        ? new this.THREE.Vector3(target.normal.x, target.normal.y, target.normal.z).normalize()
        : new this.THREE.Vector3(target.x, target.y, target.z - 18).normalize();
      const start = new this.THREE.Vector3(source.x, source.y, source.z);
      const end = new this.THREE.Vector3(target.x, target.y, target.z);
      const mid = start.clone().add(end).multiplyScalar(0.5);
      const outward = sourceNormal.clone().add(targetNormal).normalize();
      if (!Number.isFinite(outward.x)) outward.copy(mid.clone().sub(new this.THREE.Vector3(0, 0, 18)).normalize());
      const side = new this.THREE.Vector3().subVectors(end, start).cross(outward).normalize();
      if (!Number.isFinite(side.x)) side.set(0, 1, 0);
      const distance = start.distanceTo(end);
      mid.add(outward.multiplyScalar(Math.min(7.2, 0.032 * distance + strength * 1.25 + 1.25)));
      mid.add(side.multiplyScalar(curveOffset * 0.45));
      return new this.THREE.QuadraticBezierCurve3(start, mid, end);
    },
    
    makeConnectionGeometry(curve, strength) {
      return new this.THREE.TubeGeometry(curve, 40, 0.045 + strength * 0.22, 5, false);
    },

    edgeContribution(edge, source, target) {
      const fallback = (Number(source.contribution || 0) + Number(target.contribution || 0)) / 2;
      if (Number.isFinite(Number(edge.contribution))) return Number(edge.contribution);
      if (Number(source.contribution || 0) !== 0 || Number(target.contribution || 0) !== 0) return fallback;
      return null;
    },

    mixHexColor(base, target, ratio) {
      const b = new this.THREE.Color(base);
      const t = new this.THREE.Color(target);
      b.lerp(t, Math.max(0, Math.min(1, ratio)));
      return `#${b.getHexString()}`;
    },

    edgeColor(edge, source, target) {
      const contribution = this.edgeContribution(edge, source, target);
      if (contribution === null || contribution === 0) return '#f7f7f2';
      const maxAbs = Math.max(1e-9, Number(edge.colorMaxAbs || 0));
      const ratio = Math.pow(Math.min(1, Math.abs(contribution) / maxAbs), 0.72);
      return contribution >= 0
        ? this.mixHexColor('#f7f7f2', '#d8948d', ratio)
        : this.mixHexColor('#f7f7f2', '#93c69b', ratio);
    },

    nodeColor(node) {
      const contribution = Number(node.contribution || node.value || 0);
      if (!contribution) return '#f7f7f2';
      const maxAbs = Math.max(1e-9, Number(node.colorMaxAbs || 0));
      const ratio = Math.pow(Math.min(1, Math.abs(contribution) / maxAbs), 0.72);
      return contribution >= 0
        ? this.mixHexColor('#f7f7f2', '#d8948d', ratio)
        : this.mixHexColor('#f7f7f2', '#93c69b', ratio);
    },
    
    addEdgeObject(edge, nodeById, curveOffset = 0) {
      const source = nodeById.get(this.edgeEndpointName(edge.source));
      const target = nodeById.get(this.edgeEndpointName(edge.target));
      if (!source || !target) return;
      const strength = Math.max(0.08, Math.min(1, Number(edge.score || edge.strength || Math.abs(edge.value) || 0.2)));
      const curve = this.makeConnectionCurve(source, target, strength, curveOffset);
      const edgeColor = this.edgeColor(edge, source, target);
      const material = new this.THREE.ShaderMaterial({
        uniforms: {
          color: { value: new this.THREE.Color(edgeColor) },
          baseOpacity: { value: 0.34 + strength * 0.42 },
          focusCenter: { value: new this.THREE.Vector3() },
          focusActive: { value: 0 }
        },
        transparent: true,
        depthTest: true,
        depthWrite: false,
        vertexShader: `
          uniform vec3 focusCenter;
          uniform float focusActive;
          varying float vDistanceFade;
          void main() {
            float proximity = clamp(1.0 - distance(position, focusCenter) / 105.0, 0.0, 1.0);
            float focusFade = mix(0.008, 1.0, pow(proximity, 1.85));
            vDistanceFade = mix(1.0, focusFade, focusActive);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 color;
          uniform float baseOpacity;
          varying float vDistanceFade;
          void main() {
            gl_FragColor = vec4(color, baseOpacity * vDistanceFade);
          }
        `
      });
      const mesh = new this.THREE.Mesh(this.makeConnectionGeometry(curve, strength), material);
      mesh.userData.edge = edge;
      mesh.userData.sourceName = source.name;
      mesh.userData.targetName = target.name;
      mesh.userData.baseOpacity = material.uniforms.baseOpacity.value;
      mesh.renderOrder = 5;
      this.brain.canvas.add_to_scene(mesh);
      this.brain.edgeObjects.push(mesh);
      this.addConnectionSignalObject(curve, strength, source.name, target.name, edgeColor);
    },
    
    addConnectionSignalObject(curve, strength, sourceName, targetName, color = '#ffffff') {
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
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('pathT', new this.THREE.BufferAttribute(pathT, 1));
      const material = new this.THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          strength: { value: strength },
          fade: { value: 1 },
          focusCenter: { value: new this.THREE.Vector3() },
          focusActive: { value: 0 },
          color: { value: new this.THREE.Color(color) }
        },
        transparent: true,
        depthTest: true,
        depthWrite: false,
        blending: this.THREE.AdditiveBlending,
        vertexShader: `
          attribute float pathT;
          uniform float time;
          uniform float strength;
          uniform float fade;
          uniform vec3 focusCenter;
          uniform float focusActive;
          varying float vAlpha;
          void main() {
            float phase = fract(time * (0.24 + strength * 0.26));
            float d = min(abs(pathT - phase), 1.0 - abs(pathT - phase));
            float head = exp(-d * d / (0.0022 + strength * 0.0018));
            float tailDistance = mod(phase - pathT + 1.0, 1.0);
            float tail = exp(-tailDistance * tailDistance / 0.0065) * 0.28;
            float proximity = clamp(1.0 - distance(position, focusCenter) / 105.0, 0.0, 1.0);
            float distanceFade = mix(1.0, mix(0.008, 1.0, pow(proximity, 1.85)), focusActive);
            vAlpha = clamp((head + tail) * (0.72 + strength * 0.58) * fade * distanceFade, 0.0, 0.82);
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
      const signal = new this.THREE.LineSegments(geometry, material);
      signal.renderOrder = 22;
      signal.userData.startedAt = performance.now();
      signal.userData.sourceName = sourceName;
      signal.userData.targetName = targetName;
      signal.userData.baseOpacity = 1;
      signal.userData.focusFade = 1;
      this.brain.canvas.add_to_scene(signal);
      this.brain.signalObjects.push(signal);
      const haloMaterial = material.clone();
      haloMaterial.uniforms = {
        time: { value: 0 },
        strength: { value: strength },
        fade: { value: 0.46 },
        focusCenter: { value: new this.THREE.Vector3() },
        focusActive: { value: 0 },
        color: { value: new this.THREE.Color(color) }
      };
      haloMaterial.blending = this.THREE.AdditiveBlending;
      const halo = new this.THREE.LineSegments(geometry.clone(), haloMaterial);
      halo.renderOrder = 21;
      halo.scale.set(1.003, 1.003, 1.003);
      halo.userData.startedAt = performance.now();
      halo.userData.isSignalHalo = true;
      halo.userData.sourceName = sourceName;
      halo.userData.targetName = targetName;
      halo.userData.baseOpacity = 1;
      halo.userData.focusFade = 1;
      this.brain.canvas.add_to_scene(halo);
      this.brain.signalObjects.push(halo);
    },
    
    updateConnectionSignals() {
      if (!this.brain.signalObjects.length) return;
      const now = performance.now();
      this.brain.signalObjects.forEach((signal) => {
        if (signal.material?.uniforms?.time) {
          signal.material.uniforms.time.value = (now - signal.userData.startedAt) / 1000;
          const baseFade = signal.userData.isSignalHalo ? 0.46 : 1;
          signal.material.uniforms.fade.value = baseFade * (signal.userData.focusFade ?? 1);
        }
      });
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    drawBrainNetwork(nodes, edges = []) {
      if (!this.brain.canvas) return;
      this.resetBrainObjects();
      const projectedNodes = nodes.map((node) => this.projectNodeToSurface(node));
      const maxNodeContribution = Math.max(...projectedNodes.map((node) => Math.abs(Number(node.contribution || node.value || 0))), 0);
      projectedNodes.forEach((node) => {
        node.colorMaxAbs = maxNodeContribution;
      });
      const nodeById = new Map(projectedNodes.map((node) => [node.name, node]));
      const edgeCounts = new Map();
      edges.forEach((edge) => {
        [this.edgeEndpointName(edge.source), this.edgeEndpointName(edge.target)].forEach((name) => {
          edgeCounts.set(name, (edgeCounts.get(name) || 0) + 1);
        });
      });
      const edgeSlots = new Map();
      const edgeContributions = edges.map((edge) => {
        const source = nodeById.get(this.edgeEndpointName(edge.source));
        const target = nodeById.get(this.edgeEndpointName(edge.target));
        return source && target ? this.edgeContribution(edge, source, target) : null;
      });
      const maxEdgeContribution = Math.max(...edgeContributions.map((value) => Math.abs(value || 0)), 0);
      edges.forEach((edge) => {
        const sourceName = this.edgeEndpointName(edge.source);
        const targetName = this.edgeEndpointName(edge.target);
        const key = edgeCounts.get(sourceName) >= edgeCounts.get(targetName) ? sourceName : targetName;
        const slot = edgeSlots.get(key) || 0;
        const total = Math.max(1, edgeCounts.get(key) || 1);
        edgeSlots.set(key, slot + 1);
        const centeredSlot = slot - (total - 1) / 2;
        this.addEdgeObject({ ...edge, colorMaxAbs: maxEdgeContribution }, nodeById, centeredSlot * 1.2);
      });
      projectedNodes.forEach((node, index) => this.addElectrodeNode(node, index));
      this.brain.nodes = projectedNodes;
      this.brain.edges = edges;
      this.brain.canvas.needsUpdate = true;
    }
  });
}
