export function installSurfaceBase(BrainScene) {
  Object.assign(BrainScene.prototype, {
    resetBrainObjects() {
      this.brain.edgeObjects.forEach((object) => {
        object.geometry?.dispose?.();
        object.material?.dispose?.();
        object.removeFromParent();
      });
      this.brain.signalObjects.forEach((object) => {
        object.traverse?.((child) => {
          child.geometry?.dispose?.();
          child.material?.dispose?.();
        });
        object.removeFromParent();
      });
      this.brain.edgeJumpLabels.forEach((label) => label.remove());
      this.brain.edgeJumpLabels = [];
      this.brain.nodeObjects.forEach((object) => object.userData?.dispose?.());
      this.brain.nodeGlowObjects.forEach((object) => object.userData?.dispose?.());
      this.brain.labels.forEach((object) => {
        object.material?.map?.dispose?.();
        object.material?.dispose?.();
        object.removeFromParent();
      });
      this.brain.edgeObjects = [];
      this.brain.signalObjects = [];
      this.brain.nodeObjects = [];
      this.brain.nodeGlowObjects = [];
      this.brain.labels = [];
      this.brain.selectedGlowNode = null;
      this.brain.activeTooltipNode = null;
      this.els.nodeTooltip.classList.add('hidden');
      this.clearAllHighlightEffects();
    },
    
    getNodeDotTexture() {
      if (this.brain.dotTexture) return this.brain.dotTexture;
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 64, 64);
      ctx.beginPath();
      ctx.arc(32, 32, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      this.brain.dotTexture = new this.THREE.CanvasTexture(canvas);
      return this.brain.dotTexture;
    },
    
    getGlowTexture() {
      if (this.brain.glowTexture) return this.brain.glowTexture;
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
      this.brain.glowTexture = new this.THREE.CanvasTexture(canvas);
      this.brain.glowTexture.minFilter = this.THREE.LinearFilter;
      this.brain.glowTexture.magFilter = this.THREE.LinearFilter;
      this.brain.glowTexture.generateMipmaps = false;
      return this.brain.glowTexture;
    },
    
    regionId(region) {
      return {
        frontal: 1,
        central: 2,
        parietal: 3,
        occipital: 4,
        temporal: 5,
        auxiliary: 6
      }[region] || 0;
    },
    
    hemisphereId(hemisphere) {
      return hemisphere === 'left' ? -1 : hemisphere === 'right' ? 1 : 0;
    },
    
    regionMaskExpression(prefix = 'vBrainPosition') {
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
    },
    
    setBrainRegionHighlight(node) {
      this.clearBrainRegionHighlight();
      if (!node || !this.brain.canvas || !this.brain.surfaceData.length) return;
      const targetHemisphere = this.hemisphereId(node.hemisphere);
      const center = new this.THREE.Vector3(node.x, node.y, node.z);
      const radius = node.region === 'temporal' || node.region === 'auxiliary' ? 27 : 31;
      this.addRegionGlowSurface(node, radius);
      const glowPositions = [];
      const particlePositions = [];
      const normals = [];
      const seeds = [];
    
      this.brain.surfaceData.forEach((surface) => {
        if (targetHemisphere < 0 && surface.hemisphere !== 'left') return;
        if (targetHemisphere > 0 && surface.hemisphere !== 'right') return;
        const sourcePositions = surface.positions;
        const sourceNormals = surface.normals;
        const stride = Math.max(1, Math.floor(sourcePositions.length / 3 / 2400));
        for (let i = 0; i < sourcePositions.length; i += 3 * stride) {
          const x = sourcePositions[i];
          const y = sourcePositions[i + 1];
          const z = sourcePositions[i + 2];
          const distance = center.distanceTo(new this.THREE.Vector3(x, y, z));
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
      const glowGeometry = new this.THREE.BufferGeometry();
      glowGeometry.setAttribute('position', new this.THREE.Float32BufferAttribute(glowPositions, 3));
      const glowSeeds = new Float32Array(glowPositions.length / 3);
      for (let i = 0; i < glowSeeds.length; i += 1) glowSeeds[i] = Math.random();
      glowGeometry.setAttribute('seed', new this.THREE.BufferAttribute(glowSeeds, 1));
      const glowMaterial = new this.THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          color: { value: new this.THREE.Color('#ffffff') }
        },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: this.THREE.AdditiveBlending,
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
      const glow = new this.THREE.Points(glowGeometry, glowMaterial);
      glow.renderOrder = 7;
      glow.userData.startedAt = performance.now();
      this.brain.canvas.add_to_scene(glow);
      this.brain.highlightParticles.push(glow);
    
      if (!particlePositions.length) {
        if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
        return;
      }
      const particleGeometry = new this.THREE.BufferGeometry();
      particleGeometry.setAttribute('position', new this.THREE.Float32BufferAttribute(particlePositions, 3));
      particleGeometry.setAttribute('normalLift', new this.THREE.Float32BufferAttribute(normals, 3));
      particleGeometry.setAttribute('seed', new this.THREE.Float32BufferAttribute(seeds, 1));
      const particleMaterial = new this.THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          color: { value: new this.THREE.Color('#ffffff') }
        },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: this.THREE.AdditiveBlending,
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
      const particles = new this.THREE.Points(particleGeometry, particleMaterial);
      particles.renderOrder = 9;
      particles.userData.startedAt = performance.now();
      this.brain.canvas.add_to_scene(particles);
      this.brain.highlightParticles.push(particles);
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    }
  });
}
