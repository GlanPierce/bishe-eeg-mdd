export function installSurfaceEffects(BrainScene) {
  Object.assign(BrainScene.prototype, {
    addRegionGlowSurface(node, radius) {
      const targetHemisphere = this.hemisphereId(node.hemisphere);
      const center = new this.THREE.Vector3(node.x, node.y, node.z);
      this.brain.surfaceObjects.forEach((surface) => {
        const hemisphere = surface.userData?.hemisphere
          || (surface.name?.includes('left') ? 'left' : surface.name?.includes('right') ? 'right' : null);
        if (targetHemisphere < 0 && hemisphere !== 'left') return;
        if (targetHemisphere > 0 && hemisphere !== 'right') return;
        const material = new this.THREE.ShaderMaterial({
          uniforms: {
            time: { value: 0 },
            highlightCenter: { value: center },
            highlightRadius: { value: radius },
            highlightHemisphere: { value: targetHemisphere }
          },
          transparent: true,
          depthTest: true,
          depthWrite: false,
          side: this.THREE.DoubleSide,
          blending: this.THREE.AdditiveBlending,
          vertexShader: `
            varying vec3 vBrainPosition;
            void main() {
              vBrainPosition = position;
              vec3 displaced = position + normal * 0.42;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
            }
          `,
          fragmentShader: `
            uniform float time;
            uniform vec3 highlightCenter;
            uniform float highlightRadius;
            uniform int highlightHemisphere;
            varying vec3 vBrainPosition;
            void main() {
              float heightOpacity = smoothstep(-4.0, 78.0, vBrainPosition.z);
              float bottomFade = smoothstep(-28.0, -2.0, vBrainPosition.z);
              float brainOpacity = mix(0.03, 0.48, heightOpacity) * bottomFade;
              float distanceMask = 1.0 - smoothstep(highlightRadius * 0.42, highlightRadius, distance(vBrainPosition, highlightCenter));
              float breathe = 0.82 + 0.18 * sin(time * 1.25);
              float alpha = distanceMask * brainOpacity * breathe * 0.34;
              gl_FragColor = vec4(vec3(1.0), alpha);
            }
          `
        });
        const mesh = new this.THREE.Mesh(surface.geometry.clone(), material);
        mesh.renderOrder = 6;
        this.brain.canvas.add_to_scene(mesh);
        this.brain.regionGlowSurfaces.push(mesh);
      });
    },
    
    clearBrainRegionHighlight() {
      this.brain.highlightParticles.forEach((particles) => {
        particles.geometry?.dispose?.();
        particles.material?.dispose?.();
        particles.removeFromParent();
      });
      this.brain.highlightParticles = [];
      this.brain.regionGlowSurfaces.forEach((surface) => {
        surface.geometry?.dispose?.();
        surface.material?.dispose?.();
        surface.removeFromParent();
      });
      this.brain.regionGlowSurfaces = [];
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    clearAllHighlightEffects() {
      this.clearBrainRegionHighlight();
      this.clearSelectedGlowParticles();
      this.setSelectedNodeGlow(null);
      this.clearEdgeJumpLabels();
    },
    
    isInHighlightedRegion(point, node, surfaceHemisphere) {
      const targetHemisphere = this.hemisphereId(node.hemisphere);
      const isMidline = targetHemisphere === 0 || node.hemisphere === 'midline';
      if (isMidline && Math.abs(point.x) > 18) return false;
      const x = Math.abs(point.x);
      const lateralX = surfaceHemisphere === 'left' ? Math.abs(point.x + 42) : Math.abs(point.x - 42);
      switch (node.region) {
        case 'frontal':
          return point.y > 22 && point.z > -18;
        case 'central':
          return point.y > -16 && point.y < 26 && point.z > 18 && x < 70;
        case 'parietal':
          return point.y > -58 && point.y < -18 && point.z > 18 && x < 74;
        case 'occipital':
          return point.y < -58 && point.z > -14;
        case 'temporal':
          return lateralX > 18 && point.z > -30 && point.z < 48 && point.y > -46 && point.y < 42;
        case 'auxiliary':
          return lateralX > 28 && point.z > -24 && point.y > -34 && point.y < 34;
        default:
          return isMidline ? point.z > 2 : false;
      }
    },
    
    updateBrainRegionParticles() {
      if (!this.brain.highlightParticles.length && !this.brain.regionGlowSurfaces.length) return;
      const now = performance.now();
      this.brain.regionGlowSurfaces.forEach((surface) => {
        if (surface.material?.uniforms?.time) {
          surface.material.uniforms.time.value = now / 1000;
        }
      });
      this.brain.highlightParticles.forEach((particles) => {
        if (particles.material?.uniforms?.time) {
          particles.material.uniforms.time.value = (now - particles.userData.startedAt) / 1000;
        } else if (particles.material) {
          particles.material.opacity = 0.32 + 0.075 * Math.sin(now * 0.0022);
        }
      });
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    applyBrainOpacityGradient(material, hemisphere) {
      const centerX = hemisphere === 'left' ? -33 : 33;
      material.onBeforeCompile = (shader) => {
        shader.uniforms.opacityCenterX = { value: centerX };
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', '#include <common>\nvarying vec3 vBrainPosition;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvBrainPosition = transformed;');
        shader.fragmentShader = shader.fragmentShader
          .replace(
            '#include <common>',
            `#include <common>
    uniform float opacityCenterX;
    varying vec3 vBrainPosition;`
          )
          .replace(
            '#include <dithering_fragment>',
            `float heightOpacity = smoothstep(-4.0, 78.0, vBrainPosition.z);
    float bottomFade = smoothstep(-28.0, -2.0, vBrainPosition.z);
    float brainOpacity = mix(0.03, 0.48, heightOpacity) * bottomFade;
    gl_FragColor.a *= brainOpacity;
    #include <dithering_fragment>`
          );
      };
      material.needsUpdate = true;
    },
    
    async addBrainSurface() {
      this.brain.surfaceObjects.forEach((object) => object.userData?.dispose?.());
      this.brain.surfaceData = [];
      const surfaces = [
        ['left', await this.loadFreeSurferSurface(this.lhPialUrl)],
        ['right', await this.loadFreeSurferSurface(this.rhPialUrl)]
      ];
      this.brain.surfaceObjects = surfaces.map(([hemisphere, mesh]) => {
        const inst = this.brain.canvas.add_object({
          type: 'free',
          ...mesh,
          subject_code: 'EEG_TEMPLATE',
          hemisphere,
          fileName: 'pial'
        });
        inst.object.geometry.computeVertexNormals();
        this.brain.surfaceData.push({
          hemisphere,
          positions: inst.object.geometry.attributes.position.array,
          normals: inst.object.geometry.attributes.normal.array
        });
        inst.object.material.color.set('#ffffff');
        inst.object.material.opacity = 1;
        inst.object.material.transparent = true;
        inst.object.material.depthWrite = false;
        inst.object.material.side = this.THREE.DoubleSide;
        this.applyBrainOpacityGradient(inst.object.material, hemisphere);
        inst.object.userData.hemisphere = hemisphere;
        inst.object.renderOrder = -10;
        return inst.object;
      });
    },
    
    projectNodeToSurface(node) {
      const candidates = node.hemisphere === 'left'
        ? this.brain.surfaceData.filter((surface) => surface.hemisphere === 'left')
        : node.hemisphere === 'right'
          ? this.brain.surfaceData.filter((surface) => surface.hemisphere === 'right')
          : this.brain.surfaceData;
      let bestSurface = null;
      let bestIndex = 0;
      let bestDistance = Infinity;
      candidates.forEach((surface) => {
        const positions = surface.positions;
        for (let i = 0; i < positions.length; i += 3) {
          const dx = positions[i] - node.x;
          const dy = positions[i + 1] - node.y;
          const z = positions[i + 2];
          const dist = dx * dx + dy * dy + Math.max(0, 70 - z) * 55;
          if (dist < bestDistance) {
            bestDistance = dist;
            bestSurface = surface;
            bestIndex = i;
          }
        }
      });
      if (!bestSurface) return node;
      const positions = bestSurface.positions;
      const normals = bestSurface.normals;
      const cx = bestSurface.hemisphere === 'left' ? -33 : 33;
      const radial = new this.THREE.Vector3(
        positions[bestIndex] - cx,
        positions[bestIndex + 1],
        positions[bestIndex + 2] - 10
      ).normalize();
      let nx = normals[bestIndex] || radial.x;
      let ny = normals[bestIndex + 1] || radial.y;
      let nz = normals[bestIndex + 2] || radial.z;
      if (nz < 0.18) {
        nx = radial.x;
        ny = radial.y;
        nz = Math.max(0.28, radial.z);
      }
      const normal = new this.THREE.Vector3(nx, ny, nz).normalize();
      const offset = 1.35;
      return {
        ...node,
        x: positions[bestIndex] + normal.x * offset,
        y: positions[bestIndex + 1] + normal.y * offset,
        z: positions[bestIndex + 2] + normal.z * offset,
        normal: { x: normal.x, y: normal.y, z: normal.z }
      };
    }
  });
}
