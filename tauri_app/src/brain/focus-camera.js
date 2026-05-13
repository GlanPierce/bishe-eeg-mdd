export function installFocusCamera(BrainScene) {
  Object.assign(BrainScene.prototype, {
    setBrainCamera() {
      const camera = this.brain.canvas.mainCamera;
      camera.position.set(0, 0, 500);
      camera.up.set(0, 1, 0);
      camera.lookAt(0, 0, 42);
      camera.zoom = 1.02;
      camera.updateProjectionMatrix();
      this.brain.canvas.trackball?.lookAt?.({ x: 0, y: 0, z: 42 });
      this.brain.canvas.trackball?.update?.();
    },
    
    resetBrainView() {
      if (!this.brain.canvas) return;
      this.cancelBrainFocusAnimation();
      this.brain.inertia = null;
      this.hideNodeTooltip();
      this.clearBrainRegionHighlight();
      this.setSelectedNodeGlow(null);
      this.brain.focusAnimation = {
        startedAt: performance.now(),
        duration: 560,
        fromQuaternion: this.brain.canvas.origin.quaternion.clone(),
        toQuaternion: new this.THREE.Quaternion(),
        fromPosition: this.brain.canvas.origin.position.clone(),
        toPosition: new this.THREE.Vector3(0, 0, 0),
        fromZoom: this.brain.canvas.mainCamera.zoom,
        toZoom: 1.02,
        onComplete: () => this.setBrainCamera()
      };
      this.brain.canvas.needsUpdate = true;
    },
    
    easeInOutCubic(t) {
      return t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2;
    },
    
    keepModelTopReadable(quaternion) {
      const topVector = new this.THREE.Vector3(0, 0, 1).applyQuaternion(quaternion);
      topVector.z = 0;
      if (topVector.lengthSq() < 0.015) {
        topVector.copy(new this.THREE.Vector3(0, 1, 0).applyQuaternion(quaternion));
        topVector.z = 0;
      }
      if (topVector.lengthSq() < 0.015) return quaternion;
      topVector.normalize();
      if (topVector.y >= 0.22) return quaternion;
    
      const targetY = 0.46;
      const targetX = Math.sign(topVector.x || 1) * Math.sqrt(1 - targetY * targetY);
      const target = new this.THREE.Vector3(targetX, targetY, 0).normalize();
      const angle = Math.atan2(
        topVector.x * target.y - topVector.y * target.x,
        topVector.dot(target)
      );
      return new this.THREE.Quaternion()
        .setFromAxisAngle(new this.THREE.Vector3(0, 0, 1), angle)
        .multiply(quaternion);
    },
    
    keepHemispheresReadable(quaternion) {
      const rightVector = new this.THREE.Vector3(1, 0, 0).applyQuaternion(quaternion);
      rightVector.z = 0;
      if (rightVector.lengthSq() < 0.015) return quaternion;
      rightVector.normalize();
      if (rightVector.x >= 0.18) return quaternion;
    
      const targetX = 0.42;
      const targetY = Math.sign(rightVector.y || 1) * Math.sqrt(1 - targetX * targetX);
      const target = new this.THREE.Vector3(targetX, targetY, 0).normalize();
      const angle = Math.atan2(
        rightVector.x * target.y - rightVector.y * target.x,
        rightVector.dot(target)
      );
      return new this.THREE.Quaternion()
        .setFromAxisAngle(new this.THREE.Vector3(0, 0, 1), angle)
        .multiply(quaternion);
    },
    
    keepBrainOrientationReadable(quaternion) {
      return this.keepModelTopReadable(this.keepHemispheresReadable(this.keepModelTopReadable(this.keepHemispheresReadable(quaternion))));
    },
    
    updateBrainFocusAnimation() {
      if (!this.brain.focusAnimation || !this.brain.canvas) return;
      const now = performance.now();
      const progress = Math.min(1, (now - this.brain.focusAnimation.startedAt) / this.brain.focusAnimation.duration);
      const eased = this.easeInOutCubic(progress);
      this.brain.canvas.origin.quaternion.copy(this.brain.focusAnimation.fromQuaternion).slerp(this.brain.focusAnimation.toQuaternion, eased);
      this.brain.canvas.origin.position.lerpVectors(this.brain.focusAnimation.fromPosition, this.brain.focusAnimation.toPosition, eased);
      this.brain.canvas.mainCamera.zoom = this.THREE.MathUtils.lerp(this.brain.focusAnimation.fromZoom, this.brain.focusAnimation.toZoom, eased);
      this.brain.canvas.mainCamera.updateProjectionMatrix();
      this.brain.canvas.needsUpdate = true;
      if (progress >= 1) {
        const onComplete = this.brain.focusAnimation.onComplete;
        this.brain.focusAnimation = null;
        onComplete?.();
      }
    },
    
    findRenderedNode(name) {
      return this.brain.nodes.find((node) => node.name === name);
    },
    
    setNodeLabelsVisible(visible) {
      this.brain.labels.forEach((label) => {
        label.visible = visible;
      });
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    },
    
    setNodeLabelVisible(name, visible) {
      this.brain.labels.forEach((label) => {
        if (label.userData.eegNodeName === name) label.visible = visible;
      });
      if (this.brain.canvas) this.brain.canvas.needsUpdate = true;
    }
  });
}
