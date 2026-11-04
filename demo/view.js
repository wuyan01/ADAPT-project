import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';

export class View {
  constructor(stage) {
    this.stage = stage; this.meshes = new Map(); this.target = new THREE.Vector3();
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#111c18');
    this.scene.fog = new THREE.Fog('#111c18', 18, 55);
    this.camera = new THREE.PerspectiveCamera(45, 1, .05, 150);
    this.camera.up.set(0, 0, 1); this.camera.position.set(2.8, -2.8, 1.7);
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    stage.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, .75); this.controls.enableDamping = true;
    this.controls.minDistance = 1.5; this.controls.maxDistance = 15;
    this.controls.maxPolarAngle = Math.PI * .48;
    this.scene.add(new THREE.HemisphereLight('#e6f5e1', '#35422a', 2.3));
    // The shadow frustum is a small box that follows the robot (see poses()); a fixed box at the
    // origin leaves the rest of the floor unshadowed and the robot loses its shadow once it walks out.
    const sun = new THREE.DirectionalLight('#fff4dc', 3);
    this.sunOffset = new THREE.Vector3(3, -4, 8);
    sun.position.copy(this.sunOffset); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -6; sun.shadow.camera.right = 6; sun.shadow.camera.top = 6; sun.shadow.camera.bottom = -6;
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 25;
    // Bias removes the faint self-shadowing ("acne") on the flat floor that made the lit area inside
    // the old frustum look darker than the floor outside it.
    sun.shadow.bias = -0.0002; sun.shadow.normalBias = 0.02;
    this.sun = sun; this.scene.add(sun); this.scene.add(sun.target);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(150, 150), new THREE.MeshStandardMaterial({color: '#26372b', roughness: 1}));
    floor.receiveShadow = true; this.scene.add(floor);
    const grid = new THREE.GridHelper(100, 100, '#647154', '#394a38'); grid.rotation.x = Math.PI / 2; grid.position.z = .002; this.scene.add(grid);
    this.goal = new THREE.Mesh(new THREE.RingGeometry(.47, .5, 64), new THREE.MeshBasicMaterial({color: '#edbc75', side: THREE.DoubleSide}));
    this.goal.position.z = .008; this.goal.visible = false; this.scene.add(this.goal);
    this.resize = new ResizeObserver(() => {
      const {width, height} = stage.getBoundingClientRect();
      this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    }); this.resize.observe(stage);
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }
  geometry(geoms) {
    for (const item of geoms) {
      let geometry;
      if (item.vertices) {
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(item.vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(item.faces, 1)); geometry.computeVertexNormals();
      } else if (item.type === 2) geometry = new THREE.SphereGeometry(item.size[0], 16, 12);
      else geometry = new THREE.BoxGeometry(...item.size.map(s => s * 2));
      const color = new THREE.Color(...item.color.slice(0, 3));
      const material = new THREE.MeshStandardMaterial({color, metalness: .25, roughness: .48});
      const mesh = new THREE.Mesh(geometry, material); mesh.castShadow = true; mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false; this.meshes.set(item.id, mesh); this.scene.add(mesh);
    }
  }
  poses(packet) {
    const {positions: p, rotations: r} = packet;
    for (const [i, mesh] of this.meshes) {
      const a = i * 9, b = i * 3;
      mesh.matrix.set(r[a], r[a + 1], r[a + 2], p[b], r[a + 3], r[a + 4], r[a + 5], p[b + 1], r[a + 6], r[a + 7], r[a + 8], p[b + 2], 0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
    const next = new THREE.Vector3(packet.root[0], packet.root[1], Math.max(.5, packet.root[2]));
    const delta = next.clone().sub(this.controls.target); this.camera.position.add(delta); this.controls.target.copy(next);
    // Move the light and its shadow box with the robot. Snapping to shadow-map texels avoids shimmering edges.
    const texel = (this.sun.shadow.camera.right - this.sun.shadow.camera.left) / this.sun.shadow.mapSize.x;
    const sx = Math.round(packet.root[0] / texel) * texel, sy = Math.round(packet.root[1] / texel) * texel;
    this.sun.target.position.set(sx, sy, 0);
    this.sun.position.set(sx + this.sunOffset.x, sy + this.sunOffset.y, this.sunOffset.z);
    this.goal.visible = !!packet.goal;
    if (packet.goal) { this.goal.position.set(...packet.goal, .008); this.goal.material.color.set(packet.reached ? '#a4db7a' : '#edbc75'); }
  }
}
