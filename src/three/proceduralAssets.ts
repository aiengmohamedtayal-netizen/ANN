import * as THREE from 'three';

export type FinishPreset = 'titanium_lab' | 'optical_sapphire' | 'brushed_gold';

export interface MaterialPalette {
  hullMat: THREE.MeshStandardMaterial;
  gimbalMat: THREE.MeshPhysicalMaterial;
  accentTrimMat: THREE.MeshStandardMaterial;
  glassSomaMat: THREE.MeshPhysicalMaterial;
  strutMat: THREE.MeshStandardMaterial;
  plinthMat: THREE.MeshStandardMaterial;
  contactShadowMat: THREE.MeshBasicMaterial;
}

/**
 * Generates a procedural laboratory floor grid texture (zero external URLs)
 */
export function createStudioFloorGridTexture(
  maxAnisotropy: number,
): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#090d16';
  ctx.fillRect(0, 0, size, size);

  // Subtle sub-grid (8x8)
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.55)';
  ctx.lineWidth = 1;
  const step = size / 8;
  for (let i = 0; i <= 8; i++) {
    const pos = i * step;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(size, pos);
    ctx.stroke();
  }

  // Primary border frame & precision crosshair ticks
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, size - 2, size - 2);

  ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
  ctx.fillRect(size / 2 - 6, size / 2 - 1, 12, 2);
  ctx.fillRect(size / 2 - 1, size / 2 - 6, 2, 12);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(16, 16);
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a tileable grayscale brushed-metal bump/roughness map
 */
export function createBrushedMetalBumpTexture(
  maxAnisotropy: number,
): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  // Deterministic brushed streaks
  let seed = 9001;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  for (let y = 0; y < size; y += 2) {
    const shade = Math.floor(105 + rand() * 50);
    ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
    ctx.fillRect(0, y, size, 1);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a soft radial contact shadow disc texture
 */
export function createContactShadowTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createRadialGradient(
    size / 2,
    size / 2,
    4,
    size / 2,
    size / 2,
    size / 2,
  );
  grad.addColorStop(0, 'rgba(0, 0, 0, 0.75)');
  grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.32)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates a calibrated PBR material palette with >= 0.25 roughness contrast across adjacent parts
 */
export function createMaterialPalette(
  preset: FinishPreset,
  maxAnisotropy: number,
  isHeadless: boolean,
): MaterialPalette {
  const bumpTex = createBrushedMetalBumpTexture(maxAnisotropy);
  const shadowTex = createContactShadowTexture();

  const configs: Record<
    FinishPreset,
    {
      hullColor: number;
      gimbalColor: number;
      trimColor: number;
      glassTint: number;
    }
  > = {
    titanium_lab: {
      hullColor: 0x1e293b,
      gimbalColor: 0x94a3b8,
      trimColor: 0x38bdf8,
      glassTint: 0xe0f2fe,
    },
    optical_sapphire: {
      hullColor: 0x0f172a,
      gimbalColor: 0x60a5fa,
      trimColor: 0x818cf8,
      glassTint: 0xbae6fd,
    },
    brushed_gold: {
      hullColor: 0x27272a,
      gimbalColor: 0xeab308,
      trimColor: 0xf59e0b,
      glassTint: 0xfef3c7,
    },
  };

  const cfg = configs[preset];

  // 1. Matte structural hull (roughness = 0.68, metalness = 0.45)
  const hullMat = new THREE.MeshStandardMaterial({
    color: cfg.hullColor,
    roughness: 0.68,
    metalness: 0.45,
    bumpMap: bumpTex,
    bumpScale: 0.015,
  });

  // 2. Anisotropic brushed gimbal ring (roughness = 0.20, metalness = 0.92 -> delta roughness = 0.48 >= 0.25)
  const gimbalMat = new THREE.MeshPhysicalMaterial({
    color: cfg.gimbalColor,
    roughness: 0.2,
    metalness: 0.92,
    anisotropy: isHeadless ? 0 : 0.75,
    clearcoat: isHeadless ? 0 : 0.4,
    bumpMap: bumpTex,
    bumpScale: 0.01,
  });

  // 3. High-gloss accent trim collar (roughness = 0.14, metalness = 0.85)
  const accentTrimMat = new THREE.MeshStandardMaterial({
    color: cfg.trimColor,
    roughness: 0.14,
    metalness: 0.85,
  });

  // 4. Optical glass soma shell
  const glassSomaMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.08,
    metalness: 0.05,
    transmission: isHeadless ? 0 : 0.88,
    transparent: true,
    opacity: isHeadless ? 0.34 : 1.0,
    ior: 1.52,
    thickness: 0.9,
    attenuationColor: new THREE.Color(cfg.glassTint),
    attenuationDistance: 2.2,
    clearcoat: 1.0,
    clearcoatRoughness: 0.08,
    depthWrite: false,
  });

  // 5. Satin support strut (roughness = 0.52, metalness = 0.65)
  const strutMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.52,
    metalness: 0.65,
  });

  // 6. Grounded laboratory plinth base (roughness = 0.78, metalness = 0.30)
  const plinthMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.78,
    metalness: 0.3,
  });

  const contactShadowMat = new THREE.MeshBasicMaterial({
    map: shadowTex,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
  });

  return {
    hullMat,
    gimbalMat,
    accentTrimMat,
    glassSomaMat,
    strutMat,
    plinthMat,
    contactShadowMat,
  };
}

/**
 * Shared geometries cache for composite neurons so draw calls & memory stay minimal
 */
export interface SharedNeuronGeometries {
  somaGlassGeo: THREE.LatheGeometry;
  nucleusGeo: THREE.OctahedronGeometry;
  gimbalGeo: THREE.ExtrudeGeometry;
  polarCapGeo: THREE.LatheGeometry;
  biasRingGeo: THREE.TorusGeometry;
  terminalCollarGeo: THREE.CylinderGeometry;
  axonNozzleGeo: THREE.LatheGeometry;
  coolingFinGeo: THREE.ExtrudeGeometry;
  baseFootGeo: THREE.LatheGeometry;
  contactShadowGeo: THREE.CircleGeometry;
  hitboxGeo: THREE.SphereGeometry;
}

export function createSharedNeuronGeometries(): SharedNeuronGeometries {
  // 1. Turned optical soma profile via LatheGeometry
  const somaPoints: THREE.Vector2[] = [];
  for (let i = 0; i <= 16; i++) {
    const t = (i / 16) * Math.PI;
    const r = Math.sin(t) * 0.44 * (1 + 0.06 * Math.cos(t * 2));
    const y = -Math.cos(t) * 0.46;
    somaPoints.push(new THREE.Vector2(Math.max(0.001, r), y));
  }
  const somaGlassGeo = new THREE.LatheGeometry(somaPoints, 24);

  // 2. Faceted inner activation core
  const nucleusGeo = new THREE.OctahedronGeometry(0.24, 1);

  // 3. Beveled Equatorial Gimbal Ring via ExtrudeGeometry with hole
  const outerR = 0.56;
  const innerR = 0.47;
  const gimbalShape = new THREE.Shape();
  gimbalShape.absarc(0, 0, outerR, 0, Math.PI * 2, false);
  const holePath = new THREE.Path();
  holePath.absarc(0, 0, innerR, 0, Math.PI * 2, true);
  gimbalShape.holes.push(holePath);

  const gimbalGeo = new THREE.ExtrudeGeometry(gimbalShape, {
    depth: 0.08,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.02,
    bevelThickness: 0.02,
    curveSegments: 24,
  });
  gimbalGeo.center();

  // 4. Stepped Polar Collector Cap via LatheGeometry
  const capPoints = [
    new THREE.Vector2(0.01, 0.0),
    new THREE.Vector2(0.26, 0.0),
    new THREE.Vector2(0.24, 0.07),
    new THREE.Vector2(0.16, 0.09),
    new THREE.Vector2(0.14, 0.15),
    new THREE.Vector2(0.01, 0.15),
  ];
  const polarCapGeo = new THREE.LatheGeometry(capPoints, 20);

  // 5. Bias calibration torus ring
  const biasRingGeo = new THREE.TorusGeometry(0.36, 0.028, 12, 28);

  // 6. Dendrite input hexagonal manifold collar
  const terminalCollarGeo = new THREE.CylinderGeometry(0.15, 0.19, 0.14, 6);
  terminalCollarGeo.rotateZ(Math.PI / 2);

  // 7. Axon emitter nozzle via LatheGeometry
  const nozzlePoints = [
    new THREE.Vector2(0.18, 0),
    new THREE.Vector2(0.15, 0.08),
    new THREE.Vector2(0.09, 0.18),
    new THREE.Vector2(0.12, 0.22),
    new THREE.Vector2(0.04, 0.24),
  ];
  const axonNozzleGeo = new THREE.LatheGeometry(nozzlePoints, 16);
  axonNozzleGeo.rotateZ(-Math.PI / 2);

  // 8. Beveled cooling fin via ExtrudeGeometry
  const finShape = new THREE.Shape();
  finShape.moveTo(0, -0.14);
  finShape.lineTo(0.14, -0.09);
  finShape.lineTo(0.14, 0.09);
  finShape.lineTo(0, 0.14);
  finShape.closePath();
  const coolingFinGeo = new THREE.ExtrudeGeometry(finShape, {
    depth: 0.024,
    bevelEnabled: true,
    bevelSegments: 1,
    bevelSize: 0.008,
    bevelThickness: 0.008,
  });
  coolingFinGeo.center();

  // 9. Grounded Pedestal Foot via LatheGeometry (sits flush at y=0)
  const footPoints = [
    new THREE.Vector2(0.01, 0),
    new THREE.Vector2(0.38, 0),
    new THREE.Vector2(0.34, 0.06),
    new THREE.Vector2(0.18, 0.12),
    new THREE.Vector2(0.1, 0.18),
    new THREE.Vector2(0.01, 0.18),
  ];
  const baseFootGeo = new THREE.LatheGeometry(footPoints, 20);

  // 10. Radial contact shadow disc
  const contactShadowGeo = new THREE.CircleGeometry(0.62, 24);
  contactShadowGeo.rotateX(-Math.PI / 2);

  // 11. Invisible raycast proxy sphere (Layer 1)
  const hitboxGeo = new THREE.SphereGeometry(0.62, 12, 12);

  return {
    somaGlassGeo,
    nucleusGeo,
    gimbalGeo,
    polarCapGeo,
    biasRingGeo,
    terminalCollarGeo,
    axonNozzleGeo,
    coolingFinGeo,
    baseFootGeo,
    contactShadowGeo,
    hitboxGeo,
  };
}

export interface CompositeNeuronHandle {
  rootGroup: THREE.Group; // Pivoted at bottom-center (y = 0) on the studio floor
  headGroup: THREE.Group; // Positioned at elevation y = headY
  nucleusMesh: THREE.Mesh;
  nucleusMat: THREE.MeshStandardMaterial;
  biasRingMesh: THREE.Mesh;
  biasRingMat: THREE.MeshStandardMaterial;
  selectionRingMesh: THREE.Mesh;
  selectionRingMat: THREE.MeshBasicMaterial;
  hitboxMesh: THREE.Mesh;
  layerIndex: number;
  neuronIndex: number;
  basePos: THREE.Vector3;
  headY: number;
}

/**
 * Assembles an 8-part procedural Neuron Processing Unit + grounded support column & contact shadow
 */
export function buildCompositeNeuronGroup(
  layerIndex: number,
  neuronIndex: number,
  worldX: number,
  headY: number,
  worldZ: number,
  geos: SharedNeuronGeometries,
  palette: MaterialPalette,
  castShadow: boolean,
): CompositeNeuronHandle {
  // Root group sits flush on the ground plane (y = 0)
  const rootGroup = new THREE.Group();
  rootGroup.position.set(worldX, 0, worldZ);

  // Ground contact shadow at y = 0.015 (disjoint depth from floor at y = 0)
  const shadowMesh = new THREE.Mesh(
    geos.contactShadowGeo,
    palette.contactShadowMat,
  );
  shadowMesh.position.y = 0.015;
  shadowMesh.renderOrder = 1;
  rootGroup.add(shadowMesh);

  // Grounded pedestal base foot at y = 0
  const baseFoot = new THREE.Mesh(geos.baseFootGeo, palette.hullMat);
  baseFoot.castShadow = castShadow;
  baseFoot.receiveShadow = true;
  rootGroup.add(baseFoot);

  // Vertical support stanchion connecting y = 0.16 up to headY - 0.48
  const strutHeight = Math.max(0.2, headY - 0.62);
  const strutGeo = new THREE.CylinderGeometry(0.042, 0.058, strutHeight, 12);
  const strutMesh = new THREE.Mesh(strutGeo, palette.strutMat);
  strutMesh.position.y = 0.16 + strutHeight / 2;
  strutMesh.castShadow = castShadow;
  rootGroup.add(strutMesh);

  // Head assembly at elevation headY
  const headGroup = new THREE.Group();
  headGroup.position.set(0, headY, 0);
  rootGroup.add(headGroup);

  // Part 1 (Primary Silhouette): Optical Glass Soma Vessel
  const somaMesh = new THREE.Mesh(geos.somaGlassGeo, palette.glassSomaMat);
  somaMesh.renderOrder = 3;
  headGroup.add(somaMesh);

  // Part 2 (Primary Core): Faceted Inner Activation Nucleus (individual material for live activation color/emissive)
  const nucleusMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4,
    emissive: 0x0891b2,
    emissiveIntensity: 0.8,
    roughness: 0.25,
    metalness: 0.3,
  });
  const nucleusMesh = new THREE.Mesh(geos.nucleusGeo, nucleusMat);
  headGroup.add(nucleusMesh);

  // Part 3 (Secondary Read): Anisotropic Brushed Gimbal Ring
  const gimbalMesh = new THREE.Mesh(geos.gimbalGeo, palette.gimbalMat);
  gimbalMesh.rotation.x = Math.PI / 2;
  gimbalMesh.castShadow = castShadow;
  headGroup.add(gimbalMesh);

  // Part 4 (Secondary Read): Top & Bottom Polar Collector Caps
  const topCap = new THREE.Mesh(geos.polarCapGeo, palette.hullMat);
  topCap.position.y = 0.42;
  topCap.castShadow = castShadow;
  headGroup.add(topCap);

  const bottomCap = new THREE.Mesh(geos.polarCapGeo, palette.hullMat);
  bottomCap.position.y = -0.42;
  bottomCap.rotation.z = Math.PI;
  bottomCap.castShadow = castShadow;
  headGroup.add(bottomCap);

  // Part 5 (Secondary Read): Bias Threshold Calibration Ring
  const biasRingMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0284c7,
    emissiveIntensity: 0.35,
    roughness: 0.3,
    metalness: 0.8,
  });
  const biasRingMesh = new THREE.Mesh(geos.biasRingGeo, biasRingMat);
  biasRingMesh.rotation.y = Math.PI / 4;
  headGroup.add(biasRingMesh);

  // Part 6 (Accent Detail): Dendrite Input Manifold Collar (-X side)
  if (layerIndex > 0) {
    const inputCollar = new THREE.Mesh(
      geos.terminalCollarGeo,
      palette.hullMat,
    );
    inputCollar.position.set(-0.46, 0, 0);
    headGroup.add(inputCollar);
  }

  // Part 7 (Accent Detail): Axon Output Emitter Nozzle (+X side)
  const outputNozzle = new THREE.Mesh(
    geos.axonNozzleGeo,
    palette.accentTrimMat,
  );
  outputNozzle.position.set(0.42, 0, 0);
  headGroup.add(outputNozzle);

  // Part 8 (Accent Detail): 4 Radial Beveled Heat-Sink Cooling Fins
  for (let f = 0; f < 4; f++) {
    const angle = (f * Math.PI) / 2 + Math.PI / 4;
    const fin = new THREE.Mesh(geos.coolingFinGeo, palette.gimbalMat);
    fin.position.set(0, 0, 0);
    fin.rotation.y = angle;
    fin.translateX(0.44);
    headGroup.add(fin);
  }

  // Selection highlight halo (visible when hovered or selected)
  const selectionRingMat = new THREE.MeshBasicMaterial({
    color: 0x22d3ee,
    transparent: true,
    opacity: 0.0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const selectionRingMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.64, 0.72, 32),
    selectionRingMat,
  );
  selectionRingMesh.rotation.x = -Math.PI / 2;
  selectionRingMesh.position.y = -0.48;
  headGroup.add(selectionRingMesh);

  // Dedicated Raycast Proxy Hitbox on Layer 1
  const hitboxMesh = new THREE.Mesh(
    geos.hitboxGeo,
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  hitboxMesh.layers.set(1);
  hitboxMesh.userData = {
    type: 'neuron',
    layerIndex,
    neuronIndex,
  };
  headGroup.add(hitboxMesh);

  return {
    rootGroup,
    headGroup,
    nucleusMesh,
    nucleusMat,
    biasRingMesh,
    biasRingMat,
    selectionRingMesh,
    selectionRingMat,
    hitboxMesh,
    layerIndex,
    neuronIndex,
    basePos: new THREE.Vector3(worldX, 0, worldZ),
    headY,
  };
}
