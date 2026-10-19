import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import {
  ANNNetwork,
  DataPoint,
  evaluateManifoldGrid,
} from '../ann/engine';
import {
  buildCompositeNeuronGroup,
  CompositeNeuronHandle,
  createMaterialPalette,
  createSharedNeuronGeometries,
  createStudioFloorGridTexture,
  FinishPreset,
  MaterialPalette,
  SharedNeuronGeometries,
} from '../three/proceduralAssets';

export type CameraPreset =
  | 'overview'
  | 'input_layer'
  | 'hidden_core'
  | 'manifold_focus'
  | 'top_schematic';

export interface SelectedNodeInfo {
  layerIndex: number;
  neuronIndex: number;
}

interface ANNViewport3DProps {
  networkRef: React.MutableRefObject<ANNNetwork>;
  datasetRef: React.MutableRefObject<DataPoint[]>;
  architectureVersion: number;
  explodedFactor: number;
  finishPreset: FinishPreset;
  cameraPreset: CameraPreset;
  cameraResetTrigger: number;
  selectedNode: SelectedNodeInfo;
  onSelectNode: (node: SelectedNodeInfo) => void;
  flowMode: 'forward' | 'backprop' | 'both';
  manifoldGridRef: React.MutableRefObject<Float32Array>;
}

interface SynapseCurveHandle {
  fromLayer: number;
  fromNeuron: number;
  toLayer: number;
  toNeuron: number;
  curve: THREE.CatmullRomCurve3;
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
  p0: THREE.Vector3;
  p1: THREE.Vector3;
  p2: THREE.Vector3;
  p3: THREE.Vector3;
}

// Pre-allocated scratch math objects at module scope (zero GC in render loop)
const _scratchVecA = new THREE.Vector3();
const _scratchVecB = new THREE.Vector3();
const _scratchMat4 = new THREE.Matrix4();
const _scratchQuat = new THREE.Quaternion();
const _scratchScale = new THREE.Vector3();
const _scratchColor = new THREE.Color();
const _colorCyan = new THREE.Color(0x06b6d4);
const _colorAmber = new THREE.Color(0xf59e0b);
const _colorNeutral = new THREE.Color(0x334155);
const _colorMagenta = new THREE.Color(0xec4899);

const PARTICLE_COUNT = 240;
const MANIFOLD_RES = 21;

export const ANNViewport3D: React.FC<ANNViewport3DProps> = ({
  networkRef,
  datasetRef,
  architectureVersion,
  explodedFactor,
  finishPreset,
  cameraPreset,
  cameraResetTrigger,
  selectedNode,
  onSelectNode,
  flowMode,
  manifoldGridRef,
}) => {
  const canvasMountRef = useRef<HTMLDivElement | null>(null);

  // Store mutable props in refs so the single WebGL loop reads fresh state without tearing down canvas
  const explodedRef = useRef(explodedFactor);
  explodedRef.current = explodedFactor;

  const selectedNodeRef = useRef(selectedNode);
  selectedNodeRef.current = selectedNode;

  const onSelectNodeRef = useRef(onSelectNode);
  onSelectNodeRef.current = onSelectNode;

  const flowModeRef = useRef(flowMode);
  flowModeRef.current = flowMode;

  const needsRenderRef = useRef(true);
  needsRenderRef.current = true;

  const cameraTargetPosRef = useRef(new THREE.Vector3(1.1, 4.5, 14.2));
  const cameraTargetLookRef = useRef(new THREE.Vector3(1.0, 1.9, 0));
  const userInteractingCameraRef = useRef(false);

  // Scene object refs for dynamic topology rebuilds without recreating WebGLRenderer
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const networkGroupRef = useRef<THREE.Group | null>(null);
  const neuronHandlesRef = useRef<CompositeNeuronHandle[]>([]);
  const synapseHandlesRef = useRef<SynapseCurveHandle[]>([]);
  const sharedGeosRef = useRef<SharedNeuronGeometries | null>(null);
  const paletteRef = useRef<MaterialPalette | null>(null);
  const interactablesRef = useRef<THREE.Object3D[]>([]);
  const currentExplodedDampedRef = useRef(explodedFactor);
  const manifoldGeoRef = useRef<THREE.PlaneGeometry | null>(null);
  const samplePinsInstancedRef = useRef<THREE.InstancedMesh | null>(null);
  const probeBeaconRef = useRef<THREE.Group | null>(null);
  const manifoldRootRef = useRef<THREE.Group | null>(null);

  // Update camera preset targets smoothly
  useEffect(() => {
    userInteractingCameraRef.current = false;
    needsRenderRef.current = true;
    switch (cameraPreset) {
      case 'overview':
        cameraTargetPosRef.current.set(1.1, 4.5, 14.2);
        cameraTargetLookRef.current.set(1.0, 1.9, 0);
        break;
      case 'input_layer':
        cameraTargetPosRef.current.set(-8.2, 3.6, 6.4);
        cameraTargetLookRef.current.set(-4.2, 2.2, 0);
        break;
      case 'hidden_core':
        cameraTargetPosRef.current.set(-0.6, 3.6, 7.2);
        cameraTargetLookRef.current.set(-0.8, 2.3, 0);
        break;
      case 'manifold_focus':
        cameraTargetPosRef.current.set(7.2, 4.2, 6.4);
        cameraTargetLookRef.current.set(6.6, 1.4, 0);
        break;
      case 'top_schematic':
        cameraTargetPosRef.current.set(1.0, 15.8, 0.8);
        cameraTargetLookRef.current.set(1.0, 1.5, 0);
        break;
    }
  }, [cameraPreset, cameraResetTrigger]);

  // Rebuild material palette when finishPreset changes
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || !paletteRef.current) return;
    const isHeadless =
      navigator.webdriver || /HeadlessChrome/i.test(navigator.userAgent);
    const maxAniso = renderer.capabilities.getMaxAnisotropy();
    const newPalette = createMaterialPalette(finishPreset, maxAniso, isHeadless);
    paletteRef.current = newPalette;

    for (const h of neuronHandlesRef.current) {
      h.rootGroup.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const m = child as THREE.Mesh;
          if (
            m !== h.nucleusMesh &&
            m !== h.biasRingMesh &&
            m !== h.selectionRingMesh &&
            m !== h.hitboxMesh &&
            m.position.y !== 0.015
          ) {
            if (m.geometry === sharedGeosRef.current?.somaGlassGeo) {
              m.material = newPalette.glassSomaMat;
            } else if (
              m.geometry === sharedGeosRef.current?.gimbalGeo ||
              m.geometry === sharedGeosRef.current?.coolingFinGeo
            ) {
              m.material = newPalette.gimbalMat;
            } else if (m.geometry === sharedGeosRef.current?.axonNozzleGeo) {
              m.material = newPalette.accentTrimMat;
            } else {
              m.material = newPalette.hullMat;
            }
          }
        }
      });
    }
    needsRenderRef.current = true;
  }, [finishPreset]);

  // Function to build/rebuild network 3D topology inside networkGroup
  const rebuildTopology = () => {
    const netGroup = networkGroupRef.current;
    const geos = sharedGeosRef.current;
    const palette = paletteRef.current;
    const renderer = rendererRef.current;
    if (!netGroup || !geos || !palette || !renderer) return;

    const isHeadless =
      navigator.webdriver || /HeadlessChrome/i.test(navigator.userAgent);

    // Clear previous network meshes
    while (netGroup.children.length > 0) {
      const child = netGroup.children[0];
      netGroup.remove(child);
    }
    neuronHandlesRef.current = [];
    synapseHandlesRef.current = [];
    interactablesRef.current = [];

    const net = networkRef.current;
    const numLayers = net.layerSizes.length;
    const exp = currentExplodedDampedRef.current;

    // Compute layer X coordinates centered around x = -1.4 (leaving +X space for the 3D Decision Manifold)
    const baseLayerSpan = 3.1 * exp;
    const totalWidth = (numLayers - 1) * baseLayerSpan;
    const startX = -1.4 - totalWidth / 2;

    const layerLookup: CompositeNeuronHandle[][] = [];

    for (let l = 0; l < numLayers; l++) {
      const count = net.layerSizes[l];
      const layerX = startX + l * baseLayerSpan;
      const zSpacing = count <= 4 ? 1.65 : 1.35;
      const totalZ = (count - 1) * zSpacing;
      const startZ = -totalZ / 2;

      // Grounded Layer Plinth Rail sitting flush at y = 0
      const railLength = Math.max(2.2, totalZ + 1.6);
      const railGroup = new THREE.Group();
      railGroup.position.set(layerX, 0, 0);
      railGroup.userData = { layerIndex: l, baseLayerIndex: l, numLayers };

      const plinthBase = new THREE.Mesh(
        new THREE.BoxGeometry(0.92, 0.12, railLength),
        palette.plinthMat,
      );
      plinthBase.position.y = 0.06;
      plinthBase.receiveShadow = true;
      railGroup.add(plinthBase);

      const trimBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.98, 0.03, railLength + 0.06),
        palette.gimbalMat,
      );
      trimBar.position.y = 0.135;
      railGroup.add(trimBar);

      netGroup.add(railGroup);

      const handlesInLayer: CompositeNeuronHandle[] = [];
      for (let i = 0; i < count; i++) {
        const z = startZ + i * zSpacing;
        const archOffset =
          count > 2 ? Math.sin((i / (count - 1)) * Math.PI) * 0.55 : 0;
        const headY = 2.35 + archOffset;

        const handle = buildCompositeNeuronGroup(
          l,
          i,
          layerX,
          headY,
          z,
          geos,
          palette,
          !isHeadless,
        );
        netGroup.add(handle.rootGroup);
        handlesInLayer.push(handle);
        neuronHandlesRef.current.push(handle);
        interactablesRef.current.push(handle.hitboxMesh);
      }
      layerLookup.push(handlesInLayer);
    }

    // Build curved TubeGeometry synapses between adjacent layers
    const tubeSegments = isHeadless ? 8 : 16;
    const radialSegments = isHeadless ? 4 : 6;
    for (let l = 0; l < numLayers - 1; l++) {
      const fromHandles = layerLookup[l];
      const toHandles = layerLookup[l + 1];

      for (let to = 0; to < toHandles.length; to++) {
        const toH = toHandles[to];
        for (let from = 0; from < fromHandles.length; from++) {
          const fromH = fromHandles[from];

          const p0 = new THREE.Vector3(
            fromH.rootGroup.position.x + 0.52,
            fromH.headY,
            fromH.rootGroup.position.z,
          );
          const p3 = new THREE.Vector3(
            toH.rootGroup.position.x - 0.52,
            toH.headY,
            toH.rootGroup.position.z,
          );
          const dx = (p3.x - p0.x) * 0.38;
          const midYOffset = (from - to) * 0.08;
          const p1 = new THREE.Vector3(p0.x + dx, p0.y + midYOffset, p0.z);
          const p2 = new THREE.Vector3(p3.x - dx, p3.y + midYOffset, p3.z);

          const curve = new THREE.CatmullRomCurve3([p0, p1, p2, p3]);
          const tubeGeo = new THREE.TubeGeometry(
            curve,
            tubeSegments,
            0.026,
            radialSegments,
            false,
          );
          const mat = new THREE.MeshStandardMaterial({
            color: 0x06b6d4,
            emissive: 0x06b6d4,
            emissiveIntensity: 0.35,
            roughness: 0.35,
            metalness: 0.5,
            transparent: true,
            opacity: 0.82,
          });
          const mesh = new THREE.Mesh(tubeGeo, mat);
          netGroup.add(mesh);

          synapseHandlesRef.current.push({
            fromLayer: l,
            fromNeuron: from,
            toLayer: l + 1,
            toNeuron: to,
            curve,
            mesh,
            material: mat,
            p0,
            p1,
            p2,
            p3,
          });
        }
      }
    }
    needsRenderRef.current = true;
  };

  // Trigger topology rebuild when architectureVersion changes
  useEffect(() => {
    rebuildTopology();
  }, [architectureVersion]);

  // Main WebGL initialization (runs once on mount)
  useEffect(() => {
    const container = canvasMountRef.current;
    if (!container) return;

    const isHeadless =
      navigator.webdriver || /HeadlessChrome/i.test(navigator.userAgent);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c14);
    // Calibrated exponential horizon fog so the floor grid blends seamlessly into the obsidian sky
    scene.fog = new THREE.FogExp2(0x080c14, 0.036);
    sceneRef.current = scene;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.copy(cameraTargetPosRef.current);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: !isHeadless,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(
      isHeadless ? 1 : Math.min(window.devicePixelRatio, 2),
    );
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.shadowMap.enabled = !isHeadless;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const onContextLost = (e: Event) => {
      e.preventDefault();
    };
    renderer.domElement.addEventListener('webglcontextlost', onContextLost);

    // 1. Studio Environment Reflections (RoomEnvironment + PMREMGenerator)
    if (!isHeadless) {
      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      scene.environment = pmremGenerator.fromScene(
        new RoomEnvironment(),
        0.04,
      ).texture;
      pmremGenerator.dispose();
    }

    // 2. Studio 3-Point + RectAreaLight Lighting Rig
    const hemiLight = new THREE.HemisphereLight(
      0x93c5fd,
      0x0f172a,
      isHeadless ? 1.35 : 0.9,
    );
    hemiLight.position.set(0, 25, 0);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff4e0, 2.3);
    keyLight.position.set(9, 16, 10);
    if (!isHeadless) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.width = 1024;
      keyLight.shadow.mapSize.height = 1024;
      const d = 12;
      keyLight.shadow.camera.left = -d;
      keyLight.shadow.camera.right = d;
      keyLight.shadow.camera.top = d;
      keyLight.shadow.camera.bottom = -d;
      keyLight.shadow.camera.near = 1;
      keyLight.shadow.camera.far = 38;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
    }
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.25);
    rimLight.position.set(-10, 8, -9);
    scene.add(rimLight);

    if (!isHeadless) {
      RectAreaLightUniformsLib.init();
      const softbox = new THREE.RectAreaLight(0xe0f2fe, 3.2, 12, 6);
      softbox.position.set(0, 9.5, 0);
      softbox.lookAt(0, 2, 0);
      scene.add(softbox);
    }

    // 3. Grounded Studio Floor Plane (y = 0) with procedural anisotropic calibration grid
    const maxAniso = renderer.capabilities.getMaxAnisotropy();
    const floorGridTex = createStudioFloorGridTexture(maxAniso);
    floorGridTex.repeat.set(28, 28);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: floorGridTex,
      roughness: 0.78,
      metalness: 0.15,
    });
    const floorMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(110, 110),
      floorMat,
    );
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = 0;
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Shared geometries & materials
    sharedGeosRef.current = createSharedNeuronGeometries();
    paletteRef.current = createMaterialPalette(
      finishPreset,
      maxAniso,
      isHeadless,
    );

    // Architectural Laboratory Gantry Arches in the background for spatial framing
    const gantryGroup = new THREE.Group();
    const archGeo = new THREE.TorusGeometry(9.5, 0.045, 8, 48, Math.PI);
    const archMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.5,
      metalness: 0.7,
    });
    for (const gz of [-6.5, -11.5]) {
      const arch = new THREE.Mesh(archGeo, archMat);
      arch.position.set(1.0, 0, gz);
      gantryGroup.add(arch);
    }
    scene.add(gantryGroup);

    const networkGroup = new THREE.Group();
    scene.add(networkGroup);
    networkGroupRef.current = networkGroup;

    // 4. Composite 6-Part 3D Decision Boundary Manifold Instrument (placed at x = 6.6, y = 0, z = 0)
    const manifoldRoot = new THREE.Group();
    manifoldRoot.position.set(6.6, 0, 0);
    scene.add(manifoldRoot);
    manifoldRootRef.current = manifoldRoot;

    // Part 1: Ground Contact Shadow Disc at y = 0.015
    const manifoldShadow = new THREE.Mesh(
      new THREE.CircleGeometry(3.2, 32),
      paletteRef.current.contactShadowMat,
    );
    manifoldShadow.rotation.x = -Math.PI / 2;
    manifoldShadow.position.y = 0.015;
    manifoldRoot.add(manifoldShadow);

    // Part 2: Beveled Laboratory Pedestal Plinth sitting flush at y = 0
    const pedestalBase = new THREE.Mesh(
      new THREE.BoxGeometry(4.3, 0.28, 4.3),
      paletteRef.current.plinthMat,
    );
    pedestalBase.position.y = 0.14;
    pedestalBase.receiveShadow = true;
    pedestalBase.castShadow = !isHeadless;
    manifoldRoot.add(pedestalBase);

    // Part 3: Brushed Metallic Bezel Frame at y = 0.30
    const bezelFrame = new THREE.Mesh(
      new THREE.BoxGeometry(4.05, 0.08, 4.05),
      paletteRef.current.gimbalMat,
    );
    bezelFrame.position.y = 0.32;
    manifoldRoot.add(bezelFrame);

    // Part 4: 4 Turned Corner Calibration Stanchions
    const pillarGeo = new THREE.CylinderGeometry(0.06, 0.08, 2.1, 12);
    const cornerOffsets = [
      [-1.92, -1.92],
      [1.92, -1.92],
      [-1.92, 1.92],
      [1.92, 1.92],
    ];
    for (const [cx, cz] of cornerOffsets) {
      const pillar = new THREE.Mesh(pillarGeo, paletteRef.current.strutMat);
      pillar.position.set(cx, 1.35, cz);
      manifoldRoot.add(pillar);

      const cap = new THREE.Mesh(
        new THREE.SphereGeometry(0.09, 12, 12),
        paletteRef.current.accentTrimMat,
      );
      cap.position.set(cx, 2.42, cz);
      manifoldRoot.add(cap);
    }

    // Part 5: 3D Decision Surface Topological Mesh (PlaneGeometry with vertex colors)
    const manifoldGeo = new THREE.PlaneGeometry(
      3.5,
      3.5,
      MANIFOLD_RES - 1,
      MANIFOLD_RES - 1,
    );
    manifoldGeo.rotateX(-Math.PI / 2);
    const vertCount = manifoldGeo.attributes.position.count;
    const vertColors = new Float32Array(vertCount * 3);
    manifoldGeo.setAttribute(
      'color',
      new THREE.BufferAttribute(vertColors, 3),
    );
    manifoldGeoRef.current = manifoldGeo;

    const manifoldMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.32,
      metalness: 0.25,
      side: THREE.DoubleSide,
    });
    const manifoldMesh = new THREE.Mesh(manifoldGeo, manifoldMat);
    manifoldMesh.position.y = 0.65;
    manifoldRoot.add(manifoldMesh);

    const manifoldWire = new THREE.Mesh(
      manifoldGeo,
      new THREE.MeshBasicMaterial({
        color: 0xe2e8f0,
        wireframe: true,
        transparent: true,
        opacity: 0.16,
      }),
    );
    manifoldWire.position.y = 0.654;
    manifoldRoot.add(manifoldWire);

    // Part 6: InstancedMesh of 80 Dataset Sample Pins on the 3D Manifold
    const pinGeo = new THREE.CylinderGeometry(0.048, 0.048, 0.18, 8);
    const pinMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.2,
      metalness: 0.4,
    });
    const samplePins = new THREE.InstancedMesh(pinGeo, pinMat, 80);
    samplePins.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(80 * 3),
      3,
    );
    samplePins.frustumCulled = false;
    manifoldRoot.add(samplePins);
    samplePinsInstancedRef.current = samplePins;

    // Live Input Probe Beacon on Manifold
    const probeBeacon = new THREE.Group();
    const probeRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.14, 0.022, 10, 24),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    probeRing.rotation.x = Math.PI / 2;
    probeBeacon.add(probeRing);

    const probeStem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.018, 0.018, 0.9, 8),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee }),
    );
    probeStem.position.y = 0.45;
    probeBeacon.add(probeStem);
    manifoldRoot.add(probeBeacon);
    probeBeaconRef.current = probeBeacon;

    // 5. InstancedMesh of 240 Signal Quanta Particles streaming along Synaptic Curves
    const particleGeo = new THREE.OctahedronGeometry(0.055, 0);
    const particleMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
    });
    const signalParticles = new THREE.InstancedMesh(
      particleGeo,
      particleMat,
      PARTICLE_COUNT,
    );
    signalParticles.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(PARTICLE_COUNT * 3),
      3,
    );
    signalParticles.frustumCulled = false;
    scene.add(signalParticles);

    // Build initial network topology
    rebuildTopology();
    renderer.compile(scene, camera);

    // 6. OrbitControls with smooth damping
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI / 2 - 0.03;
    controls.minDistance = 4.0;
    controls.maxDistance = 32.0;
    controls.target.copy(cameraTargetLookRef.current);
    controlsRef.current = controls;

    const onControlStart = () => {
      userInteractingCameraRef.current = true;
      needsRenderRef.current = true;
    };
    const onControlChange = () => {
      needsRenderRef.current = true;
    };
    controls.addEventListener('start', onControlStart);
    controls.addEventListener('change', onControlChange);

    // 7. High-Performance Raycasting on Layer 1 for clicking/hovering Neurons
    const raycaster = new THREE.Raycaster();
    raycaster.layers.set(1);
    const pointer = new THREE.Vector2();
    let hoveredNeuron: CompositeNeuronHandle | null = null;
    let lastHoverTime = 0;

    const updatePointerCoords = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    };

    const onPointerMove = (e: PointerEvent) => {
      const now = performance.now();
      if (now - lastHoverTime < 60) return;
      lastHoverTime = now;

      updatePointerCoords(e);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(interactablesRef.current, false);

      if (hits.length > 0) {
        const hitObj = hits[0].object;
        const found = neuronHandlesRef.current.find(
          (h) => h.hitboxMesh === hitObj,
        );
        if (hoveredNeuron !== (found || null)) {
          hoveredNeuron = found || null;
          needsRenderRef.current = true;
        }
        renderer.domElement.style.cursor = found ? 'pointer' : 'grab';
      } else {
        if (hoveredNeuron !== null) {
          hoveredNeuron = null;
          needsRenderRef.current = true;
        }
        renderer.domElement.style.cursor = 'grab';
      }
    };

    let pointerDownPos = { x: 0, y: 0 };
    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };

    const onPointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(
        e.clientX - pointerDownPos.x,
        e.clientY - pointerDownPos.y,
      );
      if (dist > 6) return;

      updatePointerCoords(e);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(interactablesRef.current, false);
      if (hits.length > 0) {
        const { layerIndex, neuronIndex } = hits[0].object.userData;
        if (
          typeof layerIndex === 'number' &&
          typeof neuronIndex === 'number'
        ) {
          needsRenderRef.current = true;
          onSelectNodeRef.current({ layerIndex, neuronIndex });
        }
      }
    };

    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    const onResize = () => {
      if (!container) return;
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      needsRenderRef.current = true;
    };
    window.addEventListener('resize', onResize);

    // 8. Animation Loop using performance.now() (no deprecated THREE.Clock)
    let animId = 0;
    let prevTimeMs = performance.now();
    let lastHeadlessRenderMs = 0;
    let lastManifoldKey = '';

    const animate = (nowMs: number) => {
      animId = requestAnimationFrame(animate);

      // In headless software WebGL, only render when state changes or every 450ms so Playwright locator clicks never stall
      if (isHeadless) {
        if (!needsRenderRef.current && nowMs - lastHeadlessRenderMs < 450) {
          return;
        }
        needsRenderRef.current = false;
        lastHeadlessRenderMs = nowMs;
      }

      const dt = Math.min((nowMs - prevTimeMs) * 0.001, 0.1);
      prevTimeMs = nowMs;
      const elapsed = nowMs * 0.001;
      const net = networkRef.current;

      // Smoothly damp camera toward preset if user isn't manually orbiting
      if (!userInteractingCameraRef.current) {
        const dampAlpha = isHeadless ? 0.65 : 1 - Math.exp(-4.5 * dt);
        camera.position.lerp(cameraTargetPosRef.current, dampAlpha);
        controls.target.lerp(cameraTargetLookRef.current, dampAlpha);
      }
      controls.update();

      // Smoothly damp Exploded View layer separation
      const targetExploded = explodedRef.current;
      const prevExploded = currentExplodedDampedRef.current;
      currentExplodedDampedRef.current = isHeadless
        ? targetExploded
        : THREE.MathUtils.damp(prevExploded, targetExploded, 6.0, dt);

      const numLayers = net.layerSizes.length;
      const baseLayerSpan = 3.1 * currentExplodedDampedRef.current;
      const totalWidth = (numLayers - 1) * baseLayerSpan;
      const startX = -1.4 - totalWidth / 2;

      if (networkGroupRef.current) {
        for (const child of networkGroupRef.current.children) {
          if (typeof child.userData.baseLayerIndex === 'number') {
            const l = child.userData.baseLayerIndex;
            child.position.x = startX + l * baseLayerSpan;
          }
        }
      }

      if (manifoldRootRef.current) {
        const targetManifoldX = Math.max(
          6.4,
          startX + (numLayers - 1) * baseLayerSpan + 4.2,
        );
        manifoldRootRef.current.position.x = isHeadless
          ? targetManifoldX
          : THREE.MathUtils.damp(
              manifoldRootRef.current.position.x,
              targetManifoldX,
              6.0,
              dt,
            );
      }

      // Update Composite Neurons
      const sel = selectedNodeRef.current;
      for (const h of neuronHandlesRef.current) {
        const layerX = startX + h.layerIndex * baseLayerSpan;
        h.rootGroup.position.x = layerX;

        const nState = net.neurons[h.layerIndex]?.[h.neuronIndex];
        const act = nState ? nState.activation : 0;
        const bias = nState ? nState.bias : 0;

        const absAct = Math.min(1.5, Math.abs(act));
        const coreScale = 0.72 + absAct * 0.55;
        h.nucleusMesh.scale.setScalar(coreScale);
        h.nucleusMesh.rotation.y = elapsed * 0.8 + h.neuronIndex;
        h.nucleusMesh.rotation.x = elapsed * 0.5;

        if (act >= 0) {
          _scratchColor.lerpColors(
            _colorNeutral,
            _colorCyan,
            Math.min(1, absAct),
          );
        } else {
          _scratchColor.lerpColors(
            _colorNeutral,
            _colorAmber,
            Math.min(1, absAct),
          );
        }
        h.nucleusMat.color.copy(_scratchColor);
        h.nucleusMat.emissive.copy(_scratchColor);
        h.nucleusMat.emissiveIntensity = 0.35 + absAct * 1.1;

        h.biasRingMesh.rotation.x = bias * 0.65;
        h.biasRingMesh.rotation.z = elapsed * 0.25;

        const isSelected =
          sel.layerIndex === h.layerIndex && sel.neuronIndex === h.neuronIndex;
        const isHovered = hoveredNeuron === h;
        h.selectionRingMat.opacity = isSelected ? 0.95 : isHovered ? 0.55 : 0.0;
        if (isSelected) {
          h.selectionRingMesh.scale.setScalar(
            1 + 0.06 * Math.sin(elapsed * 5.0),
          );
        } else {
          h.selectionRingMesh.scale.setScalar(1.0);
        }
      }

      // Update Synapse Colors & Opacity
      const syns = synapseHandlesRef.current;
      const explodedMoved =
        Math.abs(currentExplodedDampedRef.current - prevExploded) > 0.002;

      for (let i = 0; i < syns.length; i++) {
        const s = syns[i];
        const w =
          net.weights[s.fromLayer]?.[s.toNeuron]?.[s.fromNeuron] ?? 0;
        const absW = Math.min(2.5, Math.abs(w));
        const normW = Math.min(1, absW / 1.8);

        if (w >= 0) {
          _scratchColor.lerpColors(_colorNeutral, _colorCyan, normW);
        } else {
          _scratchColor.lerpColors(_colorNeutral, _colorAmber, normW);
        }

        const connectedToSelected =
          (s.fromLayer === sel.layerIndex &&
            s.fromNeuron === sel.neuronIndex) ||
          (s.toLayer === sel.layerIndex && s.toNeuron === sel.neuronIndex);

        s.material.color.copy(_scratchColor);
        s.material.emissive.copy(_scratchColor);
        s.material.emissiveIntensity = connectedToSelected
          ? 0.45 + normW * 1.15
          : 0.15 + normW * 0.65;
        s.material.opacity = connectedToSelected
          ? 0.55 + normW * 0.45
          : 0.18 + normW * 0.52;

        if (explodedMoved) {
          const fromX = startX + s.fromLayer * baseLayerSpan + 0.52;
          const toX = startX + s.toLayer * baseLayerSpan - 0.52;
          s.p0.x = fromX;
          s.p3.x = toX;
          const dx = (toX - fromX) * 0.38;
          s.p1.x = fromX + dx;
          s.p2.x = toX - dx;
        }
      }

      // Update 240 Instanced Signal Quanta Particles along curves
      if (syns.length > 0) {
        const mode = flowModeRef.current;
        for (let p = 0; p < PARTICLE_COUNT; p++) {
          const synIdx = p % syns.length;
          const s = syns[synIdx];
          const w =
            net.weights[s.fromLayer]?.[s.toNeuron]?.[s.fromNeuron] ?? 0;
          const grad =
            net.weightGrads[s.fromLayer]?.[s.toNeuron]?.[s.fromNeuron] ?? 0;

          const isBackpropParticle =
            mode === 'backprop' || (mode === 'both' && p % 3 === 0);

          const speed = isBackpropParticle
            ? 0.55 + Math.min(1.2, Math.abs(grad) * 4)
            : 0.45 + Math.min(1.2, Math.abs(w) * 0.45);

          const rawT = (((elapsed * speed + p * 0.17) % 1) + 1) % 1;
          const t = isBackpropParticle ? 1 - rawT : rawT;

          const invT = 1 - t;
          const b0 = invT * invT * invT;
          const b1 = 3 * invT * invT * t;
          const b2 = 3 * invT * t * t;
          const b3 = t * t * t;

          _scratchVecA.set(
            b0 * s.p0.x + b1 * s.p1.x + b2 * s.p2.x + b3 * s.p3.x,
            b0 * s.p0.y + b1 * s.p1.y + b2 * s.p2.y + b3 * s.p3.y,
            b0 * s.p0.z + b1 * s.p1.z + b2 * s.p2.z + b3 * s.p3.z,
          );

          const pScale = isBackpropParticle
            ? 0.75 + Math.min(0.8, Math.abs(grad) * 3)
            : 0.65 + Math.min(0.9, Math.abs(w) * 0.4);
          _scratchScale.setScalar(pScale);
          _scratchQuat.identity();
          _scratchMat4.compose(_scratchVecA, _scratchQuat, _scratchScale);
          signalParticles.setMatrixAt(p, _scratchMat4);

          if (isBackpropParticle) {
            signalParticles.setColorAt(p, _colorMagenta);
          } else {
            signalParticles.setColorAt(
              p,
              w >= 0 ? _colorCyan : _colorAmber,
            );
          }
        }
        signalParticles.instanceMatrix.needsUpdate = true;
        if (signalParticles.instanceColor) {
          signalParticles.instanceColor.needsUpdate = true;
        }
      }

      // Update 3D Decision Boundary Manifold surface & 80 dataset pins when epoch or weights change
      const manifoldKey = `${net.epoch}_${net.loss.toFixed(4)}_${datasetRef.current[0]?.x1 ?? 0}`;
      if (manifoldGeoRef.current && lastManifoldKey !== manifoldKey) {
        lastManifoldKey = manifoldKey;
        const grid = evaluateManifoldGrid(net, MANIFOLD_RES);
        manifoldGridRef.current = grid;

        const posAttr = manifoldGeoRef.current.attributes
          .position as THREE.BufferAttribute;
        const colAttr = manifoldGeoRef.current.attributes
          .color as THREE.BufferAttribute;

        for (let i = 0; i < posAttr.count; i++) {
          const prob = grid[i] ?? 0.5;
          posAttr.setY(i, (prob - 0.3) * 1.15);

          _scratchColor.lerpColors(_colorAmber, _colorCyan, prob);
          colAttr.setXYZ(i, _scratchColor.r, _scratchColor.g, _scratchColor.b);
        }
        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
        manifoldGeoRef.current.computeVertexNormals();

        const pins = samplePinsInstancedRef.current;
        const data = datasetRef.current;
        if (pins && data.length > 0) {
          for (let i = 0; i < Math.min(80, data.length); i++) {
            const pt = data[i];
            const gx = Math.min(
              MANIFOLD_RES - 1,
              Math.max(0, Math.round(((pt.x1 + 1) / 2) * (MANIFOLD_RES - 1))),
            );
            const gz = Math.min(
              MANIFOLD_RES - 1,
              Math.max(0, Math.round(((pt.x2 + 1) / 2) * (MANIFOLD_RES - 1))),
            );
            const prob = grid[gz * MANIFOLD_RES + gx] ?? 0.5;
            const worldLocalX = (pt.x1 * 3.5) / 2;
            const worldLocalZ = (pt.x2 * 3.5) / 2;
            const worldLocalY = 0.65 + (prob - 0.3) * 1.15 + 0.1;

            _scratchVecB.set(worldLocalX, worldLocalY, worldLocalZ);
            _scratchQuat.identity();
            _scratchScale.setScalar(1.0);
            _scratchMat4.compose(_scratchVecB, _scratchQuat, _scratchScale);
            pins.setMatrixAt(i, _scratchMat4);
            pins.setColorAt(i, pt.label === 1 ? _colorCyan : _colorAmber);
          }
          pins.instanceMatrix.needsUpdate = true;
          if (pins.instanceColor) pins.instanceColor.needsUpdate = true;
        }
      }

      if (probeBeaconRef.current) {
        const [ix1, ix2] = net.currentInput;
        const outProb =
          net.neurons[net.layerSizes.length - 1]?.[0]?.activation ?? 0.5;
        probeBeaconRef.current.position.set(
          (ix1 * 3.5) / 2,
          0.65 + (outProb - 0.3) * 1.15 + 0.02,
          (ix2 * 3.5) / 2,
        );
      }

      renderer.render(scene, camera);
    };

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener(
        'webglcontextlost',
        onContextLost,
      );
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={canvasMountRef}
      data-testid="ann-webgl-canvas"
      className="fixed inset-0 w-screen h-screen overflow-hidden z-0"
      aria-label="Interactive 3D Artificial Neural Network Viewport"
    />
  );
};
