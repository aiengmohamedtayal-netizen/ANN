import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivationType,
  ANNNetwork,
  createNetwork,
  DatasetType,
  evaluateDataset,
  evaluateManifoldGrid,
  forwardPass,
  generateDataset,
  trainEpoch,
} from './ann/engine';
import {
  ANNViewport3D,
  CameraPreset,
  SelectedNodeInfo,
} from './components/ANNViewport3D';
import { ControlDockHUD, DockTab } from './components/ControlDockHUD';
import { FinishPreset } from './three/proceduralAssets';

export default function App() {
  // Initial deterministic dataset & pre-warmed network (5 initial epochs so first paint shows meaningful structure)
  const [datasetType, setDatasetType] = useState<DatasetType>('concentric_rings');
  const [activationType, setActivationType] = useState<ActivationType>('tanh');
  const [learningRate, setLearningRate] = useState<number>(0.12);
  const [flowMode, setFlowMode] = useState<'forward' | 'backprop' | 'both'>('both');
  const [explodedFactor, setExplodedFactor] = useState<number>(1.0);
  const [finishPreset, setFinishPreset] = useState<FinishPreset>('titanium_lab');
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('overview');
  const [cameraResetTrigger, setCameraResetTrigger] = useState<number>(0);
  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo>({
    layerIndex: 1,
    neuronIndex: 0,
  });
  const [isDockCollapsed, setIsDockCollapsed] = useState<boolean>(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<DockTab>('training');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [architectureVersion, setArchitectureVersion] = useState<number>(1);
  const [, setHudTick] = useState<number>(0);

  const datasetRef = useRef(generateDataset('concentric_rings', 80, 1337));
  const networkRef = useRef<ANNNetwork>((() => {
    const net = createNetwork([2, 6, 4, 1], 'tanh', 42);
    const data = datasetRef.current;
    for (let i = 0; i < 6; i++) {
      trainEpoch(net, data, 0.12);
    }
    return net;
  })());
  const manifoldGridRef = useRef<Float32Array>(
    evaluateManifoldGrid(networkRef.current, 21),
  );

  const forceHudUpdate = useCallback(() => {
    setHudTick((t) => t + 1);
  }, []);

  // Keyboard shortcut 'H' to collapse/expand bottom control dock
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (e.key === 'h' || e.key === 'H') {
        setIsDockCollapsed((prev) => !prev);
      } else if (e.key === ' ') {
        e.preventDefault();
        setIsRunning((prev) => !prev);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // Continuous training interval (throttled to 10Hz so React HUD updates stay silky-smooth)
  useEffect(() => {
    if (!isRunning) return;
    const timer = window.setInterval(() => {
      trainEpoch(networkRef.current, datasetRef.current, learningRate);
      forceHudUpdate();
    }, 100);
    return () => window.clearInterval(timer);
  }, [isRunning, learningRate, forceHudUpdate]);

  const handleStepEpoch = useCallback(() => {
    trainEpoch(networkRef.current, datasetRef.current, learningRate);
    forceHudUpdate();
  }, [learningRate, forceHudUpdate]);

  const handleResetNetwork = useCallback(() => {
    setIsRunning(false);
    const net = createNetwork(
      networkRef.current.layerSizes,
      activationType,
      Math.floor(42 + (networkRef.current.epoch % 97)),
    );
    const metrics = evaluateDataset(net, datasetRef.current);
    net.loss = metrics.loss;
    net.accuracy = metrics.accuracy;
    net.lossHistory = [metrics.loss];
    net.accuracyHistory = [metrics.accuracy];
    networkRef.current = net;
    forceHudUpdate();
  }, [activationType, forceHudUpdate]);

  const handleChangeDataset = useCallback(
    (newType: DatasetType) => {
      setDatasetType(newType);
      const newData = generateDataset(newType, 80, 1337);
      datasetRef.current = newData;
      const metrics = evaluateDataset(networkRef.current, newData);
      networkRef.current.loss = metrics.loss;
      networkRef.current.accuracy = metrics.accuracy;
      forceHudUpdate();
    },
    [forceHudUpdate],
  );

  const handleChangeActivation = useCallback(
    (newAct: ActivationType) => {
      setActivationType(newAct);
      networkRef.current.activation = newAct;
      forwardPass(networkRef.current, networkRef.current.currentInput);
      const metrics = evaluateDataset(networkRef.current, datasetRef.current);
      networkRef.current.loss = metrics.loss;
      networkRef.current.accuracy = metrics.accuracy;
      forceHudUpdate();
    },
    [forceHudUpdate],
  );

  const handleAddHiddenLayer = useCallback(() => {
    const curr = networkRef.current.layerSizes;
    if (curr.length >= 6) return;
    const nextSizes = [...curr.slice(0, -1), 4, 1];
    const net = createNetwork(nextSizes, activationType, 42);
    const metrics = evaluateDataset(net, datasetRef.current);
    net.loss = metrics.loss;
    net.accuracy = metrics.accuracy;
    net.lossHistory = [metrics.loss];
    networkRef.current = net;
    setArchitectureVersion((v) => v + 1);
    forceHudUpdate();
  }, [activationType, forceHudUpdate]);

  const handleRemoveHiddenLayer = useCallback(() => {
    const curr = networkRef.current.layerSizes;
    if (curr.length <= 3) return;
    const nextSizes = [...curr.slice(0, -2), 1];
    const net = createNetwork(nextSizes, activationType, 42);
    const metrics = evaluateDataset(net, datasetRef.current);
    net.loss = metrics.loss;
    net.accuracy = metrics.accuracy;
    net.lossHistory = [metrics.loss];
    networkRef.current = net;
    setSelectedNode({ layerIndex: 1, neuronIndex: 0 });
    setArchitectureVersion((v) => v + 1);
    forceHudUpdate();
  }, [activationType, forceHudUpdate]);

  const handleChangeLayerNeurons = useCallback(
    (layerIdx: number, delta: number) => {
      const curr = [...networkRef.current.layerSizes];
      if (layerIdx <= 0 || layerIdx >= curr.length - 1) return;
      const nextCount = Math.max(2, Math.min(8, curr[layerIdx] + delta));
      if (nextCount === curr[layerIdx]) return;
      curr[layerIdx] = nextCount;
      const net = createNetwork(curr, activationType, 42);
      const metrics = evaluateDataset(net, datasetRef.current);
      net.loss = metrics.loss;
      net.accuracy = metrics.accuracy;
      net.lossHistory = [metrics.loss];
      networkRef.current = net;
      setSelectedNode({ layerIndex: layerIdx, neuronIndex: 0 });
      setArchitectureVersion((v) => v + 1);
      forceHudUpdate();
    },
    [activationType, forceHudUpdate],
  );

  const handleChangeProbeInput = useCallback(
    (x1: number, x2: number) => {
      forwardPass(networkRef.current, [x1, x2]);
      forceHudUpdate();
    },
    [forceHudUpdate],
  );

  const handleChangeNeuronBias = useCallback(
    (layerIdx: number, neuronIdx: number, newBias: number) => {
      const neuron = networkRef.current.neurons[layerIdx]?.[neuronIdx];
      if (!neuron) return;
      neuron.bias = newBias;
      forwardPass(networkRef.current, networkRef.current.currentInput);
      const metrics = evaluateDataset(networkRef.current, datasetRef.current);
      networkRef.current.loss = metrics.loss;
      networkRef.current.accuracy = metrics.accuracy;
      forceHudUpdate();
    },
    [forceHudUpdate],
  );

  const handleSelectNode = useCallback((node: SelectedNodeInfo) => {
    setSelectedNode(node);
    setIsInspectorOpen(true);
  }, []);

  const handleChangeCameraPreset = useCallback((preset: CameraPreset) => {
    setCameraPreset(preset);
    setCameraResetTrigger((c) => c + 1);
  }, []);

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-[#080c14] select-none">
      {/* Full-Bleed Three.js 3D Viewport */}
      <ANNViewport3D
        networkRef={networkRef}
        datasetRef={datasetRef}
        architectureVersion={architectureVersion}
        explodedFactor={explodedFactor}
        finishPreset={finishPreset}
        cameraPreset={cameraPreset}
        cameraResetTrigger={cameraResetTrigger}
        selectedNode={selectedNode}
        onSelectNode={handleSelectNode}
        flowMode={flowMode}
        manifoldGridRef={manifoldGridRef}
      />

      {/* Ergonomic HUD & Collapsible Bottom Control Dock */}
      <ControlDockHUD
        net={networkRef.current}
        isRunning={isRunning}
        onToggleRun={() => setIsRunning((r) => !r)}
        onStepEpoch={handleStepEpoch}
        onResetNetwork={handleResetNetwork}
        datasetType={datasetType}
        onChangeDataset={handleChangeDataset}
        activationType={activationType}
        onChangeActivation={handleChangeActivation}
        learningRate={learningRate}
        onChangeLearningRate={setLearningRate}
        flowMode={flowMode}
        onChangeFlowMode={setFlowMode}
        explodedFactor={explodedFactor}
        onChangeExplodedFactor={setExplodedFactor}
        finishPreset={finishPreset}
        onChangeFinishPreset={setFinishPreset}
        cameraPreset={cameraPreset}
        onChangeCameraPreset={handleChangeCameraPreset}
        selectedNode={selectedNode}
        onSelectNode={setSelectedNode}
        onAddHiddenLayer={handleAddHiddenLayer}
        onRemoveHiddenLayer={handleRemoveHiddenLayer}
        onChangeLayerNeurons={handleChangeLayerNeurons}
        onChangeProbeInput={handleChangeProbeInput}
        onChangeNeuronBias={handleChangeNeuronBias}
        isDockCollapsed={isDockCollapsed}
        onToggleDockCollapsed={() => setIsDockCollapsed((c) => !c)}
        isInspectorOpen={isInspectorOpen}
        onToggleInspector={() => setIsInspectorOpen((o) => !o)}
        activeTab={activeTab}
        onChangeTab={setActiveTab}
      />
    </main>
  );
}
