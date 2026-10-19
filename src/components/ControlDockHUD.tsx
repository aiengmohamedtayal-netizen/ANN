import React from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Cpu,
  Eye,
  Layers,
  Minus,
  Play,
  Plus,
  RotateCcw,
  SkipForward,
  Sliders,
  Square,
} from 'lucide-react';
import {
  ActivationType,
  ANNNetwork,
  DatasetType,
} from '../ann/engine';
import { FinishPreset } from '../three/proceduralAssets';
import { CameraPreset, SelectedNodeInfo } from './ANNViewport3D';

export type DockTab = 'training' | 'architecture' | 'studio';

interface ControlDockHUDProps {
  net: ANNNetwork;
  isRunning: boolean;
  onToggleRun: () => void;
  onStepEpoch: () => void;
  onResetNetwork: () => void;
  datasetType: DatasetType;
  onChangeDataset: (d: DatasetType) => void;
  activationType: ActivationType;
  onChangeActivation: (a: ActivationType) => void;
  learningRate: number;
  onChangeLearningRate: (lr: number) => void;
  flowMode: 'forward' | 'backprop' | 'both';
  onChangeFlowMode: (m: 'forward' | 'backprop' | 'both') => void;
  explodedFactor: number;
  onChangeExplodedFactor: (val: number) => void;
  finishPreset: FinishPreset;
  onChangeFinishPreset: (f: FinishPreset) => void;
  cameraPreset: CameraPreset;
  onChangeCameraPreset: (c: CameraPreset) => void;
  selectedNode: SelectedNodeInfo;
  onSelectNode: (node: SelectedNodeInfo) => void;
  onAddHiddenLayer: () => void;
  onRemoveHiddenLayer: () => void;
  onChangeLayerNeurons: (layerIdx: number, delta: number) => void;
  onChangeProbeInput: (x1: number, x2: number) => void;
  onChangeNeuronBias: (layerIdx: number, neuronIdx: number, newBias: number) => void;
  isDockCollapsed: boolean;
  onToggleDockCollapsed: () => void;
  isInspectorOpen: boolean;
  onToggleInspector: () => void;
  activeTab: DockTab;
  onChangeTab: (tab: DockTab) => void;
}

const DATASET_LABELS: Record<DatasetType, string> = {
  xor_spiral: 'XOR Quadrants',
  concentric_rings: 'Concentric Rings',
  gaussian_clusters: 'Gaussian Clusters',
  sine_boundary: 'Sine Wave Manifold',
};

const ACTIVATION_LABELS: Record<ActivationType, { name: string; formula: string }> = {
  tanh: { name: 'Tanh', formula: 'tanh(z)' },
  relu: { name: 'ReLU', formula: 'max(0.08z, z)' },
  sigmoid: { name: 'Sigmoid', formula: '1 / (1 + exp(-z))' },
  gelu: { name: 'GELU', formula: '0.5z(1 + tanh(z))' },
};

export const ControlDockHUD: React.FC<ControlDockHUDProps> = ({
  net,
  isRunning,
  onToggleRun,
  onStepEpoch,
  onResetNetwork,
  datasetType,
  onChangeDataset,
  activationType,
  onChangeActivation,
  learningRate,
  onChangeLearningRate,
  flowMode,
  onChangeFlowMode,
  explodedFactor,
  onChangeExplodedFactor,
  finishPreset,
  onChangeFinishPreset,
  cameraPreset,
  onChangeCameraPreset,
  selectedNode,
  onSelectNode,
  onAddHiddenLayer,
  onRemoveHiddenLayer,
  onChangeLayerNeurons,
  onChangeProbeInput,
  onChangeNeuronBias,
  isDockCollapsed,
  onToggleDockCollapsed,
  isInspectorOpen,
  onToggleInspector,
  activeTab,
  onChangeTab,
}) => {
  const topologyString = net.layerSizes.join(' · ');
  const totalParams = net.weights.reduce(
    (acc, layerW, l) =>
      acc +
      layerW.reduce((rAcc, row) => rAcc + row.length, 0) +
      net.layerSizes[l + 1],
    0,
  );

  const selectedLayerIdx = Math.min(
    net.layerSizes.length - 1,
    Math.max(0, selectedNode.layerIndex),
  );
  const selectedNeuronIdx = Math.min(
    (net.layerSizes[selectedLayerIdx] ?? 1) - 1,
    Math.max(0, selectedNode.neuronIndex),
  );
  const activeNeuron =
    net.neurons[selectedLayerIdx]?.[selectedNeuronIdx] ?? net.neurons[0][0];

  const layerRoleLabel =
    selectedLayerIdx === 0
      ? 'Input Feature Node'
      : selectedLayerIdx === net.layerSizes.length - 1
        ? 'Output Logit Node'
        : `Hidden Layer ${selectedLayerIdx} Unit`;

  const sparkPoints = (() => {
    const hist = net.lossHistory;
    if (hist.length < 2) return '0,18 80,18';
    const maxL = Math.max(0.75, ...hist);
    const minL = Math.min(0.01, ...hist);
    const range = Math.max(0.05, maxL - minL);
    return hist
      .map((val, idx) => {
        const x = (idx / (hist.length - 1)) * 80;
        const y = 22 - ((val - minL) / range) * 18;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  })();

  const outputProb =
    net.neurons[net.layerSizes.length - 1]?.[0]?.activation ?? 0.5;

  return (
    <>
      {/* TOP-LEFT MINIMAL SPEC & TELEMETRY BAR (Zero backdrop-filter, unboxed metadata) */}
      <header className="fixed top-3 left-3 right-3 z-20 pointer-events-none flex flex-wrap items-start justify-between gap-2">
        <div className="pointer-events-auto max-w-full flex flex-wrap items-center gap-3 px-3.5 py-2 bg-[#0b101b] border border-slate-800 rounded-lg shadow-lg">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400 shrink-0" aria-hidden="true" />
            <span className="text-sm font-bold tracking-tight text-slate-100 whitespace-nowrap">
              SynapseLab 3D
            </span>
            <span className="text-slate-600" aria-hidden="true">
              ·
            </span>
            <span className="text-xs font-mono text-slate-400 whitespace-nowrap">
              MLP [{topologyString}]
            </span>
            <span className="hidden md:inline text-slate-600" aria-hidden="true">
              ·
            </span>
            <span className="hidden md:inline text-xs font-mono text-slate-400 whitespace-nowrap">
              {totalParams} params
            </span>
          </div>

          <div className="h-4 w-px bg-slate-800 hidden sm:block" />

          {/* Live Quantitative Telemetry Readout */}
          <div className="flex flex-wrap items-center gap-3 text-xs font-mono tabular-nums">
            <div>
              <span className="text-slate-400 mr-1">Epoch</span>
              <span
                data-testid="metric-epoch"
                className="text-slate-100 font-semibold"
              >
                {net.epoch}
              </span>
            </div>
            <span className="text-slate-700" aria-hidden="true">
              /
            </span>
            <div>
              <span className="text-slate-400 mr-1">Loss</span>
              <span
                data-testid="metric-loss"
                className="text-cyan-400 font-semibold"
              >
                {net.loss.toFixed(4)}
              </span>
            </div>
            <span className="text-slate-700" aria-hidden="true">
              /
            </span>
            <div>
              <span className="text-slate-400 mr-1">Acc</span>
              <span
                data-testid="metric-accuracy"
                className="text-emerald-400 font-semibold"
              >
                {(net.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <span className="hidden lg:inline text-slate-700" aria-hidden="true">
              /
            </span>
            <div className="hidden lg:block">
              <span className="text-slate-400 mr-1">P(y=1|x)</span>
              <span
                data-testid="metric-prediction"
                className={`font-semibold ${
                  outputProb >= 0.5 ? 'text-cyan-400' : 'text-amber-400'
                }`}
              >
                {(outputProb * 100).toFixed(1)}%
              </span>
            </div>

            {/* Mini Loss Convergence Sparkline */}
            <svg
              width="72"
              height="22"
              viewBox="0 0 80 24"
              className="hidden md:block bg-slate-950 border border-slate-800/80 rounded px-1"
              aria-label="Loss convergence sparkline"
            >
              <polyline
                fill="none"
                stroke="#06b6d4"
                strokeWidth="1.5"
                points={sparkPoints}
              />
            </svg>
          </div>

          {/* Toggle Node Mathematical Inspector Popover */}
          <button
            type="button"
            onClick={onToggleInspector}
            aria-expanded={isInspectorOpen}
            className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 ${
              isInspectorOpen
                ? 'bg-cyan-950/80 border-cyan-500/60 text-cyan-200'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white hover:border-slate-600'
            }`}
          >
            <Eye className="w-3.5 h-3.5" aria-hidden="true" />
            <span>
              Node L{selectedLayerIdx}:N{selectedNeuronIdx}
            </span>
          </button>
        </div>

        {/* COMPACT NODE MATHEMATICAL INSPECTOR POPOVER (Top-Right, only when open) */}
        {isInspectorOpen && (
          <aside
            aria-label="Selected Neuron Mathematical Inspector"
            className="pointer-events-auto w-full sm:w-80 bg-[#0b101b] border border-slate-800 rounded-lg p-3 shadow-xl text-xs"
          >
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
              <div>
                <div className="font-semibold text-slate-100">
                  Neuron L{selectedLayerIdx} · Unit {selectedNeuronIdx}
                </div>
                <div className="text-slate-400 text-[11px]">{layerRoleLabel}</div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    const nextIdx =
                      (selectedNeuronIdx + 1) %
                      net.layerSizes[selectedLayerIdx];
                    onSelectNode({
                      layerIndex: selectedLayerIdx,
                      neuronIndex: nextIdx,
                    });
                  }}
                  className="px-2 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 rounded text-[11px] whitespace-nowrap"
                >
                  Next Unit
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const nextL =
                      (selectedLayerIdx + 1) % net.layerSizes.length;
                    onSelectNode({ layerIndex: nextL, neuronIndex: 0 });
                  }}
                  className="px-2 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 rounded text-[11px] whitespace-nowrap"
                >
                  Next Layer
                </button>
              </div>
            </div>

            {/* Live Equations & State Readout */}
            <div className="grid grid-cols-3 gap-2 mb-2.5 font-mono tabular-nums bg-slate-950 p-2 rounded border border-slate-800/80">
              <div>
                <div className="text-[10px] text-slate-400">Pre-Act (z)</div>
                <div className="text-slate-200 font-semibold">
                  {activeNeuron.z >= 0 ? '+' : ''}
                  {activeNeuron.z.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-400">Output (a)</div>
                <div
                  data-testid="inspector-activation"
                  className={`font-semibold ${
                    activeNeuron.activation >= 0
                      ? 'text-cyan-400'
                      : 'text-amber-400'
                  }`}
                >
                  {activeNeuron.activation >= 0 ? '+' : ''}
                  {activeNeuron.activation.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-400">Grad (dL/dz)</div>
                <div className="text-pink-400 font-semibold">
                  {activeNeuron.delta >= 0 ? '+' : ''}
                  {activeNeuron.delta.toFixed(3)}
                </div>
              </div>
            </div>

            {selectedLayerIdx > 0 ? (
              <div className="mb-2">
                <div className="flex items-center justify-between mb-1">
                  <label
                    htmlFor="inspector-bias-slider"
                    className="text-slate-300 font-medium"
                  >
                    Bias Threshold (b)
                  </label>
                  <span className="font-mono tabular-nums text-cyan-400">
                    {activeNeuron.bias >= 0 ? '+' : ''}
                    {activeNeuron.bias.toFixed(2)}
                  </span>
                </div>
                <input
                  id="inspector-bias-slider"
                  type="range"
                  min="-2.5"
                  max="2.5"
                  step="0.05"
                  value={activeNeuron.bias}
                  onChange={(e) =>
                    onChangeNeuronBias(
                      selectedLayerIdx,
                      selectedNeuronIdx,
                      parseFloat(e.target.value),
                    )
                  }
                  className="w-full"
                />
              </div>
            ) : (
              <div className="mb-2 text-slate-400 text-[11px]">
                Input layer node directly transmits feature coordinate{' '}
                <span className="font-mono text-slate-200">
                  x{selectedNeuronIdx + 1} ={' '}
                  {net.currentInput[selectedNeuronIdx]?.toFixed(2)}
                </span>
                .
              </div>
            )}

            {/* Incoming Synaptic Weights */}
            {selectedLayerIdx > 0 && (
              <div>
                <div className="text-[11px] text-slate-400 mb-1">
                  Incoming Synaptic Weights (Layer {selectedLayerIdx - 1} to{' '}
                  {selectedLayerIdx})
                </div>
                <div className="grid grid-cols-3 gap-1.5 font-mono tabular-nums">
                  {(net.weights[selectedLayerIdx - 1]?.[selectedNeuronIdx] ?? []).map(
                    (w, fromIdx) => (
                      <div
                        key={fromIdx}
                        className="px-2 py-1 bg-slate-900 rounded border border-slate-800 flex items-center justify-between"
                      >
                        <span className="text-slate-400 text-[10px]">
                          w{fromIdx}
                        </span>
                        <span
                          className={
                            w >= 0 ? 'text-cyan-400' : 'text-amber-400'
                          }
                        >
                          {w >= 0 ? '+' : ''}
                          {w.toFixed(2)}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}
          </aside>
        )}
      </header>

      {/* UNIFIED COLLAPSIBLE BOTTOM CONTROL DOCK (Pinned bottom-center, zero viewport occlusion) */}
      <div
        className="fixed bottom-3 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center pointer-events-none"
        style={{ width: 'min(980px, 95vw)' }}
      >
        {/* Collapse / Expand Toggle Pill */}
        <button
          type="button"
          onClick={onToggleDockCollapsed}
          aria-expanded={!isDockCollapsed}
          className="pointer-events-auto mb-1 px-3.5 py-1 bg-[#0b101b] hover:bg-slate-900 text-slate-300 hover:text-white border border-slate-800 rounded-t-md text-xs font-medium flex items-center gap-1.5 shadow-md transition-colors whitespace-nowrap"
        >
          {isDockCollapsed ? (
            <>
              <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
              <span>Show Controls (H)</span>
            </>
          ) : (
            <>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
              <span>Hide Controls (H)</span>
            </>
          )}
        </button>

        {/* Main Dock Panel */}
        {!isDockCollapsed && (
          <section
            aria-label="Neural Network Studio Controls"
            className="pointer-events-auto w-full bg-[#0b101b] border border-slate-800 rounded-xl p-2.5 sm:p-3 shadow-2xl"
          >
            {/* Top Row: Transport Actions + Mode Tabs (fits cleanly on 1 line on desktop, wraps cleanly on mobile) */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 mb-2.5 border-b border-slate-800/90">
              {/* Primary Training Transport Controls */}
              <div className="flex items-center gap-1.5 w-full sm:w-auto justify-between sm:justify-start">
                <button
                  type="button"
                  onClick={onToggleRun}
                  className={`flex-1 sm:flex-initial justify-center px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                    isRunning
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 hover:bg-amber-500/30'
                      : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
                  }`}
                >
                  {isRunning ? (
                    <>
                      <Square className="w-3.5 h-3.5 fill-current" />
                      <span>Pause</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>Train Network</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={onStepEpoch}
                  disabled={isRunning}
                  className="flex-1 sm:flex-initial justify-center px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
                >
                  <SkipForward className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Step Epoch</span>
                </button>

                <button
                  type="button"
                  onClick={onResetNetwork}
                  className="flex-1 sm:flex-initial justify-center px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              </div>

              {/* Segmented Tab Switcher */}
              <div
                role="tablist"
                aria-label="Control Dock Categories"
                className="grid grid-cols-3 sm:flex items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-lg w-full sm:w-auto"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'training'}
                  onClick={() => onChangeTab('training')}
                  className={`justify-center px-1.5 sm:px-2.5 py-1 rounded-md text-[11px] sm:text-xs font-medium flex items-center gap-1 sm:gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                    activeTab === 'training'
                      ? 'bg-slate-800 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Activity className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-cyan-400 shrink-0" />
                  <span>Training & Data</span>
                </button>

                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'architecture'}
                  onClick={() => onChangeTab('architecture')}
                  className={`justify-center px-1.5 sm:px-2.5 py-1 rounded-md text-[11px] sm:text-xs font-medium flex items-center gap-1 sm:gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                    activeTab === 'architecture'
                      ? 'bg-slate-800 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Layers className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-cyan-400 shrink-0" />
                  <span>Topology & Probe</span>
                </button>

                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'studio'}
                  onClick={() => onChangeTab('studio')}
                  className={`justify-center px-1.5 sm:px-2.5 py-1 rounded-md text-[11px] sm:text-xs font-medium flex items-center gap-1 sm:gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                    activeTab === 'studio'
                      ? 'bg-slate-800 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Sliders className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-cyan-400 shrink-0" />
                  <span>Studio & Camera</span>
                </button>
              </div>
            </div>

            {/* Bottom Row: Active Tab Controls */}
            {activeTab === 'training' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-12 gap-3 items-center text-xs">
                {/* Dataset Selector (3 cols) */}
                <div className="lg:col-span-3">
                  <label
                    htmlFor="select-dataset"
                    className="block text-[11px] text-slate-400 mb-1"
                  >
                    2D Classification Manifold
                  </label>
                  <select
                    id="select-dataset"
                    value={datasetType}
                    onChange={(e) =>
                      onChangeDataset(e.target.value as DatasetType)
                    }
                    className="w-full px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded-md text-slate-100 text-xs focus:outline-none focus:border-cyan-500"
                  >
                    {(Object.keys(DATASET_LABELS) as DatasetType[]).map((k) => (
                      <option key={k} value={k}>
                        {DATASET_LABELS[k]}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Activation Function Selector (4 cols) */}
                <div className="lg:col-span-4">
                  <span className="block text-[11px] text-slate-400 mb-1">
                    Hidden Activation ({ACTIVATION_LABELS[activationType].formula})
                  </span>
                  <div className="grid grid-cols-4 gap-1">
                    {(Object.keys(ACTIVATION_LABELS) as ActivationType[]).map(
                      (act) => (
                        <button
                          key={act}
                          type="button"
                          onClick={() => onChangeActivation(act)}
                          className={`py-1.5 px-1.5 rounded border text-[11px] font-medium transition-colors whitespace-nowrap truncate ${
                            activationType === act
                              ? 'bg-cyan-950/90 border-cyan-500/60 text-cyan-200'
                              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {ACTIVATION_LABELS[act].name}
                        </button>
                      ),
                    )}
                  </div>
                </div>

                {/* Learning Rate Slider (2 cols) */}
                <div className="lg:col-span-2">
                  <div className="flex items-center justify-between mb-1">
                    <label
                      htmlFor="slider-lr"
                      className="text-[11px] text-slate-400"
                    >
                      LR (eta)
                    </label>
                    <span className="font-mono tabular-nums text-cyan-400">
                      {learningRate.toFixed(3)}
                    </span>
                  </div>
                  <input
                    id="slider-lr"
                    type="range"
                    min="0.01"
                    max="0.25"
                    step="0.005"
                    value={learningRate}
                    onChange={(e) =>
                      onChangeLearningRate(parseFloat(e.target.value))
                    }
                    className="w-full"
                  />
                </div>

                {/* Particle Pulse Visualization Mode (3 cols) */}
                <div className="lg:col-span-3">
                  <span className="block text-[11px] text-slate-400 mb-1">
                    3D Signal Quanta Stream
                  </span>
                  <div className="grid grid-cols-3 gap-1">
                    {(
                      [
                        { id: 'forward', label: 'Forward' },
                        { id: 'backprop', label: 'Backprop' },
                        { id: 'both', label: 'Both' },
                      ] as const
                    ).map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => onChangeFlowMode(item.id)}
                        className={`py-1.5 px-1.5 rounded border text-[11px] font-medium transition-colors whitespace-nowrap truncate ${
                          flowMode === item.id
                            ? 'bg-slate-800 border-slate-600 text-white'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'architecture' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center text-xs">
                {/* Hidden Layer Count Controls */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] text-slate-400">
                      Hidden Layers ({net.layerSizes.length - 2} active)
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={onRemoveHiddenLayer}
                      disabled={net.layerSizes.length <= 3}
                      className="flex-1 py-1.5 px-2.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-200 border border-slate-700 rounded text-xs font-medium flex items-center justify-center gap-1 whitespace-nowrap"
                    >
                      <Minus className="w-3.5 h-3.5" />
                      <span>Remove Layer</span>
                    </button>
                    <button
                      type="button"
                      onClick={onAddHiddenLayer}
                      disabled={net.layerSizes.length >= 6}
                      className="flex-1 py-1.5 px-2.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-200 border border-slate-700 rounded text-xs font-medium flex items-center justify-center gap-1 whitespace-nowrap"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      <span>Add Layer</span>
                    </button>
                  </div>
                </div>

                {/* Per-Hidden-Layer Neuron Steppers */}
                <div>
                  <span className="block text-[11px] text-slate-400 mb-1">
                    Neurons per Hidden Layer (2–8 units)
                  </span>
                  <div className="flex items-center gap-1.5 overflow-x-auto">
                    {net.layerSizes.slice(1, -1).map((count, idx) => {
                      const actualLayer = idx + 1;
                      return (
                        <div
                          key={actualLayer}
                          className="flex items-center gap-1 px-2 py-1 bg-slate-900 border border-slate-800 rounded"
                        >
                          <span className="text-[11px] font-mono text-slate-400">
                            H{actualLayer}:
                          </span>
                          <button
                            type="button"
                            aria-label={`Decrease neurons in Hidden Layer ${actualLayer}`}
                            onClick={() =>
                              onChangeLayerNeurons(actualLayer, -1)
                            }
                            disabled={count <= 2}
                            className="p-0.5 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
                          >
                            <Minus className="w-3 h-3" />
                          </button>
                          <span className="font-mono tabular-nums font-semibold text-cyan-400 px-0.5">
                            {count}
                          </span>
                          <button
                            type="button"
                            aria-label={`Increase neurons in Hidden Layer ${actualLayer}`}
                            onClick={() =>
                              onChangeLayerNeurons(actualLayer, 1)
                            }
                            disabled={count >= 8}
                            className="p-0.5 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
                          >
                            <Plus className="w-3 h-3" />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Live Input Probe Coordinates Scrubber */}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label
                        htmlFor="probe-x1-slider"
                        className="text-[11px] text-slate-400"
                      >
                        Probe Input x1
                      </label>
                      <span className="font-mono tabular-nums text-cyan-400">
                        {net.currentInput[0].toFixed(2)}
                      </span>
                    </div>
                    <input
                      id="probe-x1-slider"
                      type="range"
                      min="-1"
                      max="1"
                      step="0.05"
                      value={net.currentInput[0]}
                      onChange={(e) =>
                        onChangeProbeInput(
                          parseFloat(e.target.value),
                          net.currentInput[1],
                        )
                      }
                      className="w-full"
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label
                        htmlFor="probe-x2-slider"
                        className="text-[11px] text-slate-400"
                      >
                        Probe Input x2
                      </label>
                      <span className="font-mono tabular-nums text-cyan-400">
                        {net.currentInput[1].toFixed(2)}
                      </span>
                    </div>
                    <input
                      id="probe-x2-slider"
                      type="range"
                      min="-1"
                      max="1"
                      step="0.05"
                      value={net.currentInput[1]}
                      onChange={(e) =>
                        onChangeProbeInput(
                          net.currentInput[0],
                          parseFloat(e.target.value),
                        )
                      }
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'studio' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center text-xs">
                {/* Exploded View Layer Separation Slider */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label
                      htmlFor="slider-exploded"
                      className="text-[11px] text-slate-400"
                    >
                      Exploded Layer Separation
                    </label>
                    <span className="font-mono tabular-nums text-cyan-400">
                      {explodedFactor.toFixed(2)}×
                    </span>
                  </div>
                  <input
                    id="slider-exploded"
                    type="range"
                    min="0.65"
                    max="1.85"
                    step="0.05"
                    value={explodedFactor}
                    onChange={(e) =>
                      onChangeExplodedFactor(parseFloat(e.target.value))
                    }
                    className="w-full"
                  />
                </div>

                {/* PBR Material Finish Preset */}
                <div>
                  <span className="block text-[11px] text-slate-400 mb-1">
                    PBR Hardware Finish
                  </span>
                  <div className="flex items-center gap-1">
                    {(
                      [
                        { id: 'titanium_lab', label: 'Titanium' },
                        { id: 'optical_sapphire', label: 'Sapphire' },
                        { id: 'brushed_gold', label: 'Gold' },
                      ] as const
                    ).map((f) => (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => onChangeFinishPreset(f.id)}
                        className={`flex-1 py-1.5 px-2 rounded border text-xs font-medium transition-colors whitespace-nowrap ${
                          finishPreset === f.id
                            ? 'bg-cyan-950/90 border-cyan-500/60 text-cyan-200'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Damped Camera Viewpoints */}
                <div>
                  <span className="block text-[11px] text-slate-400 mb-1">
                    Camera Viewpoint Preset
                  </span>
                  <div className="flex items-center gap-1 overflow-x-auto">
                    {(
                      [
                        { id: 'overview', label: 'Overview' },
                        { id: 'input_layer', label: 'Inputs' },
                        { id: 'hidden_core', label: 'Hidden' },
                        { id: 'manifold_focus', label: 'Manifold' },
                        { id: 'top_schematic', label: 'Top-Down' },
                      ] as const
                    ).map((cam) => (
                      <button
                        key={cam.id}
                        type="button"
                        onClick={() => onChangeCameraPreset(cam.id)}
                        className={`py-1.5 px-2 rounded border text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                          cameraPreset === cam.id
                            ? 'bg-slate-800 border-slate-600 text-white'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {cam.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </>
  );
};
