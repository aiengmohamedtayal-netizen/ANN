export type ActivationType = 'tanh' | 'relu' | 'sigmoid' | 'gelu';

export type DatasetType =
  | 'xor_spiral'
  | 'concentric_rings'
  | 'gaussian_clusters'
  | 'sine_boundary';

export interface DataPoint {
  x1: number;
  x2: number;
  label: 0 | 1;
}

export interface NeuronState {
  layerIndex: number;
  neuronIndex: number;
  bias: number;
  z: number; // pre-activation
  activation: number; // post-activation
  delta: number; // backprop error gradient
  biasGrad: number;
}

export interface SynapseState {
  id: string;
  fromLayer: number;
  fromNeuron: number;
  toLayer: number;
  toNeuron: number;
  weight: number;
  gradient: number;
  signalStrength: number; // |a_from * weight|
}

export interface ANNNetwork {
  layerSizes: number[];
  activation: ActivationType;
  neurons: NeuronState[][];
  // weights[l][toIdx][fromIdx] connects layer l -> layer l+1
  weights: number[][][];
  weightGrads: number[][][];
  weightMomentum: number[][][];
  biasMomentum: number[][];
  epoch: number;
  loss: number;
  accuracy: number;
  lossHistory: number[];
  accuracyHistory: number[];
  currentInput: [number, number];
  currentTarget: 0 | 1;
  lastStepMagnitude: number;
}

/**
 * Deterministic seeded PRNG (Mulberry32)
 */
export function createRNG(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Activation functions and their analytical derivatives with respect to pre-activation z
 */
export function activate(z: number, type: ActivationType): number {
  const clamped = Math.max(-20, Math.min(20, z));
  switch (type) {
    case 'relu':
      // Leaky ReLU with small negative slope to prevent dead neurons in compact networks
      return clamped > 0 ? clamped : 0.08 * clamped;
    case 'tanh':
      return Math.tanh(clamped);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-clamped));
    case 'gelu': {
      // Hendrycks & Gimpel approximation
      const c = Math.sqrt(2 / Math.PI);
      const inner = c * (clamped + 0.044715 * clamped * clamped * clamped);
      return 0.5 * clamped * (1 + Math.tanh(inner));
    }
  }
}

export function activateDerivative(z: number, type: ActivationType): number {
  const clamped = Math.max(-20, Math.min(20, z));
  switch (type) {
    case 'relu':
      return clamped > 0 ? 1.0 : 0.08;
    case 'tanh': {
      const t = Math.tanh(clamped);
      return 1 - t * t;
    }
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-clamped));
      return s * (1 - s);
    }
    case 'gelu': {
      const c = Math.sqrt(2 / Math.PI);
      const x3 = clamped * clamped * clamped;
      const inner = c * (clamped + 0.044715 * x3);
      const tanhInner = Math.tanh(inner);
      const sech2 = 1 - tanhInner * tanhInner;
      const dInner = c * (1 + 3 * 0.044715 * clamped * clamped);
      return 0.5 * (1 + tanhInner) + 0.5 * clamped * sech2 * dInner;
    }
  }
}

/**
 * Generates deterministic 2D classification datasets in [-1, 1]^2
 */
export function generateDataset(
  type: DatasetType,
  count = 80,
  seed = 1337,
): DataPoint[] {
  const rng = createRNG(seed);
  const points: DataPoint[] = [];

  for (let i = 0; i < count; i++) {
    const label: 0 | 1 = i < count / 2 ? 0 : 1;
    const idxInClass = i % (count / 2);
    const frac = idxInClass / (count / 2);

    if (type === 'xor_spiral') {
      // Four-quadrant XOR with slight spiral twist
      const quad = idxInClass % 2;
      const baseAngle =
        label === 1
          ? quad === 0
            ? Math.PI * 0.25
            : Math.PI * 1.25
          : quad === 0
            ? Math.PI * 0.75
            : Math.PI * 1.75;
      const radius = 0.32 + 0.48 * frac;
      const twist = (frac - 0.5) * 0.45;
      const jitterX = (rng() - 0.5) * 0.14;
      const jitterY = (rng() - 0.5) * 0.14;
      points.push({
        x1: Math.max(-1, Math.min(1, Math.cos(baseAngle + twist) * radius + jitterX)),
        x2: Math.max(-1, Math.min(1, Math.sin(baseAngle + twist) * radius + jitterY)),
        label,
      });
    } else if (type === 'concentric_rings') {
      const angle = frac * Math.PI * 2 + (rng() - 0.5) * 0.15;
      const radius = label === 1 ? 0.22 + (rng() - 0.5) * 0.14 : 0.78 + (rng() - 0.5) * 0.14;
      points.push({
        x1: Math.cos(angle) * radius,
        x2: Math.sin(angle) * radius,
        label,
      });
    } else if (type === 'gaussian_clusters') {
      const centerA = label === 1 ? [0.48, 0.45] : [-0.48, -0.45];
      const centerB = label === 1 ? [-0.45, 0.52] : [0.45, -0.52];
      const center = idxInClass % 2 === 0 ? centerA : centerB;
      const x1 = center[0] + (rng() - 0.5) * 0.42;
      const x2 = center[1] + (rng() - 0.5) * 0.42;
      points.push({
        x1: Math.max(-1, Math.min(1, x1)),
        x2: Math.max(-1, Math.min(1, x2)),
        label,
      });
    } else {
      // sine_boundary
      const x1 = -0.9 + 1.8 * frac + (rng() - 0.5) * 0.08;
      const wave = 0.42 * Math.sin(x1 * Math.PI * 1.5);
      const offset = (0.18 + 0.45 * rng()) * (label === 1 ? 1 : -1);
      const x2 = Math.max(-1, Math.min(1, wave + offset));
      points.push({ x1, x2, label });
    }
  }

  return points;
}

/**
 * Creates and initializes a multi-layer ANN with Xavier/He initialization
 */
export function createNetwork(
  layerSizes: number[] = [2, 5, 4, 1],
  activation: ActivationType = 'tanh',
  seed = 42,
): ANNNetwork {
  const sanitizedSizes = [
    2,
    ...layerSizes.slice(1, -1).map((n) => Math.max(2, Math.min(8, Math.round(n)))),
    1,
  ];
  const rng = createRNG(seed);

  const neurons: NeuronState[][] = sanitizedSizes.map((count, l) =>
    Array.from({ length: count }, (_, i) => ({
      layerIndex: l,
      neuronIndex: i,
      bias: l === 0 ? 0 : (rng() - 0.5) * 0.35,
      z: 0,
      activation: 0,
      delta: 0,
      biasGrad: 0,
    })),
  );

  const weights: number[][][] = [];
  const weightGrads: number[][][] = [];
  const weightMomentum: number[][][] = [];
  const biasMomentum: number[][] = sanitizedSizes.map((count) =>
    new Array(count).fill(0),
  );

  for (let l = 0; l < sanitizedSizes.length - 1; l++) {
    const fanIn = sanitizedSizes[l];
    const fanOut = sanitizedSizes[l + 1];
    const scale = Math.sqrt(2.4 / (fanIn + fanOut));
    const layerW: number[][] = [];
    const layerG: number[][] = [];
    const layerM: number[][] = [];

    for (let to = 0; to < fanOut; to++) {
      const rowW: number[] = [];
      const rowG: number[] = [];
      const rowM: number[] = [];
      for (let from = 0; from < fanIn; from++) {
        // Box-Muller approximate normal distribution
        const u1 = Math.max(1e-6, rng());
        const u2 = rng();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        rowW.push(normal * scale);
        rowG.push(0);
        rowM.push(0);
      }
      layerW.push(rowW);
      layerG.push(rowG);
      layerM.push(rowM);
    }
    weights.push(layerW);
    weightGrads.push(layerG);
    weightMomentum.push(layerM);
  }

  const net: ANNNetwork = {
    layerSizes: sanitizedSizes,
    activation,
    neurons,
    weights,
    weightGrads,
    weightMomentum,
    biasMomentum,
    epoch: 0,
    loss: 0.693,
    accuracy: 0.5,
    lossHistory: [],
    accuracyHistory: [],
    currentInput: [0.45, 0.35],
    currentTarget: 1,
    lastStepMagnitude: 0,
  };

  forwardPass(net, net.currentInput);
  return net;
}

/**
 * Runs forward propagation for a 2D input vector [x1, x2] and returns output probability in (0, 1)
 */
export function forwardPass(
  net: ANNNetwork,
  input: [number, number],
): number {
  net.currentInput = [input[0], input[1]];

  // Input layer (layer 0)
  net.neurons[0][0].z = input[0];
  net.neurons[0][0].activation = input[0];
  if (net.neurons[0][1]) {
    net.neurons[0][1].z = input[1];
    net.neurons[0][1].activation = input[1];
  }

  const lastLayerIdx = net.layerSizes.length - 1;

  for (let l = 1; l <= lastLayerIdx; l++) {
    const prevNeurons = net.neurons[l - 1];
    const currNeurons = net.neurons[l];
    const layerWeights = net.weights[l - 1];
    const isOutput = l === lastLayerIdx;

    for (let to = 0; to < currNeurons.length; to++) {
      let sum = currNeurons[to].bias;
      const wRow = layerWeights[to];
      for (let from = 0; from < prevNeurons.length; from++) {
        sum += prevNeurons[from].activation * wRow[from];
      }
      currNeurons[to].z = sum;
      // Output layer always uses Sigmoid for binary probability [0, 1]
      currNeurons[to].activation = isOutput
        ? activate(sum, 'sigmoid')
        : activate(sum, net.activation);
    }
  }

  return net.neurons[lastLayerIdx][0].activation;
}

/**
 * Evaluates network loss (Binary Cross-Entropy + L2) and accuracy on a dataset without mutating neuron states
 */
export function evaluateDataset(
  net: ANNNetwork,
  dataset: DataPoint[],
  l2Lambda = 0.0005,
): { loss: number; accuracy: number } {
  const savedInput: [number, number] = [...net.currentInput];
  let totalLoss = 0;
  let correct = 0;

  for (let i = 0; i < dataset.length; i++) {
    const pt = dataset[i];
    const pred = forwardPass(net, [pt.x1, pt.x2]);
    const eps = 1e-6;
    const p = Math.max(eps, Math.min(1 - eps, pred));
    totalLoss += -(pt.label * Math.log(p) + (1 - pt.label) * Math.log(1 - p));
    const predictedLabel = p >= 0.5 ? 1 : 0;
    if (predictedLabel === pt.label) correct++;
  }

  let weightSqSum = 0;
  let weightCount = 0;
  for (let l = 0; l < net.weights.length; l++) {
    for (let to = 0; to < net.weights[l].length; to++) {
      for (let from = 0; from < net.weights[l][to].length; from++) {
        const w = net.weights[l][to][from];
        weightSqSum += w * w;
        weightCount++;
      }
    }
  }

  // Restore current probe input
  forwardPass(net, savedInput);

  const avgBce = dataset.length > 0 ? totalLoss / dataset.length : 0;
  const regTerm = weightCount > 0 ? 0.5 * l2Lambda * weightSqSum : 0;
  return {
    loss: avgBce + regTerm,
    accuracy: dataset.length > 0 ? correct / dataset.length : 0,
  };
}

/**
 * Performs 1 full epoch of backpropagation with momentum across the dataset
 */
export function trainEpoch(
  net: ANNNetwork,
  dataset: DataPoint[],
  learningRate = 0.08,
  l2Lambda = 0.0005,
  momentum = 0.75,
): { loss: number; accuracy: number; stepMagnitude: number } {
  if (dataset.length === 0) {
    return { loss: net.loss, accuracy: net.accuracy, stepMagnitude: 0 };
  }

  const savedInput: [number, number] = [...net.currentInput];
  const numLayers = net.layerSizes.length;
  const lastLayer = numLayers - 1;

  // Zero out accumulated gradients
  const accWeightGrads: number[][][] = net.weights.map((layerW) =>
    layerW.map((row) => new Array(row.length).fill(0)),
  );
  const accBiasGrads: number[][] = net.layerSizes.map((count) =>
    new Array(count).fill(0),
  );

  let totalLoss = 0;
  let correct = 0;

  for (let s = 0; s < dataset.length; s++) {
    const sample = dataset[s];
    const pred = forwardPass(net, [sample.x1, sample.x2]);
    const eps = 1e-6;
    const p = Math.max(eps, Math.min(1 - eps, pred));

    totalLoss += -(
      sample.label * Math.log(p) +
      (1 - sample.label) * Math.log(1 - p)
    );
    if ((p >= 0.5 ? 1 : 0) === sample.label) {
      correct++;
    }

    // Output layer delta: dL/dz = a - y (for sigmoid + binary cross-entropy)
    const outNeuron = net.neurons[lastLayer][0];
    outNeuron.delta = p - sample.label;
    accBiasGrads[lastLayer][0] += outNeuron.delta;

    // Backpropagate through hidden layers
    for (let l = lastLayer - 1; l >= 1; l--) {
      const currLayer = net.neurons[l];
      const nextLayer = net.neurons[l + 1];
      const nextWeights = net.weights[l]; // connects l -> l+1

      for (let i = 0; i < currLayer.length; i++) {
        let errorSum = 0;
        for (let k = 0; k < nextLayer.length; k++) {
          errorSum += nextWeights[k][i] * nextLayer[k].delta;
        }
        const dAct = activateDerivative(currLayer[i].z, net.activation);
        const delta = errorSum * dAct;
        currLayer[i].delta = delta;
        accBiasGrads[l][i] += delta;
      }
    }

    // Accumulate weight gradients
    for (let l = 0; l < lastLayer; l++) {
      const fromNeurons = net.neurons[l];
      const toNeurons = net.neurons[l + 1];
      for (let to = 0; to < toNeurons.length; to++) {
        const delta = toNeurons[to].delta;
        for (let from = 0; from < fromNeurons.length; from++) {
          accWeightGrads[l][to][from] += delta * fromNeurons[from].activation;
        }
      }
    }
  }

  const invN = 1 / dataset.length;
  let stepMagSq = 0;

  // Apply weight and bias updates with momentum
  for (let l = 0; l < lastLayer; l++) {
    for (let to = 0; to < net.weights[l].length; to++) {
      for (let from = 0; from < net.weights[l][to].length; from++) {
        const w = net.weights[l][to][from];
        const grad = accWeightGrads[l][to][from] * invN + l2Lambda * w;
        net.weightGrads[l][to][from] = grad;
        const vel =
          momentum * net.weightMomentum[l][to][from] - learningRate * grad;
        net.weightMomentum[l][to][from] = vel;
        net.weights[l][to][from] = Math.max(-6, Math.min(6, w + vel));
        stepMagSq += vel * vel;
      }
    }
  }

  for (let l = 1; l <= lastLayer; l++) {
    for (let i = 0; i < net.neurons[l].length; i++) {
      const bGrad = accBiasGrads[l][i] * invN;
      net.neurons[l][i].biasGrad = bGrad;
      const bVel = momentum * net.biasMomentum[l][i] - learningRate * bGrad;
      net.biasMomentum[l][i] = bVel;
      net.neurons[l][i].bias = Math.max(
        -4,
        Math.min(4, net.neurons[l][i].bias + bVel),
      );
      stepMagSq += bVel * bVel;
    }
  }

  // Re-run forward pass on saved probe input so live 3D neuron activations & gradients reflect the probe
  const probePred = forwardPass(net, savedInput);
  const outN = net.neurons[lastLayer][0];
  outN.delta = probePred - net.currentTarget;
  for (let l = lastLayer - 1; l >= 1; l--) {
    const currLayer = net.neurons[l];
    const nextLayer = net.neurons[l + 1];
    const nextWeights = net.weights[l];
    for (let i = 0; i < currLayer.length; i++) {
      let err = 0;
      for (let k = 0; k < nextLayer.length; k++) {
        err += nextWeights[k][i] * nextLayer[k].delta;
      }
      currLayer[i].delta = err * activateDerivative(currLayer[i].z, net.activation);
    }
  }

  net.epoch += 1;
  net.loss = totalLoss * invN;
  net.accuracy = correct * invN;
  net.lastStepMagnitude = Math.sqrt(stepMagSq);

  net.lossHistory.push(net.loss);
  if (net.lossHistory.length > 80) net.lossHistory.shift();

  net.accuracyHistory.push(net.accuracy);
  if (net.accuracyHistory.length > 80) net.accuracyHistory.shift();

  return {
    loss: net.loss,
    accuracy: net.accuracy,
    stepMagnitude: net.lastStepMagnitude,
  };
}

/**
 * Evaluates a 2D grid of predictions across [-1, 1]^2 for the 3D Decision Boundary Manifold mesh
 */
export function evaluateManifoldGrid(
  net: ANNNetwork,
  gridSize = 21,
): Float32Array {
  const savedInput: [number, number] = [...net.currentInput];
  const result = new Float32Array(gridSize * gridSize);

  let idx = 0;
  for (let row = 0; row < gridSize; row++) {
    const x2 = -1 + (2 * row) / (gridSize - 1);
    for (let col = 0; col < gridSize; col++) {
      const x1 = -1 + (2 * col) / (gridSize - 1);
      result[idx++] = forwardPass(net, [x1, x2]);
    }
  }

  forwardPass(net, savedInput);
  return result;
}

/**
 * Flattens all synapses for UI inspection and 3D rendering
 */
export function getSynapseList(net: ANNNetwork): SynapseState[] {
  const list: SynapseState[] = [];
  for (let l = 0; l < net.weights.length; l++) {
    const fromLayer = net.neurons[l];
    for (let to = 0; to < net.weights[l].length; to++) {
      for (let from = 0; from < net.weights[l][to].length; from++) {
        const w = net.weights[l][to][from];
        const g = net.weightGrads[l][to][from];
        const aFrom = fromLayer[from]?.activation ?? 0;
        list.push({
          id: `L${l}_${from}_to_L${l + 1}_${to}`,
          fromLayer: l,
          fromNeuron: from,
          toLayer: l + 1,
          toNeuron: to,
          weight: w,
          gradient: g,
          signalStrength: Math.abs(aFrom * w),
        });
      }
    }
  }
  return list;
}
