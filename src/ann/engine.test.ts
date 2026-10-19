import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activate,
  activateDerivative,
  createNetwork,
  evaluateDataset,
  evaluateManifoldGrid,
  forwardPass,
  generateDataset,
  getSynapseList,
  trainEpoch,
} from './engine.ts';

test('activation functions and derivatives behave accurately', () => {
  assert.equal(activate(0, 'sigmoid'), 0.5);
  assert.equal(activateDerivative(0, 'sigmoid'), 0.25);

  assert.equal(activate(0, 'tanh'), 0);
  assert.equal(activateDerivative(0, 'tanh'), 1);

  assert.equal(activate(2.5, 'relu'), 2.5);
  assert.ok(Math.abs(activate(-1, 'relu') - -0.08) < 1e-6);
  assert.equal(activateDerivative(2.5, 'relu'), 1.0);
  assert.equal(activateDerivative(-2.5, 'relu'), 0.08);

  const geluPos = activate(1.5, 'gelu');
  assert.ok(geluPos > 1.3 && geluPos <= 1.5);
  assert.ok(activateDerivative(1.0, 'gelu') > 0.5);
});

test('dataset generator creates deterministic balanced 2D samples within [-1, 1]', () => {
  const spiral = generateDataset('xor_spiral', 60, 99);
  assert.equal(spiral.length, 60);
  const posCount = spiral.filter((d) => d.label === 1).length;
  const negCount = spiral.filter((d) => d.label === 0).length;
  assert.equal(posCount, 30);
  assert.equal(negCount, 30);

  for (const pt of spiral) {
    assert.ok(pt.x1 >= -1 && pt.x1 <= 1);
    assert.ok(pt.x2 >= -1 && pt.x2 <= 1);
  }
});

test('createNetwork and forwardPass produce valid probabilities and synapse counts', () => {
  const net = createNetwork([2, 5, 4, 1], 'tanh', 42);
  assert.deepEqual(net.layerSizes, [2, 5, 4, 1]);

  const out = forwardPass(net, [0.6, -0.4]);
  assert.ok(out > 0 && out < 1, `Output probability ${out} must be in (0, 1)`);

  const synapses = getSynapseList(net);
  // 2*5 + 5*4 + 4*1 = 10 + 20 + 4 = 34 synapses
  assert.equal(synapses.length, 34);
});

test('trainEpoch reduces loss and improves accuracy on non-linear concentric rings', () => {
  const dataset = generateDataset('concentric_rings', 60, 101);
  const net = createNetwork([2, 6, 4, 1], 'tanh', 101);

  const initialMetrics = evaluateDataset(net, dataset);
  for (let e = 0; e < 35; e++) {
    trainEpoch(net, dataset, 0.14, 0.0002, 0.8);
  }
  const finalMetrics = evaluateDataset(net, dataset);

  assert.equal(net.epoch, 35);
  assert.ok(
    finalMetrics.loss < initialMetrics.loss,
    `Expected final loss (${finalMetrics.loss}) < initial loss (${initialMetrics.loss})`,
  );
  assert.ok(
    finalMetrics.accuracy >= initialMetrics.accuracy,
    `Expected final accuracy (${finalMetrics.accuracy}) >= initial accuracy (${initialMetrics.accuracy})`,
  );
});

test('evaluateManifoldGrid returns expected Float32Array of probabilities without mutating currentInput', () => {
  const net = createNetwork([2, 4, 1], 'relu', 7);
  forwardPass(net, [0.25, -0.75]);
  const grid = evaluateManifoldGrid(net, 11);

  assert.equal(grid.length, 121);
  for (let i = 0; i < grid.length; i++) {
    assert.ok(grid[i] >= 0 && grid[i] <= 1);
  }
  assert.deepEqual(net.currentInput, [0.25, -0.75]);
});
