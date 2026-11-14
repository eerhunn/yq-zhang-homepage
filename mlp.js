(() => {
  const samples = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  const initial = [2, -2, -2, 2, -1, -1, 3, 3, -2.5];
  const names = ['W₁₁', 'W₁₂', 'W₂₁', 'W₂₂', 'b₁', 'b₂', 'v₁', 'v₂', 'b输出'];
  const colors = ['#66ddff', '#ffa987', '#ffe18c', '#b8a3ff', '#90e6ac', '#f6a3d2', '#a7c5ff', '#d8ef65', '#f2f5ff'];
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  function forward(p, x, y) {
    const h1 = sigmoid(x * p[0] + y * p[2] + p[4]);
    const h2 = sigmoid(x * p[1] + y * p[3] + p[5]);
    const z = h1 * p[6] + h2 * p[7] + p[8];
    return { h1, h2, z, probability: sigmoid(z) };
  }

  function evaluate(p) {
    const gradient = Array(p.length).fill(0);
    let loss = 0;
    const predictions = samples.map(([x, y, target]) => {
      const { h1, h2, z, probability } = forward(p, x, y);
      loss += Math.max(z, 0) - z * target + Math.log1p(Math.exp(-Math.abs(z)));
      const d = (probability - target) / samples.length;
      const d1 = d * p[6] * h1 * (1 - h1);
      const d2 = d * p[7] * h2 * (1 - h2);
      [x * d1, x * d2, y * d1, y * d2, d1, d2, h1 * d, h2 * d, d]
        .forEach((value, i) => { gradient[i] += value; });
      return probability;
    });
    return { loss: loss / samples.length, gradient, predictions };
  }

  function train(rate = 0.03, epochs = 300) {
    const p = initial.slice();
    const m = p.map(() => 0);
    const v = p.map(() => 0);
    const history = [{ p: p.slice(), ...evaluate(p) }];
    for (let t = 1; t <= epochs; t++) {
      history[t - 1].gradient.forEach((g, i) => {
        m[i] = 0.9 * m[i] + 0.1 * g;
        v[i] = 0.999 * v[i] + 0.001 * g * g;
        const correctedM = m[i] / (1 - 0.9 ** t);
        const correctedV = v[i] / (1 - 0.999 ** t);
        p[i] -= rate * correctedM / (Math.sqrt(correctedV) + 1e-8);
      });
      history.push({ p: p.slice(), ...evaluate(p) });
    }
    return history;
  }

  if (typeof module !== 'undefined') {
    module.exports = { train, evaluate, forward, samples };
    return;
  }

  const byId = id => document.getElementById(id);
  const history = train();
  const slider = byId('epoch');
  const select = byId('parameter');
  const play = byId('play');
  let timer = null;
  names.forEach((name, i) => select.add(new Option(name, i)));

  function lineChart(id, values, epoch, color) {
    const canvas = byId(id);
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const left = 58, right = width - 18, top = 15, bottom = height - 31;
    const spread = Math.max(0.05, Math.max(...values) - Math.min(...values));
    const min = Math.min(...values) - spread * 0.08;
    const max = Math.max(...values) + spread * 0.08;
    const x = i => left + i * (right - left) / (values.length - 1);
    const y = value => bottom - (value - min) / (max - min) * (bottom - top);
    ctx.clearRect(0, 0, width, height);
    ctx.font = '14px system-ui';
    ctx.fillStyle = '#afc2df';
    ctx.strokeStyle = '#253651';
    for (let i = 0; i < 4; i++) {
      const value = min + (max - min) * i / 3;
      const row = y(value);
      ctx.fillText(value.toFixed(2), 2, row + 4);
      ctx.beginPath();
      ctx.moveTo(left, row);
      ctx.lineTo(right, row);
      ctx.stroke();
    }
    [0, 100, 200, 300].forEach(t => ctx.fillText(t, x(t) - 10, height - 8));
    ctx.beginPath();
    values.forEach((value, i) => i === 0 ? ctx.moveTo(x(i), y(value)) : ctx.lineTo(x(i), y(value)));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x(epoch), y(values[epoch]), 5, 0, Math.PI * 2);
    ctx.fill();
    canvas.setAttribute('aria-label', `训练曲线，第 ${epoch} 轮的数值为 ${values[epoch].toFixed(4)}`);
  }

  function decisionMap(p) {
    const canvas = byId('map');
    const ctx = canvas.getContext('2d');
    const n = 60, inset = 31, size = 240;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const x = -0.2 + 1.4 * (col + 0.5) / n;
        const y = 1.2 - 1.4 * (row + 0.5) / n;
        const p1 = forward(p, x, y).probability;
        const rgb = [95 + 159 * p1, 150 - 10 * p1, 245 - 129 * p1];
        ctx.fillStyle = `rgb(${rgb.map(Math.round).join(',')})`;
        ctx.fillRect(inset + col * 4, inset + row * 4, 4, 4);
      }
    }
    ctx.fillStyle = '#dce6f7';
    ctx.font = '16px system-ui';
    ctx.fillText('x₁ →', 234, 296);
    ctx.fillText('x₂ ↑', 5, 21);
    samples.forEach(([x, y, target]) => {
      const px = inset + (x + 0.2) / 1.4 * size;
      const py = inset + (1.2 - y) / 1.4 * size;
      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI * 2);
      ctx.fillStyle = '#111a2c';
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(target, px - 4.5, py + 5);
    });
  }

  function render() {
    const epoch = Number(slider.value);
    const { loss, p, predictions } = history[epoch];
    const i = Number(select.value);
    const correct = predictions.filter((value, j) => Number(value >= 0.5) === samples[j][2]).length;
    byId('epoch-value').textContent = epoch;
    byId('loss-value').textContent = loss.toFixed(4);
    byId('accuracy').textContent = `${correct} / 4`;
    byId('parameters').innerHTML = names.map((name, j) =>
      `<tr><th scope="row">${name}</th><td>${p[j].toFixed(4)}</td><td>${(p[j] - initial[j]).toFixed(4)}</td></tr>`).join('');
    byId('predictions').innerHTML = samples.map(([x, y, target], j) =>
      `<tr><th scope="row">(${x}, ${y})</th><td>${target}</td><td>${predictions[j].toFixed(4)}</td><td>${Number(predictions[j] >= 0.5)}</td></tr>`).join('');
    lineChart('loss', history.map(state => state.loss), epoch, colors[0]);
    lineChart('trace', history.map(state => state.p[i]), epoch, colors[i]);
    decisionMap(p);
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = '播放训练';
  }

  play.addEventListener('click', () => {
    if (timer !== null) return stop();
    if (Number(slider.value) === 300) slider.value = 0;
    render();
    play.textContent = '暂停';
    timer = setInterval(() => {
      slider.value = Math.min(300, Number(slider.value) + 2);
      render();
      if (Number(slider.value) >= 300) stop();
    }, 35);
  });
  byId('reset').addEventListener('click', () => { stop(); slider.value = 0; render(); });
  byId('end').addEventListener('click', () => { stop(); slider.value = 300; render(); });
  slider.addEventListener('input', () => { stop(); render(); });
  select.addEventListener('change', render);
  render();
})();
