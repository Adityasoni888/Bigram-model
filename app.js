import { AutoModelForCausalLM, AutoTokenizer, Tensor } from '@huggingface/transformers';

const MODEL_ID = 'Xenova/distilgpt2';
const MAX_TEXT = 600;
const $ = (id) => document.getElementById(id);
let tokenizer;
let model;
let modelTask;
let simulator = [
  { token: 'signal', probability: 0.5 },
  { token: 'noise', probability: 0.5 },
];
let lastGenerated = '';
let toastTimer;

function say(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2500);
}

function setStatus(message, kind = '') {
  $('modelStatus').textContent = message;
  $('modelStatus').previousElementSibling.className = `status-dot ${kind}`;
}

async function loadModel() {
  if (model) return;
  if (!modelTask) {
    modelTask = (async () => {
      setStatus('Loading model…', 'loading');
      $('modelProgress').classList.remove('hidden');
      $('progressLabel').textContent = 'Loading DistilGPT-2 into this browser. First use can take a few minutes…';
      try {
        tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
        model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype: 'q8' });
        setStatus('Model ready', 'ready');
        $('progressLabel').textContent = 'Model ready. Your text stays on this device.';
        setTimeout(() => $('modelProgress').classList.add('hidden'), 1800);
      } catch (error) {
        modelTask = null;
        const networkBlocked = error.message === 'Failed to fetch';
        setStatus(networkBlocked ? 'Network unavailable' : 'Model unavailable', '');
        $('progressLabel').textContent = networkBlocked
          ? 'The model host could not be reached. Check your connection and retry.'
          : `Model could not load: ${error.message}. Check your connection and retry.`;
        throw error;
      }
    })();
  }
  return modelTask;
}

function toTokenIds(encoded) {
  // Transformers.js returns int64 token IDs. Convert them to plain numbers
  // before adding the beginning marker needed to score the first visible token.
  return Array.from(encoded.input_ids.data, Number);
}

function tokenTensor(ids) {
  return new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
}

function softmaxTop(logits, offset, vocabSize, count = 6) {
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) if (logits[offset + i] > max) max = logits[offset + i];
  let total = 0;
  const top = [];
  for (let i = 0; i < vocabSize; i++) {
    const p = Math.exp(logits[offset + i] - max);
    total += p;
    if (top.length < count || p > top.at(-1).weight) {
      let place = top.findIndex((item) => p > item.weight);
      if (place < 0) place = top.length;
      top.splice(place, 0, { id: i, weight: p });
      if (top.length > count) top.pop();
    }
  }
  return top.map(({ id, weight }) => ({ id, probability: weight / total }));
}

async function inspectText(text) {
  await loadModel();
  const encoded = await tokenizer(text, { add_special_tokens: false, truncation: true, max_length: 120 });
  const visibleIds = toTokenIds(encoded);
  if (!visibleIds.length) throw new Error('Enter at least one token to inspect.');
  // GPT-2 uses its end token as a sequence-start context marker.
  const startId = Number(tokenizer.eos_token_id ?? 50256);
  const ids = [startId, ...visibleIds];
  const output = await model({ input_ids: tokenTensor(ids) });
  const logits = output.logits.data;
  const vocabSize = output.logits.dims.at(-1);
  const rows = visibleIds.map((id, index) => {
    const offset = index * vocabSize;
    let max = -Infinity;
    let bestId = 0;
    for (let v = 0; v < vocabSize; v++) if (logits[offset + v] > max) max = logits[offset + v];
    for (let v = 0; v < vocabSize; v++) if (logits[offset + v] === max) { bestId = v; break; }
    let total = 0;
    for (let v = 0; v < vocabSize; v++) total += Math.exp(logits[offset + v] - max);
    const probability = Math.exp(logits[offset + id] - max) / total;
    return { id, token: tokenizer.decode([id]), probability, expected: bestId === id };
  });
  const logMean = rows.reduce((sum, row) => sum + Math.log(Math.max(row.probability, Number.MIN_VALUE)), 0) / rows.length;
  return { text, rows, ppl: Math.exp(-logMean), logMean, truncated: visibleIds.length >= 120 };
}

async function nextOptions(text, count = 6) {
  await loadModel();
  const encoded = await tokenizer(text, { add_special_tokens: false, truncation: true, max_length: 120 });
  const ids = toTokenIds(encoded);
  if (!ids.length) ids.push(Number(tokenizer.eos_token_id ?? 50256));
  const output = await model({ input_ids: tokenTensor(ids) });
  const vocabSize = output.logits.dims.at(-1);
  const offset = (ids.length - 1) * vocabSize;
  return softmaxTop(output.logits.data, offset, vocabSize, count).map((choice) => ({
    ...choice,
    text: tokenizer.decode([choice.id]),
  }));
}

function formatPpl(value) {
  if (!Number.isFinite(value)) return '∞';
  if (value >= 1000) return value.toExponential(2);
  if (value >= 100) return value.toFixed(1);
  if (value >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function describePpl(value) {
  if (value < 5) return ['HIGH CONFIDENCE', 'The sequence looks familiar to this model.', 'good'];
  if (value < 30) return ['EXPECTED', 'The model has a fairly clear path through this text.', 'good'];
  if (value < 120) return ['LESS EXPECTED', 'Some token choices are less predictable in context.', 'mid'];
  if (value < 600) return ['UNUSUAL', 'The sequence contains choices the model rarely expects.', 'high'];
  return ['HIGH UNCERTAINTY', 'The model finds this sequence difficult to anticipate.', 'high'];
}

function renderScore(result) {
  $('scoreEmpty').classList.add('hidden');
  $('scoreResult').classList.remove('hidden');
  const [label, read, tone] = describePpl(result.ppl);
  $('scoreBadge').textContent = label;
  $('scoreBadge').className = `badge ${tone}`;
  $('pplValue').textContent = formatPpl(result.ppl);
  $('scoreRead').textContent = read;
  $('tokenTotal').textContent = `${result.rows.length} tokens${result.truncated ? ' · input truncated at 120' : ''}`;
  const meterPosition = Math.min(100, Math.max(1, Math.log10(Math.max(1, result.ppl)) / 3 * 100));
  $('scoreMeter').style.left = `${meterPosition}%`;
  $('tokenProfile').replaceChildren(...result.rows.map((row) => {
    const piece = document.createElement('span');
    const tone = row.probability > 0.15 ? 'low' : row.probability < 0.005 ? 'high' : '';
    piece.className = `token-piece ${tone}`;
    piece.textContent = row.token.replaceAll(' ', '␠') || '∅';
    piece.title = `Token: ${JSON.stringify(row.token)} · probability ${row.probability.toPrecision(3)} · ${row.expected ? 'among top prediction' : 'not the top prediction'}`;
    return piece;
  }));
  $('formulaNote').textContent = `PPL = exp(− mean(log P(token))) across ${result.rows.length} model tokens.`;
}

function readHistory() {
  try { return JSON.parse(localStorage.getItem('token-observatory-history') || '[]'); }
  catch { return []; }
}

function saveHistory(history) {
  localStorage.setItem('token-observatory-history', JSON.stringify(history.slice(0, 50)));
  renderHistory();
}

function addHistory(result) {
  const history = readHistory();
  history.unshift({ id: crypto.randomUUID(), text: result.text, ppl: result.ppl, tokens: result.rows.length, date: new Date().toISOString() });
  saveHistory(history);
}

function renderHistory() {
  const query = $('historySearch').value.trim().toLowerCase();
  const sort = $('historySort').value;
  const history = readHistory().filter((row) => row.text.toLowerCase().includes(query));
  history.sort((a, b) => sort === 'recent' ? new Date(b.date) - new Date(a.date) : sort === 'surprising' ? b.ppl - a.ppl : a.ppl - b.ppl);
  $('historyCount').textContent = `${history.length} ${history.length === 1 ? 'entry' : 'entries'}`;
  const list = $('historyList');
  if (!history.length) {
    list.innerHTML = `<div class="history-empty">${query ? 'No entries match this filter.' : 'Your inspected sequences will be collected here.'}</div>`;
    return;
  }
  list.replaceChildren(...history.map((row) => {
    const item = document.createElement('div'); item.className = 'history-item';
    const txt = document.createElement('span'); txt.className = 'history-text'; txt.textContent = row.text; txt.title = row.text;
    const ppl = document.createElement('span'); ppl.className = 'h-ppl'; ppl.textContent = `PPL ${formatPpl(row.ppl)}`;
    const tokens = document.createElement('span'); tokens.className = 'h-tokens'; tokens.textContent = `${row.tokens} tokens`;
    const date = document.createElement('span'); date.className = 'h-date'; date.textContent = new Date(row.date).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    const load = document.createElement('button'); load.textContent = 'Inspect again ↗'; load.addEventListener('click', () => { $('sentenceInput').value = row.text; $('sentenceInput').dispatchEvent(new Event('input')); $('analyzeBtn').click(); window.location.hash = 'analyze'; });
    item.append(txt, ppl, tokens, date, load); return item;
  }));
}

async function runAnalyze() {
  const text = $('sentenceInput').value.trim();
  if (!text) { say('Add some text before inspecting.'); $('sentenceInput').focus(); return; }
  $('analyzeBtn').disabled = true;
  $('analyzeBtn').textContent = 'Inspecting…';
  $('modelProgress').classList.remove('hidden');
  $('progressLabel').textContent = 'Running the model on this device…';
  try {
    const result = await inspectText(text);
    renderScore(result); addHistory(result);
    $('progressLabel').textContent = 'Inspection complete. Your text stayed in this browser.';
    say('Sequence inspected.');
  } catch (error) { say(error.message || 'Could not inspect this sequence.'); }
  finally { $('analyzeBtn').disabled = false; $('analyzeBtn').innerHTML = 'Inspect sequence <span>↗</span>'; setTimeout(() => $('modelProgress').classList.add('hidden'), 1800); }
}

async function runPredict() {
  const context = $('promptInput').value.trim();
  if (!context) { say('Add a starting context first.'); $('promptInput').focus(); return; }
  $('predictBtn').disabled = true; $('predictBtn').textContent = 'Thinking…';
  $('suggestions').innerHTML = '<div class="suggestion-empty">Reading the context…</div>';
  try {
    const choices = await nextOptions(context);
    $('suggestions').replaceChildren(...choices.map((choice) => {
      const button = document.createElement('button'); button.className = 'suggestion';
      const word = document.createElement('b'); word.textContent = choice.text.replaceAll(' ', '␠') || '∅';
      const prob = document.createElement('span'); prob.textContent = `${(choice.probability * 100).toFixed(2)}%`;
      const marker = document.createElement('small'); marker.textContent = 'ADD TO CONTEXT +';
      button.append(word, prob, marker); button.title = `Add ${JSON.stringify(choice.text)} to the context`;
      button.addEventListener('click', () => { $('promptInput').value += choice.text; $('promptInput').dispatchEvent(new Event('input')); runPredict(); });
      return button;
    }));
  } catch (error) { $('suggestions').innerHTML = '<div class="suggestion-empty">Suggestions are unavailable. Check the model connection and retry.</div>'; say(error.message); }
  finally { $('predictBtn').disabled = false; $('predictBtn').innerHTML = 'Suggest next <span>→</span>'; }
}

async function runGenerate() {
  const prompt = $('promptInput').value.trim();
  if (!prompt) { say('Add a starting context first.'); return; }
  $('generateBtn').disabled = true; $('generateBtn').textContent = 'Generating…';
  try {
    await loadModel();
    const encoded = await tokenizer(prompt, { add_special_tokens: false, truncation: true, max_length: 120 });
    const ids = toTokenIds(encoded);
    const output = await model.generate(tokenTensor(ids), { max_new_tokens: 10, do_sample: true, temperature: 0.8, top_k: 40 });
    const allIds = Array.from(output.data, Number);
    lastGenerated = tokenizer.decode(allIds, { skip_special_tokens: true });
    $('generatedText').textContent = lastGenerated;
    $('generatedBox').classList.remove('hidden');
  } catch (error) { say(error.message || 'Generation failed.'); }
  finally { $('generateBtn').disabled = false; $('generateBtn').textContent = 'Generate 10 tokens ↗'; }
}

function drawSim() {
  const rows = $('simRows');
  rows.replaceChildren(...simulator.map((entry, index) => {
    const row = document.createElement('div'); row.className = 'sim-row';
    const token = document.createElement('input'); token.type = 'text'; token.value = entry.token; token.setAttribute('aria-label', `Token ${index + 1} label`); token.maxLength = 24;
    token.addEventListener('input', () => { simulator[index].token = token.value; });
    const slider = document.createElement('input'); slider.type = 'range'; slider.min = '0.001'; slider.max = '1'; slider.step = '0.001'; slider.value = entry.probability; slider.setAttribute('aria-label', `Probability for ${entry.token}`);
    const value = document.createElement('span'); value.className = 'prob-value'; value.textContent = Number(entry.probability).toFixed(3);
    slider.addEventListener('input', () => { simulator[index].probability = Number(slider.value); value.textContent = Number(slider.value).toFixed(3); updateSim(); });
    const remove = document.createElement('button'); remove.className = 'remove-row'; remove.textContent = '×'; remove.title = `Remove ${entry.token}`; remove.setAttribute('aria-label', `Remove ${entry.token}`);
    remove.addEventListener('click', () => { if (simulator.length <= 1) return say('Keep at least one token in the sandbox.'); simulator.splice(index, 1); drawSim(); });
    row.append(token, slider, value, remove); return row;
  }));
  updateSim();
}

function updateSim() {
  const values = simulator.map((x) => Math.max(0.001, x.probability));
  const logMean = values.reduce((sum, p) => sum + Math.log(p), 0) / values.length;
  const ppl = Math.exp(-logMean);
  $('simPpl').textContent = formatPpl(ppl);
  $('simN').textContent = `N = ${values.length}`;
  const productLog = values.reduce((sum, p) => sum - Math.log(p), 0);
  const product = productLog < 690 ? Math.exp(productLog) : Infinity;
  $('inverseProduct').textContent = values.map((p) => formatPpl(1 / p)).join(' × ') + ` = ${formatPpl(product)}`;
  $('simFormula').innerHTML = `${formatPpl(product)}<sup>1/${values.length}</sup> = ${formatPpl(ppl)}`;
  $('simInterpretation').textContent = `A model with these probabilities is as uncertain as choosing between ${formatPpl(ppl)} equally likely options per token.`;
  const choiceViz = $('choiceViz');
  const blocks = Math.min(32, Math.max(1, Math.ceil(ppl)));
  choiceViz.replaceChildren(...Array.from({ length: blocks }, (_, i) => { const bar = document.createElement('i'); bar.style.height = `${8 + ((i * 17 + 11) % 17)}px`; bar.style.animationDelay = `${Math.min(i * 8, 150)}ms`; return bar; }));
}

const simPresets = {
  phrase: [{ token: 'the', probability: .5 }, { token: 'light', probability: .8 }, { token: 'turned', probability: .2 }],
  uniform: ['signal', 'pattern', 'context', 'token', 'model', 'sequence', 'probability', 'output'].map((token) => ({ token, probability: .125 })),
  certain: [{ token: 'and', probability: 1 }, { token: 'then', probability: 1 }, { token: 'again', probability: 1 }],
  surprise: [{ token: 'the', probability: .8 }, { token: 'unexpected', probability: .02 }, { token: 'ending', probability: .6 }],
  coin: [{ token: 'option A', probability: .5 }, { token: 'option B', probability: .5 }],
};

const comparePairs = {
  order: ['Birds fly over the quiet lake.', 'Over quiet the fly lake birds.'],
  meaning: ['A warm welcome waited at home.', 'A storm of ice fell on the city.'],
  style: ['I could not believe what happened.', 'could not believe happen I, what!'],
};

async function compareSequences() {
  const a = $('compareA').value.trim(), b = $('compareB').value.trim();
  if (!a || !b) return say('Fill both sequences to compare.');
  $('compareBtn').disabled = true; $('compareBtn').textContent = 'Comparing…';
  try {
    const [one, two] = await Promise.all([inspectText(a), inspectText(b)]);
    $('pplA').textContent = formatPpl(one.ppl); $('pplB').textContent = formatPpl(two.ppl);
    const scale = Math.max(one.ppl, two.ppl, 1);
    $('barA').style.width = `${Math.max(2, one.ppl / scale * 100)}%`;
    $('barB').style.width = `${Math.max(2, two.ppl / scale * 100)}%`;
    $('readA').textContent = `${one.rows.length} tokens · ${describePpl(one.ppl)[0].toLowerCase()}`;
    $('readB').textContent = `${two.rows.length} tokens · ${describePpl(two.ppl)[0].toLowerCase()}`;
    const difference = Math.abs(one.ppl - two.ppl);
    const moreNatural = one.ppl < two.ppl ? 'Sequence A' : one.ppl > two.ppl ? 'Sequence B' : 'Both sequences';
    $('compareSummary').textContent = `${moreNatural} has lower perplexity here. The score gap is ${formatPpl(difference)}. These are model expectations, not measures of truth or quality.`;
  } catch (error) { say(error.message || 'Could not compare the sequences.'); }
  finally { $('compareBtn').disabled = false; $('compareBtn').innerHTML = 'Compare sequences <span>↗</span>'; }
}

function exportCSV() {
  const history = readHistory();
  if (!history.length) return say('There are no session entries to export.');
  const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const csv = ['text,perplexity,tokens,inspected_at', ...history.map((item) => [item.text, item.ppl, item.tokens, item.date].map(quote).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = 'token-observatory-sessions.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('sentenceInput').addEventListener('input', (event) => { $('charCount').textContent = `${event.target.value.length} chars`; });
$('sentenceInput').dispatchEvent(new Event('input'));
$('analyzeBtn').addEventListener('click', runAnalyze);
document.querySelectorAll('[data-sentence]').forEach((button) => button.addEventListener('click', () => { $('sentenceInput').value = button.dataset.sentence; $('sentenceInput').dispatchEvent(new Event('input')); }));
document.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); runAnalyze(); } });
$('promptInput').addEventListener('input', () => { $('promptCount').textContent = `${$('promptInput').value.length} / 300`; });
$('promptInput').dispatchEvent(new Event('input'));
$('predictBtn').addEventListener('click', runPredict);
$('generateBtn').addEventListener('click', runGenerate);
$('clearPrompt').addEventListener('click', () => { $('promptInput').value = ''; $('promptInput').dispatchEvent(new Event('input')); $('suggestions').innerHTML = '<div class="suggestion-empty">Your next-token options will appear here.</div>'; $('generatedBox').classList.add('hidden'); });
$('useGenerated').addEventListener('click', () => { if (lastGenerated) { $('promptInput').value = lastGenerated; $('promptInput').dispatchEvent(new Event('input')); $('promptInput').focus(); } });
document.querySelectorAll('[data-pair]').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('[data-pair]').forEach((chip) => chip.classList.toggle('active', chip === button)); [$('compareA').value, $('compareB').value] = comparePairs[button.dataset.pair]; }));
$('compareBtn').addEventListener('click', compareSequences);
document.querySelectorAll('[data-sim]').forEach((button) => button.addEventListener('click', () => { simulator = simPresets[button.dataset.sim].map((item) => ({ ...item })); drawSim(); }));
$('addToken').addEventListener('click', () => { simulator.push({ token: `token ${simulator.length + 1}`, probability: .5 }); drawSim(); });
$('historySearch').addEventListener('input', renderHistory);
$('historySort').addEventListener('change', renderHistory);
$('clearHistory').addEventListener('click', () => { if (!readHistory().length) return say('The session log is already empty.'); localStorage.removeItem('token-observatory-history'); renderHistory(); say('Session log cleared.'); });
$('exportHistory').addEventListener('click', exportCSV);
$('themeToggle').addEventListener('click', () => { document.body.classList.toggle('light'); localStorage.setItem('token-observatory-light', document.body.classList.contains('light')); });
if (localStorage.getItem('token-observatory-light') === 'true') document.body.classList.add('light');
$('presentToggle').addEventListener('click', () => { document.body.classList.toggle('present-mode'); say(document.body.classList.contains('present-mode') ? 'Presentation view enabled.' : 'Presentation view closed.'); });
renderHistory();
drawSim();
