const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const TELEMETRY_INTERVAL_MS = 30_000;
const useMongo = Boolean(process.env.MONGODB_URI);
const dataDirectory = path.join(__dirname, 'data');
const telemetryCsvPath = path.join(dataDirectory, 'telemetry.csv');
const telemetryCsvColumns = ['id', 'batteryId', 'timestamp', 'soc', 'voltage', 'current', 'temperatureC', 'stateOfHealth', 'cycleCount', 'location', 'status', 'source'];
const mlWorkerPath = path.join(__dirname, 'ml_model', 'inference_worker.py');
const projectPython = path.join(__dirname, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const mlPython = process.env.ML_PYTHON || (fs.existsSync(projectPython) ? projectPython : (process.platform === 'win32' ? 'python' : 'python3'));
let mlWorker = null;
let mlStatus = { status: 'starting', model: 'battery-soh-rul-random-forest', error: null };
let mlBuffer = '';
let mlRequestSequence = 0;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

function startMlWorker() {
  if (!fs.existsSync(mlWorkerPath)) {
    mlStatus = { ...mlStatus, status: 'unavailable', error: 'ML inference worker was not found' };
    return;
  }
  mlWorker = spawn(mlPython, [mlWorkerPath], { stdio: ['pipe', 'pipe', 'pipe'] });
  mlWorker.stdout.setEncoding('utf8');
  mlWorker.stdout.on('data', (chunk) => {
    mlBuffer += chunk;
    const lines = mlBuffer.split(/\r?\n/);
    mlBuffer = lines.pop() || '';
    lines.filter(Boolean).forEach((line) => {
      let message;
      try { message = JSON.parse(line); } catch { return; }
      if (message.type === 'ready') {
        mlStatus = { ...mlStatus, status: 'ready', error: null };
      } else if (message.type === 'startup_error') {
        mlStatus = { ...mlStatus, status: 'unavailable', error: message.error };
      } else if (message.requestId && pendingPredictions.has(message.requestId)) {
        const pending = pendingPredictions.get(message.requestId);
        pendingPredictions.delete(message.requestId);
        if (message.type === 'prediction') pending.resolve(message.prediction);
        else pending.reject(new Error(message.error || 'ML prediction failed'));
      }
    });
  });
  mlWorker.stderr.setEncoding('utf8');
  mlWorker.stderr.on('data', (chunk) => console.error(`[ML worker] ${chunk.trim()}`));
  mlWorker.on('error', (error) => {
    mlStatus = { ...mlStatus, status: 'unavailable', error: error.message };
  });
  mlWorker.on('exit', (code) => {
    mlStatus = { ...mlStatus, status: 'unavailable', error: code ? `ML worker exited (${code})` : 'ML worker stopped' };
    pendingPredictions.forEach(({ reject }) => reject(new Error(mlStatus.error)));
    pendingPredictions.clear();
  });
}

function requestMlPrediction(battery, reading) {
  if (!mlWorker || mlWorker.killed || mlStatus.status === 'unavailable') {
    return Promise.reject(new Error(mlStatus.error || 'ML model is not ready'));
  }
  const requestId = `ML-${++mlRequestSequence}`;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingPredictions.delete(requestId);
      reject(new Error('ML prediction timed out'));
    }, 15000);
    pendingPredictions.set(requestId, {
      resolve: (prediction) => { clearTimeout(timeout); resolve(prediction); },
      reject: (error) => { clearTimeout(timeout); reject(error); }
    });
    mlWorker.stdin.write(`${JSON.stringify({ requestId, batteryId: battery.id, capacityKwh: battery.capacityKwh, telemetry: reading })}\n`, (error) => {
      if (error) {
        clearTimeout(timeout);
        pendingPredictions.delete(requestId);
        reject(error);
      }
    });
  });
}

function persistPrediction(prediction) {
  if (prediction.decision === 'PUSH_ON_CHAIN') {
    prediction.recordedOnChain = true;
    appendBlock('ML_HEALTH_PREDICTION', prediction);
  } else {
    prediction.recordedOnChain = false;
    prediction.reviewReason = 'Confidence below threshold; not written to ledger';
  }
  predictions.set(prediction.batteryId, prediction);
  return prediction;
}

function evaluateTelemetryWithMl(battery, reading) {
  requestMlPrediction(battery, reading)
    .then(persistPrediction)
    .catch((error) => console.error(`[ML prediction ${battery.id}] ${error.message}`));
}

const batteries = new Map();
const telemetry = [];
const chain = [];
const listings = new Map();
const ownershipHistory = new Map();
const predictions = new Map();
const pendingPredictions = new Map();
const purchaseRequests = new Map();
startMlWorker();

function appendTelemetryCsv(reading) {
  fs.mkdirSync(dataDirectory, { recursive: true });
  if (!fs.existsSync(telemetryCsvPath)) fs.writeFileSync(telemetryCsvPath, `${telemetryCsvColumns.join(',')}\n`);
  const row = telemetryCsvColumns.map((column) => JSON.stringify(reading[column] ?? '')).join(',');
  fs.appendFileSync(telemetryCsvPath, `${row}\n`);
}

function now() {
  return new Date().toISOString();
}

function hashBlock(block) {
  const { hash, ...content } = block;
  return crypto.createHash('sha256').update(JSON.stringify(content)).digest('hex');
}

function appendBlock(type, payload) {
  const block = {
    index: chain.length,
    timestamp: now(),
    type,
    payload,
    previousHash: chain.length ? chain[chain.length - 1].hash : '0'.repeat(64)
  };
  block.hash = hashBlock(block);
  chain.push(block);
  return block;
}

function recordOwnership(batteryId, event) {
  const history = ownershipHistory.get(batteryId) || [];
  history.unshift(event);
  ownershipHistory.set(batteryId, history);
}

function healthFromTelemetry(reading) {
  if (reading.temperatureC > 45 || reading.stateOfHealth < 70 || reading.voltage < 300) return 'critical';
  if (reading.temperatureC > 38 || reading.stateOfHealth < 85 || reading.soc < 20) return 'warning';
  return 'healthy';
}

function generateTelemetry(batteryId) {
  const battery = batteries.get(batteryId);
  if (!battery) return null;
  const previous = battery.latestTelemetry;
  const ageHours = (Date.now() - new Date(battery.manufacturedAt).getTime()) / 36e5;
  const drift = Math.min(ageHours / 10000, 0.12);
  const reading = {
    id: crypto.randomUUID(),
    batteryId,
    timestamp: now(),
    soc: Number(Math.max(8, Math.min(98, (previous?.soc ?? 76) + (Math.random() * 6 - 3))).toFixed(1)),
    voltage: Number((previous?.voltage ?? 365 + Math.random() * 8 + 2).toFixed(2)),
    current: Number((Math.random() * 50 - 25).toFixed(2)),
    temperatureC: Number((previous?.temperatureC ?? 27 + Math.random() * 4 + drift * 10).toFixed(2)),
    stateOfHealth: Number(Math.max(65, 100 - drift * 100 - Math.random() * 4).toFixed(1)),
    cycleCount: battery.cycleCount + Math.floor(Math.random() * 2),
    location: battery.location,
    source: 'simulated-iot'
  };
  reading.status = healthFromTelemetry(reading);
  battery.cycleCount = reading.cycleCount;
  battery.latestTelemetry = reading;
  telemetry.unshift(reading);
  if (telemetry.length > 500) telemetry.pop();
  appendTelemetryCsv(reading);
  appendBlock('TELEMETRY_RECORDED', reading);
  evaluateTelemetryWithMl(battery, reading);
  return reading;
}

function seed() {
  const battery = {
    id: 'EVBP-001',
    serialNumber: 'BAT-2026-0001',
    chemistry: 'NMC',
    capacityKwh: 82.4,
    manufacturer: 'Northstar Cells',
    owner: 'Apex Mobility',
    location: 'Austin, TX',
    manufacturedAt: '2024-05-18T09:00:00.000Z',
    cycleCount: 642,
    latestTelemetry: null
  };
  batteries.set(battery.id, battery);
  recordOwnership(battery.id, { timestamp: battery.manufacturedAt, owner: battery.owner, type: 'REGISTERED', transactionId: null });
  appendBlock('BATTERY_REGISTERED', { ...battery, latestTelemetry: undefined });
  generateTelemetry(battery.id);
}
seed();

function seedMarketplaceInventory() {
  const seededBatteries = [
    { id: 'EVBP-002', serialNumber: 'BAT-2025-0142', chemistry: 'LFP', capacityKwh: 60, manufacturer: 'Volta Forge', owner: 'GreenRoute Logistics', location: 'Dallas, TX', manufacturedAt: '2023-11-02T09:00:00.000Z', cycleCount: 811 },
    { id: 'EVBP-003', serialNumber: 'BAT-2025-0218', chemistry: 'NMC', capacityKwh: 74.5, manufacturer: 'Northstar Cells', owner: 'Metro Charge', location: 'Houston, TX', manufacturedAt: '2024-02-14T09:00:00.000Z', cycleCount: 486 },
    { id: 'EVBP-004', serialNumber: 'BAT-2026-0037', chemistry: 'LMO', capacityKwh: 48, manufacturer: 'IonWorks', owner: 'Apex Mobility', location: 'San Antonio, TX', manufacturedAt: '2024-08-21T09:00:00.000Z', cycleCount: 275 }
  ];
  seededBatteries.forEach((battery) => {
    batteries.set(battery.id, { ...battery, latestTelemetry: null });
    recordOwnership(battery.id, { timestamp: battery.manufacturedAt, owner: battery.owner, type: 'REGISTERED', transactionId: null });
    appendBlock('BATTERY_REGISTERED', { ...battery, latestTelemetry: undefined });
    generateTelemetry(battery.id);
  });

  [
    { batteryId: 'EVBP-002', seller: 'GreenRoute Logistics', price: 8900, title: 'Long-range LFP fleet pack', condition: 'Used - good', warrantyMonths: 8, description: 'Durable LFP pack with strong cycle life and verified telemetry.' },
    { batteryId: 'EVBP-003', seller: 'Metro Charge', price: 11750, title: '74.5 kWh NMC replacement pack', condition: 'Used - excellent', warrantyMonths: 14, description: 'High-capacity replacement pack with a clean digital passport.' }
  ].forEach((details) => {
    const battery = batteries.get(details.batteryId);
    const listing = { ...details, id: `LIST-${crypto.randomUUID().slice(0, 8).toUpperCase()}`, location: battery.location, status: 'ACTIVE', createdAt: now(), battery: { serialNumber: battery.serialNumber, chemistry: battery.chemistry, capacityKwh: battery.capacityKwh, manufacturer: battery.manufacturer, cycleCount: battery.cycleCount, latestTelemetry: battery.latestTelemetry } };
    listings.set(listing.id, listing);
    appendBlock('BATTERY_LISTED', { listingId: listing.id, batteryId: listing.batteryId, seller: listing.seller, price: listing.price, timestamp: listing.createdAt });
  });
}
seedMarketplaceInventory();

app.get('/api/health', (_req, res) => res.json({ ok: true, storage: useMongo ? 'mongodb-configured' : 'local-memory', nextTelemetryInSeconds: 30 }));
app.get('/api/ml/status', (_req, res) => res.json(mlStatus));
app.get('/api/batteries/:id/prediction', (req, res) => {
  if (!batteries.has(req.params.id)) return res.status(404).json({ error: 'Battery not found' });
  const prediction = predictions.get(req.params.id);
  if (!prediction) return res.status(202).json({ ...mlStatus, prediction: null, message: 'No ML prediction is available yet' });
  res.json(prediction);
});
app.post('/api/batteries/:id/predict', async (req, res) => {
  const battery = batteries.get(req.params.id);
  if (!battery) return res.status(404).json({ error: 'Battery not found' });
  if (!battery.latestTelemetry) return res.status(409).json({ error: 'Battery has no telemetry to evaluate' });
  try {
    const prediction = persistPrediction(await requestMlPrediction(battery, battery.latestTelemetry));
    res.json(prediction);
  } catch (error) {
    res.status(503).json({ error: error.message, mlStatus });
  }
});
app.get('/api/batteries', (_req, res) => res.json([...batteries.values()]));
app.get('/api/batteries/:id', (req, res) => {
  const battery = batteries.get(req.params.id);
  if (!battery) return res.status(404).json({ error: 'Battery not found' });
  res.json(battery);
});
app.get('/api/batteries/:id/telemetry', (req, res) => {
  const limit = Math.min(Number(req.query.limit) || 30, 500);
  res.json(telemetry.filter((item) => item.batteryId === req.params.id).slice(0, limit));
});
app.post('/api/batteries', (req, res) => {
  const { serialNumber, chemistry = 'NMC', capacityKwh, manufacturer, owner, location } = req.body;
  if (!serialNumber || !capacityKwh || !owner) return res.status(400).json({ error: 'serialNumber, capacityKwh, and owner are required' });
  const battery = { id: `EVBP-${String(batteries.size + 1).padStart(3, '0')}`, serialNumber, chemistry, capacityKwh: Number(capacityKwh), manufacturer: manufacturer || 'Unknown', owner, location: location || 'Unknown', manufacturedAt: now(), cycleCount: 0, latestTelemetry: null };
  batteries.set(battery.id, battery);
  recordOwnership(battery.id, { timestamp: battery.manufacturedAt, owner: battery.owner, type: 'REGISTERED', transactionId: null });
  appendBlock('BATTERY_REGISTERED', { ...battery, latestTelemetry: undefined });
  generateTelemetry(battery.id);
  res.status(201).json(battery);
});
app.post('/api/batteries/:id/telemetry', (req, res) => {
  const battery = batteries.get(req.params.id);
  if (!battery) return res.status(404).json({ error: 'Battery not found' });
  const reading = generateTelemetry(req.params.id);
  Object.assign(reading, req.body, { id: reading.id, batteryId: req.params.id, timestamp: reading.timestamp, source: req.body.source || 'manual-iot' });
  reading.status = healthFromTelemetry(reading);
  battery.latestTelemetry = reading;
  telemetry[0] = reading;
  appendTelemetryCsv(reading);
  appendBlock('TELEMETRY_UPDATED', reading);
  evaluateTelemetryWithMl(battery, reading);
  res.status(201).json(reading);
});
app.post('/api/batteries/:id/transfer', (req, res) => {
  const battery = batteries.get(req.params.id);
  if (!battery || !req.body.newOwner) return res.status(400).json({ error: 'Battery and newOwner are required' });
  const previousOwner = battery.owner;
  battery.owner = req.body.newOwner;
  const transactionId = `TX-${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
  const transfer = { batteryId: battery.id, previousOwner, newOwner: battery.owner, reason: req.body.reason || 'private transfer', transactionId, timestamp: now() };
  recordOwnership(battery.id, { timestamp: transfer.timestamp, owner: battery.owner, previousOwner, type: 'TRANSFERRED', transactionId });
  const block = appendBlock('OWNERSHIP_TRANSFERRED', transfer);
  res.json({ battery, block });
});
app.get('/api/batteries/:id/ownership-history', (req, res) => res.json(ownershipHistory.get(req.params.id) || []));
app.get('/api/marketplace/listings', (_req, res) => res.json([...listings.values()].filter((listing) => ['ACTIVE', 'BOOKED'].includes(listing.status))));
app.post('/api/marketplace/listings', (req, res) => {
  const { batteryId, seller, price, title, description, condition = 'Used - good', location, warrantyMonths = 0 } = req.body;
  const battery = batteries.get(batteryId);
  if (!battery) return res.status(404).json({ error: 'Battery not found' });
  if (!seller || seller !== battery.owner) return res.status(400).json({ error: 'Seller must match the current battery owner' });
  if (!price || Number(price) <= 0 || !title) return res.status(400).json({ error: 'title and a positive price are required' });
  if ([...listings.values()].some((listing) => listing.batteryId === batteryId && listing.status === 'ACTIVE')) return res.status(409).json({ error: 'Battery is already listed for sale' });
  const listing = { id: `LIST-${crypto.randomUUID().slice(0, 8).toUpperCase()}`, batteryId, seller, price: Number(price), title, description: description || 'Verified EV battery with digital passport.', condition, location: location || battery.location, warrantyMonths: Number(warrantyMonths), status: 'ACTIVE', createdAt: now(), battery: { serialNumber: battery.serialNumber, chemistry: battery.chemistry, capacityKwh: battery.capacityKwh, manufacturer: battery.manufacturer, cycleCount: battery.cycleCount, latestTelemetry: battery.latestTelemetry } };
  listings.set(listing.id, listing);
  appendBlock('BATTERY_LISTED', { listingId: listing.id, batteryId, seller, price: listing.price, timestamp: listing.createdAt });
  res.status(201).json(listing);
});
app.delete('/api/marketplace/listings/:id', (req, res) => {
  const listing = listings.get(req.params.id);
  if (!listing || listing.status !== 'ACTIVE') return res.status(404).json({ error: 'Active listing not found' });
  listing.status = 'CANCELLED';
  listing.cancelledAt = now();
  res.json(listing);
});
app.post('/api/marketplace/listings/:id/purchase', (req, res) => {
  const listing = listings.get(req.params.id);
  if (!listing || listing.status !== 'ACTIVE') return res.status(409).json({ error: 'This battery is unavailable or already booked' });
  const { buyer, buyerEmail = '' } = req.body;
  if (!buyer || buyer === listing.seller) return res.status(400).json({ error: 'A different buyer name is required' });
  if (!buyerEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(buyerEmail)) return res.status(400).json({ error: 'A valid buyer email is required' });
  const request = { id: `REQ-${crypto.randomUUID().slice(0, 8).toUpperCase()}`, listingId: listing.id, batteryId: listing.batteryId, buyer: buyer.trim(), buyerEmail: buyerEmail.trim().toLowerCase(), seller: listing.seller, price: listing.price, status: 'PENDING', requestedAt: now() };
  purchaseRequests.set(request.id, request);
  listing.status = 'BOOKED';
  listing.requestId = request.id;
  listing.bookedAt = request.requestedAt;
  appendBlock('PURCHASE_REQUESTED', { requestId: request.id, listingId: listing.id, batteryId: listing.batteryId, buyer: request.buyer, buyerEmail: request.buyerEmail, seller: request.seller, timestamp: request.requestedAt });
  res.status(201).json({ request, listing, message: 'Booking request sent to the seller. Ownership will change only if the seller accepts.' });
});
app.get('/api/marketplace/seller-requests', (req, res) => {
  const seller = String(req.query.seller || '').trim().toLowerCase();
  if (!seller) return res.status(400).json({ error: 'Seller name is required' });
  res.json([...purchaseRequests.values()].filter((request) => request.seller.toLowerCase() === seller).map((request) => ({ ...request, listing: listings.get(request.listingId), battery: batteries.get(request.batteryId) })));
});
app.get('/api/marketplace/buyer-orders', (req, res) => {
  const email = String(req.query.email || '').trim().toLowerCase();
  if (!email) return res.status(400).json({ error: 'Buyer email is required' });
  res.json([...purchaseRequests.values()].filter((request) => request.buyerEmail === email).map((request) => ({ ...request, listing: listings.get(request.listingId), battery: batteries.get(request.batteryId) })));
});
app.post('/api/marketplace/requests/:id/decision', (req, res) => {
  const request = purchaseRequests.get(req.params.id);
  if (!request || request.status !== 'PENDING') return res.status(404).json({ error: 'Pending booking request not found' });
  if (!req.body.seller || req.body.seller.trim().toLowerCase() !== request.seller.toLowerCase()) return res.status(403).json({ error: 'Only the listed seller can decide this request' });
  const listing = listings.get(request.listingId);
  if (!listing || listing.requestId !== request.id || listing.status !== 'BOOKED') return res.status(409).json({ error: 'The booked listing is no longer available' });
  const decision = String(req.body.decision || '').toUpperCase();
  if (!['ACCEPT', 'REJECT'].includes(decision)) return res.status(400).json({ error: 'Decision must be ACCEPT or REJECT' });
  request.status = decision === 'ACCEPT' ? 'ACCEPTED' : 'REJECTED';
  request.decidedAt = now();
  if (decision === 'REJECT') {
    listing.status = 'ACTIVE';
    delete listing.requestId;
    delete listing.bookedAt;
    appendBlock('PURCHASE_REQUEST_REJECTED', { requestId: request.id, listingId: listing.id, batteryId: listing.batteryId, seller: request.seller, buyer: request.buyer, timestamp: request.decidedAt });
    return res.json({ request, listing, message: 'Request declined; the battery is available again.' });
  }
  const battery = batteries.get(listing.batteryId);
  const transactionId = `TX-${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
  const completedAt = request.decidedAt;
  const previousOwner = battery.owner;
  battery.owner = request.buyer;
  listing.status = 'SOLD';
  listing.buyer = request.buyer;
  listing.buyerEmail = request.buyerEmail;
  listing.completedAt = completedAt;
  request.transactionId = transactionId;
  recordOwnership(battery.id, { timestamp: completedAt, owner: request.buyer, previousOwner, type: 'MARKETPLACE_PURCHASE', transactionId });
  const block = appendBlock('OWNERSHIP_TRANSFERRED', { transactionId, requestId: request.id, listingId: listing.id, batteryId: battery.id, previousOwner, newOwner: request.buyer, buyerEmail: request.buyerEmail, price: listing.price, timestamp: completedAt });
  res.json({ request, listing, battery, transactionId, block, certificateUrl: `/api/marketplace/certificates/${transactionId}.pdf` });
});
app.get('/api/marketplace/certificates/:transactionId.pdf', (req, res) => {
  const request = [...purchaseRequests.values()].find((item) => item.transactionId === req.params.transactionId && item.status === 'ACCEPTED');
  if (!request) return res.status(404).json({ error: 'Completed purchase certificate not found' });
  const battery = batteries.get(request.batteryId);
  const history = ownershipHistory.get(request.batteryId) || [];
  const transfer = history.find((event) => event.transactionId === request.transactionId);
  const listing = listings.get(request.listingId);
  const certificateNumber = `VP-${request.transactionId}`;
  const lines = [
    'VOLTPASS | BATTERY OWNERSHIP CERTIFICATE',
    '',
    `Certificate: ${certificateNumber}`,
    `Transaction: ${request.transactionId}`,
    `Transfer date: ${request.decidedAt}`,
    '',
    'OWNERSHIP TRANSFER',
    `Previous owner: ${transfer?.previousOwner || request.seller}`,
    `New owner: ${request.buyer}`,
    `Buyer email: ${request.buyerEmail}`,
    `Seller: ${request.seller}`,
    `Sale price: USD ${Number(request.price).toLocaleString('en-US')}`,
    '',
    'BATTERY PASSPORT',
    `Asset ID: ${battery.id}`,
    `Serial number: ${battery.serialNumber}`,
    `Chemistry: ${battery.chemistry}`,
    `Capacity: ${battery.capacityKwh} kWh`,
    `Manufacturer: ${battery.manufacturer}`,
    `Listing: ${listing.title}`,
    `Ledger block: ${chain.find((block) => block.type === 'OWNERSHIP_TRANSFERRED' && block.payload.transactionId === request.transactionId)?.index ?? 'N/A'}`,
    '',
    'This certificate records a completed transfer in the VoltPass digital passport ledger.',
    'Keep this document with the battery service and ownership records.'
  ];
  const escapePdf = (text) => text.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
  const commands = ['BT', '/F1 10 Tf', '50 790 Td', '14 TL', ...lines.flatMap((line, index) => index === 0 ? [`(${escapePdf(line)}) Tj`] : ['T*', `(${escapePdf(line)}) Tj`]), 'ET'].join('\n');
  const stream = Buffer.from(commands, 'latin1');
  const objects = [
    Buffer.from('<< /Type /Catalog /Pages 2 0 R >>', 'latin1'),
    Buffer.from('<< /Type /Pages /Kids [3 0 R] /Count 1 >>', 'latin1'),
    Buffer.from('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>', 'latin1'),
    Buffer.from('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>', 'latin1'),
    Buffer.concat([Buffer.from(`<< /Length ${stream.length} >>\nstream\n`, 'latin1'), stream, Buffer.from('\nendstream', 'latin1')])
  ];
  const chunks = [Buffer.from('%PDF-1.4\n', 'latin1')];
  const offsets = [0];
  let offset = chunks[0].length;
  objects.forEach((object, index) => {
    offsets.push(offset);
    const wrapped = Buffer.concat([Buffer.from(`${index + 1} 0 obj\n`, 'latin1'), object, Buffer.from('\nendobj\n', 'latin1')]);
    chunks.push(wrapped);
    offset += wrapped.length;
  });
  const xrefOffset = offset;
  const xref = [`xref\n0 ${objects.length + 1}\n`, '0000000000 65535 f \n', ...offsets.slice(1).map((entry) => `${String(entry).padStart(10, '0')} 00000 n \n`)].join('');
  chunks.push(Buffer.from(`${xref}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`, 'latin1'));
  const pdf = Buffer.concat(chunks);
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename="voltpass-${request.transactionId}-certificate.pdf"`);
  res.send(pdf);
});
app.get('/api/chain', (_req, res) => res.json({ valid: chain.every((block, index) => block.hash === hashBlock(block) && (index === 0 ? block.previousHash === '0'.repeat(64) : block.previousHash === chain[index - 1].hash)), length: chain.length, blocks: chain.slice().reverse() }));

setInterval(() => batteries.forEach((_battery, id) => generateTelemetry(id)), TELEMETRY_INTERVAL_MS);
app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.listen(PORT, () => console.log(`EV Battery Passport running at http://localhost:${PORT}`));
