# VoltPass — EV Battery Passport POC

A runnable prototype for a realtime EV battery passport. It generates one IoT telemetry reading every 30 seconds, appends every reading to `data/telemetry.csv`, and stores the same event in a backend-only hash-linked ledger. The dashboard displays battery state and health trends without exposing simulation controls or blockchain internals.

## Run

```powershell
npm install
npm start
```

Open `http://localhost:3000`.

Open `http://localhost:3000/marketplace.html` for the VoltPass battery marketplace. Owners can publish a battery listing with price, condition, location, warranty, and description. Buyers submit a booking request with their name and email. The listing remains visible with a booked status while the seller reviews it. Only seller acceptance completes the sale, transfers ownership, and records the transaction in the hash-linked ledger. Buyers can view pending and completed bookings on the passport dashboard using the same email; completed transfers include a downloadable PDF ownership certificate.

## Deploy a live demo

The root `render.yaml` configures a Render Node web service, installs the Python ML dependencies into `.venv`, and uses `/api/health` for health checks. Push this repository to GitHub, sign in to Render, choose **New → Blueprint**, connect the repository, and apply the Blueprint. Render will provide the final public `onrender.com` URL after the first successful deployment.

This Blueprint uses Render's free plan for a POC. Free services can sleep when idle, and local in-memory marketplace bookings, ownership, and ledger events reset on restarts/redeploys. Use persistent storage/database and seller authentication before treating the public demo as a production marketplace.

The demo starts with four batteries. Two are already listed for sale, one remains unlisted for seller testing, and the original passport battery remains available for inspection.

## ML model and ledger integration

The supplied archive is extracted under `ml_model/files(1)`. Its serialized Random Forest models predict battery state of health (SOH) and remaining useful life (RUL). The Node.js server starts a persistent Python inference worker when it starts and evaluates each new telemetry reading. The dashboard displays the latest prediction, confidence, reuse guidance, and whether the confidence gate allowed the prediction onto the hash-linked ledger.

Install the Python dependencies listed in `requirements.txt` into the Python environment used by the server. The included model artifacts were serialized with scikit-learn 1.7.0. If Python is not on `PATH`, set `ML_PYTHON` to its executable path. The model's confidence threshold is 0.90; predictions below it remain review-only and are not appended to the ledger. Use the dashboard's **Re-run prediction** action to evaluate the latest reading on demand.

**Model limitation:** the bundled training dataset is synthetic, and the live pack telemetry lacks several cell-level model features. The inference worker therefore supplies documented proxies for missing features. Treat these predictions as a POC only; validate on representative, calibrated battery data before operational use. The current blockchain is an in-memory hash-linked demo ledger, not an external or persistent blockchain network.

## API

- `GET /api/batteries`
- `GET /api/batteries/:id`
- `GET /api/batteries/:id/telemetry?limit=30`
- `GET /api/batteries/:id/prediction`
- `POST /api/batteries/:id/predict`
- `GET /api/ml/status`
- `POST /api/batteries/:id/telemetry`
- `POST /api/batteries/:id/transfer` with `{ "newOwner": "..." }`
- `GET /api/batteries/:id/ownership-history`
- `GET /api/marketplace/listings`
- `POST /api/marketplace/listings`
- `DELETE /api/marketplace/listings/:id`
- `POST /api/marketplace/listings/:id/purchase`
- `GET /api/marketplace/seller-requests?seller=...`
- `GET /api/marketplace/buyer-orders?email=...`
- `POST /api/marketplace/requests/:id/decision` with `{ "seller": "...", "decision": "ACCEPT" }` or `REJECT`
- `GET /api/marketplace/certificates/:transactionId.pdf`
- `GET /api/chain`

The prototype uses local memory for the current process and CSV for telemetry export. Set `MONGODB_URI` before starting to mark MongoDB as configured; the next production step is replacing the in-memory Maps and ledger with Mongoose models while keeping the REST contracts unchanged. The dashboard loads React 18 from its UMD runtime and keeps API polling/state in `public/app.js`.
