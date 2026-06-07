type ModelContributions = {
  mobilenetv3: number;
  efficientnetb2: number;
  densenet121: number;
};

export type PredictResponse = {
  diagnosis: string;
  risk_score: number;
  confidence_label: string;
  heatmap: string;
  model_contributions?: ModelContributions;
};

export type BatchItemStatus = 'queued' | 'processing' | 'done' | 'error';

export type BatchPredictItem = {
  id: string;
  fileName: string;
  localPreviewUrl: string | null;
  status: BatchItemStatus;
  result: PredictResponse | null;
  error: string | null;
};

export type PredictUiState = {
  loading: boolean;
  error: string | null;
  data: PredictResponse | null; // selected item's result (for backwards compatibility)
  items: BatchPredictItem[];
  currentIndex: number;
};

export type UploadRecord = {
  downloadURL: string;
  fileName: string;
  uploadedAt: string;
};

/**
 * Production API host (HTTPS). Browser calls this directly — long `/predict` cannot go through
 * Vercel `vercel.json` rewrites (edge proxy times out in ~seconds–1min while inference runs minutes).
 * Override with VITE_CHEXIT_API_URL. Requires DNS `api` → droplet + TLS (e.g. Nginx + Certbot).
 */
export const CHEXIT_DEFAULT_API_ORIGIN = 'https://api.chexit.app';

/**
 * Same-origin `/api/*` — only for local dev (Vite proxy) or explicit `VITE_USE_RELATIVE_API=1`
 * (e.g. short requests / preview). Do not rely on Vercel → DO rewrites for multi-minute `/predict`.
 */
function canUseViteApiProxy(): boolean {
  if (import.meta.env.VITE_CHEXIT_API_URL?.trim()) {
    return false;
  }
  if (import.meta.env.VITE_USE_RELATIVE_API === '1') {
    return true;
  }
  if (import.meta.env.DEV) {
    return true;
  }
  if (typeof window === 'undefined') {
    return false;
  }
  const h = window.location.hostname;
  return h === 'localhost' || h === '127.0.0.1' || h === '[::1]';
}

function resolvedApiOrigin(): string {
  const trimmed = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  if (trimmed) {
    return trimmed.replace(/\/$/, '');
  }
  return CHEXIT_DEFAULT_API_ORIGIN;
}

function predictUrl(): string {
  if (canUseViteApiProxy()) {
    return '/api/predict';
  }
  return `${resolvedApiOrigin()}/predict`;
}

function uploadUrl(): string {
  if (canUseViteApiProxy()) {
    return '/api/upload';
  }
  return `${resolvedApiOrigin()}/upload`;
}

function latestUploadUrl(): string {
  if (canUseViteApiProxy()) {
    return '/api/uploads/latest';
  }
  return `${resolvedApiOrigin()}/uploads/latest`;
}

function toAbsoluteApiUrl(urlOrPath: string): string {
  if (/^https?:\/\//i.test(urlOrPath)) {
    return urlOrPath;
  }
  const base = canUseViteApiProxy() ? window.location.origin : resolvedApiOrigin();
  return `${base}${urlOrPath.startsWith('/') ? '' : '/'}${urlOrPath}`;
}

function apiLabelForErrors(): string {
  if (canUseViteApiProxy()) {
    return import.meta.env.DEV ? '/api (Vite → :8000)' : '/api (same origin → backend)';
  }
  return resolvedApiOrigin();
}

function parseErrorDetail(body: unknown): string {
  if (!body || typeof body !== 'object') return 'Request failed';
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => (typeof e === 'object' && e && 'msg' in e ? String((e as { msg: string }).msg) : String(e)))
      .join(', ');
  }
  return 'Request failed';
}

function networkErrorHint(label: string): string {
  const httpsPage =
    typeof window !== 'undefined' && window.location.protocol === 'https:';
  const directHttp = (import.meta.env.VITE_CHEXIT_API_URL?.trim() ?? '').startsWith(
    'http://',
  );
  if (httpsPage && directHttp) {
    return (
      `Cannot call ${label} from an HTTPS page (mixed content). ` +
      `Remove VITE_CHEXIT_API_URL from .env so dev uses the Vite /api proxy, or use an https:// API URL, or open the app at http://localhost:5173.`
    );
  }
  return (
    `Cannot reach the API (${label}). ` +
      `If testing locally: run FastAPI on port 8000 and use npm run dev so /api proxies. ` +
      `If using the hosted API: open ${CHEXIT_DEFAULT_API_ORIGIN}/health in a browser. ` +
      `Override the API host with VITE_CHEXIT_API_URL. Avoid uvicorn --reload during long /predict.`
  );
}

/** Score-CAM + TF can run several minutes; browsers rarely cancel, but proxies might. */
const PREDICT_TIMEOUT_MS = 10 * 60 * 1000;

function predictAbortSignal(): AbortSignal {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(PREDICT_TIMEOUT_MS);
  }
  const c = new AbortController();
  setTimeout(() => c.abort(), PREDICT_TIMEOUT_MS);
  return c.signal;
}

function isAbortError(e: unknown): boolean {
  if (e instanceof DOMException && e.name === 'AbortError') {
    return true;
  }
  return e instanceof Error && e.name === 'AbortError';
}

function logClient(stage: string, detail?: Record<string, unknown>): void {
  if (import.meta.env.PROD) {
    return;
  }
  const ts = new Date().toISOString();
  if (detail) {
    console.log(`[Chexit ${ts}]`, stage, detail);
  } else {
    console.log(`[Chexit ${ts}]`, stage);
  }
}

/**
 * Extra bytes around the file in multipart/form-data (boundaries, disposition, CRLFs).
 * Browsers vary; use Network `Content-Length` when you need an exact upload total.
 */
const PREDICT_MULTIPART_META_BYTES_ESTIMATE = 900;

function logPredictBandwidthSummary(args: {
  file: File;
  bodyText: string;
  heatmapBase64: string;
  response: Response;
  predictUrl: string;
}): void {
  const { file, bodyText, heatmapBase64, response, predictUrl } = args;
  const meta = PREDICT_MULTIPART_META_BYTES_ESTIMATE;
  const resolvedUrl = response.url || predictUrl;

  let resourceTiming: Record<string, unknown> | null = null;
  if (typeof performance !== 'undefined' && typeof performance.getEntriesByName === 'function') {
    const entries = performance.getEntriesByName(resolvedUrl, 'resource') as PerformanceResourceTiming[];
    const last = entries[entries.length - 1];
    if (last) {
      resourceTiming = {
        transferSize: last.transferSize,
        encodedBodySize: last.encodedBodySize,
        decodedBodySize: last.decodedBodySize,
        nextHopProtocol: last.nextHopProtocol,
      };
    }
  }

  const cl = response.headers.get('content-length');
  logClient('predict: bandwidth summary (dev)', {
    note:
      'Upload totals: use Network Content-Length or upload_imageBytes+upload_approxMultipartMetaBytes. ' +
      'Response: response_bodyChars is full JSON text length (ASCII ≈ bytes). ' +
      'resourceTiming is response-focused (decodedBodySize ≈ JSON bytes when not compressed).',
    upload_imageBytes: file.size,
    upload_approxMultipartMetaBytes: meta,
    upload_approxTotalBytes: file.size + meta,
    response_contentLengthHeader: cl ? Number(cl) : null,
    response_bodyChars: bodyText.length,
    response_heatmapBase64Chars: heatmapBase64.length,
    response_approxJsonWithoutHeatmapStringChars: bodyText.length - heatmapBase64.length,
    resourceTiming,
  });
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const url = predictUrl();
  const label = apiLabelForErrors();
  logClient('predict: starting request', {
    url,
    fileName: file.name,
    fileSizeBytes: file.size,
    fileType: file.type,
  });

  const formData = new FormData();
  formData.append('file', file);

  const t0 = performance.now();
  let res: Response;
  try {
    logClient('predict: fetch POST /predict (server runs U-Net → MobileNet → Score-CAM; may take several minutes)');
    res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: predictAbortSignal(),
    });
    logClient('predict: response headers received', {
      status: res.status,
      ok: res.ok,
      elapsedSec: Math.round((performance.now() - t0) / 1000),
    });
  } catch (e) {
    logClient('predict: fetch failed', {
      error: e instanceof Error ? e.message : String(e),
      elapsedSec: Math.round((performance.now() - t0) / 1000),
    });
    if (isAbortError(e)) {
      throw new Error(
        `Analyze timed out after ${Math.round(PREDICT_TIMEOUT_MS / 60000)} minutes, or the request was cancelled. ` +
          `Try a smaller image, or run the API without --reload (see chexit-backend/run_dev.sh).`,
      );
    }
    const failedFetch =
      e instanceof TypeError &&
      (e.message === 'Failed to fetch' || e.message.includes('Load failed'));
    if (failedFetch) {
      throw new Error(networkErrorHint(label));
    }
    throw e;
  }

  const bodyText = await res.text();
  const contentType = res.headers.get('content-type') ?? '';
  if (
    !res.ok &&
    url.startsWith('/api') &&
    (res.status === 502 || res.status === 503 || res.status === 504)
  ) {
    throw new Error(
      `HTTP ${res.status}: The /api proxy (e.g. Vercel → your droplet) timed out. ` +
        `Inference can take several minutes; use the direct API host instead: HTTPS on api.yourdomain.com, ` +
        `set VITE_CHEXIT_API_URL at build time, and CHEXIT_CORS_ORIGINS on the server for https://chexit.app.`,
    );
  }
  logClient('predict: response body received', {
    status: res.status,
    ok: res.ok,
    contentType,
    bodyChars: bodyText.length,
    elapsedSec: Math.round((performance.now() - t0) / 1000),
  });

  if (!res.ok && res.status === 405) {
    throw new Error(
      'HTTP 405: POST was rejected. If the URL is /api/predict on Vercel, the SPA rewrite caught it — unset VITE_USE_RELATIVE_API or use the default Render API. ' +
        'Locally use npm run dev so /api proxies to port 8000.',
    );
  }

  let raw: unknown;
  try {
    raw = bodyText.length ? JSON.parse(bodyText) : null;
  } catch {
    const looksHtml =
      bodyText.trimStart().toLowerCase().startsWith('<!') ||
      bodyText.trimStart().toLowerCase().startsWith('<html');
    const hint = looksHtml
      ? `The server returned HTML instead of JSON (often a 404/SPA page). Use npm run dev for /api proxy, or expect JSON from ${CHEXIT_DEFAULT_API_ORIGIN}/predict.`
      : 'The response was not valid JSON (connection cut, proxy error, or wrong endpoint).';
    logClient('predict: JSON.parse failed', {
      hint,
      snippet: bodyText.slice(0, 200).replace(/\s+/g, ' '),
    });
    throw new Error(`${hint} First chars: ${bodyText.slice(0, 140).replace(/\s+/g, ' ')}`);
  }

  if (!res.ok) {
    let message = res.statusText;
    if (raw && typeof raw === 'object') {
      try {
        message = parseErrorDetail(raw);
      } catch {
        /* keep statusText */
      }
    }
    logClient('predict: HTTP error', { message });
    throw new Error(message || `Request failed (${res.status})`);
  }

  const out = normalizePredictResponse(raw);
  logClient('predict: success', {
    diagnosis: out.diagnosis,
    risk_score: out.risk_score,
    confidence_label: out.confidence_label,
    heatmapBase64Chars: out.heatmap.length,
    totalElapsedSec: Math.round((performance.now() - t0) / 1000),
  });
  logPredictBandwidthSummary({
    file,
    bodyText,
    heatmapBase64: out.heatmap,
    response: res,
    predictUrl: url,
  });
  return out;
}

export async function uploadImage(file: File): Promise<UploadRecord> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(uploadUrl(), {
    method: 'POST',
    body: formData,
  });
  const raw = (await res.json()) as Record<string, unknown>;
  if (!res.ok) {
    throw new Error(parseErrorDetail(raw) || `Upload failed (${res.status})`);
  }
  const downloadURL = pickStr(raw, 'download_url');
  const fileName = pickStr(raw, 'file_name');
  const uploadedAt = pickStr(raw, 'uploaded_at');
  if (!downloadURL || !fileName || !uploadedAt) {
    throw new Error('Invalid upload response from API.');
  }
  return {
    downloadURL: toAbsoluteApiUrl(downloadURL),
    fileName,
    uploadedAt,
  };
}

export async function fetchLatestUpload(): Promise<UploadRecord | null> {
  const res = await fetch(latestUploadUrl(), { method: 'GET' });
  if (res.status === 404) {
    return null;
  }
  const raw = (await res.json()) as Record<string, unknown>;
  if (!res.ok) {
    throw new Error(parseErrorDetail(raw) || `Failed to fetch latest upload (${res.status})`);
  }
  const downloadURL = pickStr(raw, 'downloadURL');
  const fileName = pickStr(raw, 'fileName');
  const uploadedAt = pickStr(raw, 'uploadedAt');
  if (!downloadURL) {
    return null;
  }
  return {
    downloadURL: toAbsoluteApiUrl(downloadURL),
    fileName: fileName || 'image',
    uploadedAt: uploadedAt || '',
  };
}

export function createBatchItems(files: File[]): BatchPredictItem[] {
  return files.map((file, idx) => {
    const isDicom =
      file.type === 'application/dicom' ||
      file.type === 'application/octet-stream' ||
      /\.(dcm|dicom)$/i.test(file.name);
    return {
      id: `${Date.now()}-${idx}-${file.name}`,
      fileName: file.name,
      localPreviewUrl: isDicom ? null : URL.createObjectURL(file),
      status: 'queued',
      result: null,
      error: null,
    };
  });
}

export function releaseBatchPreviewUrls(items: BatchPredictItem[]): void {
  items.forEach((item) => {
    if (item.localPreviewUrl) {
      URL.revokeObjectURL(item.localPreviewUrl);
    }
  });
}

export async function predictImagesSequential(
  files: File[],
  onItemUpdate: (nextItems: BatchPredictItem[], currentIndex: number) => void,
): Promise<BatchPredictItem[]> {
  const items = createBatchItems(files);
  // `viewIndex` tracks which image the dashboard should show. It only advances
  // when an image finishes — kicking off the *next* image's analysis must not
  // yank the view forward, so users can inspect the just-completed result.
  let viewIndex = 0;
  onItemUpdate([...items], viewIndex);
  for (let i = 0; i < files.length; i += 1) {
    items[i] = { ...items[i], status: 'processing', error: null };
    onItemUpdate([...items], viewIndex);
    try {
      const result = await predictImage(files[i]);
      items[i] = { ...items[i], status: 'done', result, error: null };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analyze failed';
      items[i] = { ...items[i], status: 'error', result: null, error: message };
    }
    viewIndex = i;
    onItemUpdate([...items], viewIndex);
  }
  return items;
}

function pickStr(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (v != null && String(v).trim() !== '') {
      return String(v);
    }
  }
  return '';
}

function pickNum(o: Record<string, unknown>, ...keys: string[]): number {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === 'number' && Number.isFinite(v)) {
      return v;
    }
    if (typeof v === 'string' && v.trim() !== '') {
      const n = Number(v);
      if (Number.isFinite(n)) {
        return n;
      }
    }
  }
  return NaN;
}

function clampPct(n: number): number {
  return Math.max(0, Math.min(100, n));
}

function pickContributionNum(o: Record<string, unknown>, key: string): number {
  const v = o[key];
  if (typeof v === 'number' && Number.isFinite(v)) return clampPct(v);
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    if (Number.isFinite(n)) return clampPct(n);
  }
  return 0;
}

// function normalizePredictResponse(raw: unknown): PredictResponse {
//   if (!raw || typeof raw !== 'object') {
//     throw new Error('Invalid API response: expected a JSON object.');
//   }
//   const o = raw as Record<string, unknown>;
//   const diagnosis = pickStr(o, 'diagnosis', 'Diagnosis');
//   const risk_score = pickNum(o, 'risk_score', 'riskScore');
//   if (!Number.isFinite(risk_score)) {
//     throw new Error('Invalid API response: risk_score is not a number.');
//   }
//   const confidence_label = pickStr(o, 'confidence_label', 'confidenceLabel');
//   let heatmap = pickStr(o, 'heatmap', 'Heatmap');
//   if (heatmap.startsWith('data:')) {
//     const comma = heatmap.indexOf(',');
//     if (comma !== -1) {
//       heatmap = heatmap.slice(comma + 1);
//     }
//   }
//   if (!heatmap.trim()) {
//     throw new Error('Invalid API response: empty heatmap.');
//   }
//   return { diagnosis, risk_score, confidence_label, heatmap };
// }

function normalizePredictResponse(raw: unknown): PredictResponse {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Invalid API response: expected a JSON object.');
  }

  const o = raw as Record<string, unknown>;
  const diagnosis = pickStr(o, 'diagnosis', 'Diagnosis');
  const risk_score = pickNum(o, 'risk_score', 'riskScore');
  if (!Number.isFinite(risk_score)) {
    throw new Error('Invalid API response: risk_score is not a number.');
  }

  const confidence_label = pickStr(o, 'confidence_label', 'confidenceLabel');

  let heatmap = pickStr(o, 'heatmap', 'Heatmap');
  if (heatmap.startsWith('data:')) {
    const comma = heatmap.indexOf(',');
    if (comma !== -1) {
      heatmap = heatmap.slice(comma + 1);
    }
  }
  if (!heatmap.trim()) {
    throw new Error('Invalid API response: empty heatmap.');
  }

  let model_contributions: ModelContributions | undefined;
  const mcRaw = o.model_contributions;
  if (mcRaw && typeof mcRaw === 'object') {
    const mc = mcRaw as Record<string, unknown>;
    model_contributions = {
      mobilenetv3: pickContributionNum(mc, 'mobilenet-v3-large'),
      efficientnetb2: pickContributionNum(mc, 'efficientnet-b2'),
      densenet121: pickContributionNum(mc, 'densenet-121'),
    };
  }

  return { diagnosis, risk_score, confidence_label, heatmap, model_contributions };
}
