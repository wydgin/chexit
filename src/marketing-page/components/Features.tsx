import * as React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Collapse from '@mui/material/Collapse';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import ChevronLeft from '@mui/icons-material/ChevronLeft';
import ChevronRight from '@mui/icons-material/ChevronRight';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import FlagOutlinedIcon from '@mui/icons-material/FlagOutlined';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import type { Theme } from '@mui/material/styles';
import type { PredictUiState } from '../../api/chexit';
import { fetchLatestUpload, type UploadRecord } from '../../api/chexit';

type FlagCategory = 'false-positive' | 'false-negative' | 'wrong-region' | 'image-quality' | 'other';

type AnomalyFlag = {
  categories: FlagCategory[];
  note: string;
  savedAt: string;
  snapshot: {
    fileName: string;
    diagnosis: string;
    riskScore: number | null;
    confidence: string;
  };
};

type FlagMap = Record<string, AnomalyFlag>;

const FLAG_CATEGORIES: { id: FlagCategory; label: string }[] = [
  { id: 'false-positive', label: 'False positive' },
  { id: 'false-negative', label: 'False negative' },
  { id: 'wrong-region', label: 'Wrong region' },
  { id: 'image-quality', label: 'Image quality' },
  { id: 'other', label: 'Other' },
];

const FLAG_CATEGORY_IDS = FLAG_CATEGORIES.map((c) => c.id);

function categoryLabel(id: FlagCategory): string {
  return FLAG_CATEGORIES.find((c) => c.id === id)?.label ?? id;
}

function isFlagCategory(value: unknown): value is FlagCategory {
  return typeof value === 'string' && (FLAG_CATEGORY_IDS as readonly string[]).includes(value);
}

/** Accept new schema (`categories[]`) and the legacy single-`category` shape side-by-side. */
function migrateFlag(raw: unknown): AnomalyFlag | null {
  if (!raw || typeof raw !== 'object') return null;
  const f = raw as Record<string, unknown>;
  let categories: FlagCategory[] = [];
  if (Array.isArray(f.categories)) {
    categories = f.categories.filter(isFlagCategory);
  } else if (isFlagCategory(f.category)) {
    categories = [f.category];
  }
  if (categories.length === 0) return null;
  const snapshotRaw = (f.snapshot && typeof f.snapshot === 'object'
    ? (f.snapshot as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  return {
    categories,
    note: typeof f.note === 'string' ? f.note : '',
    savedAt: typeof f.savedAt === 'string' ? f.savedAt : new Date().toISOString(),
    snapshot: {
      fileName: typeof snapshotRaw.fileName === 'string' ? snapshotRaw.fileName : 'image',
      diagnosis: typeof snapshotRaw.diagnosis === 'string' ? snapshotRaw.diagnosis : '',
      riskScore:
        typeof snapshotRaw.riskScore === 'number' && Number.isFinite(snapshotRaw.riskScore)
          ? snapshotRaw.riskScore
          : null,
      confidence: typeof snapshotRaw.confidence === 'string' ? snapshotRaw.confidence : '',
    },
  };
}

type ZoomView = 'input' | 'heatmap';

type ImageZoomTriggerProps = {
  src: string;
  alt: string;
  onError?: () => void;
  onOpen: () => void;
};

const ImageZoomTrigger = React.memo(function ImageZoomTrigger({
  src,
  alt,
  onError,
  onOpen,
}: ImageZoomTriggerProps) {
  const cardBorder = 'divider';
  return (
    <Box
      component="button"
      type="button"
      onClick={onOpen}
      aria-label={`View full size: ${alt}`}
      sx={{
        display: 'block',
        width: '100%',
        height: '100%',
        p: 0,
        border: 'none',
        bgcolor: 'transparent',
        cursor: 'zoom-in',
        borderRadius: 2,
        '&:focus-visible': {
          outline: '2px solid',
          outlineColor: 'primary.main',
          outlineOffset: 2,
        },
      }}
    >
      <Box
        component="img"
        src={src}
        alt={alt}
        decoding="async"
        onError={onError}
        sx={{
          borderRadius: 2,
          bgcolor: 'background.default',
          border: '1px solid',
          borderColor: cardBorder,
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          display: 'block',
          pointerEvents: 'none',
        }}
      />
    </Box>
  );
});

/** Jet-style legend bar: blue (low) → yellow → red (high). No red at the blue end. */
function HeatmapLegendBar() {
  return (
    <Box
      aria-hidden
      sx={{
        height: 10,
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'divider',
        background:
          'linear-gradient(90deg, #1d4ed8 0%, #3b82f6 38%, #facc15 72%, #ef4444 100%)',
      }}
    />
  );
}

const HEATMAP_GUIDE_TEXT =
  'Cooler blue areas received less AI model attention; red and yellow highlight regions that most influenced the TB risk estimate. This is not a diagnosis by itself — use it together with the risk score above.';

/** Map API / legacy diagnosis strings to user-facing risk labels. */
function formatDiagnosisLabel(diagnosis: string): string {
  const d = diagnosis.trim();
  if (/tb\s*positive|^positive$/i.test(d)) return 'High risk';
  if (/tb\s*negative|^negative$/i.test(d)) return 'Low risk';
  return d;
}

function isHighRiskFromResult(
  diagnosis: string,
  riskPct: number | null,
  confidenceLabel = '',
): boolean {
  const label = formatDiagnosisLabel(diagnosis);
  if (/high/i.test(label)) return true;
  if (/low/i.test(label)) return false;
  if (/high/i.test(confidenceLabel)) return true;
  if (riskPct != null && riskPct >= 50) return true;
  return false;
}

/** Default media area — fixed size; never grows when the card border stretches. */
const RESULTS_IMAGE_MEDIA_FRAME_SX = {
  width: '100%',
  aspectRatio: '1 / 1',
  maxHeight: { xs: '52vh', md: '56vh' },
  flexShrink: 0,
  flexGrow: 0,
  contain: 'layout',
} as const;

/**
 * When model contributions expand: image card borders match diagnosis height.
 * Grid stays top-aligned; columns stretch via align-self. Media frame is fixed; spacer grows below.
 */
const RESULTS_GRID_STRETCH_SX = {
  '& .results-image-column': {
    alignSelf: 'stretch',
    boxSizing: 'border-box',
    display: 'flex',
    flexDirection: 'column',
    pb: 2.5,
  },
  '& .results-image-column .results-image-card': {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    minHeight: 0,
  },
  '& .results-image-column .results-image-card-content': {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
    minHeight: 0,
  },
  '& .results-image-column .results-image-media-slot': {
    flex: '0 0 auto',
    width: '100%',
  },
  '& .results-image-column .results-image-card-stretch-spacer': {
    flex: '1 1 auto',
    minHeight: 0,
  },
  '& .results-diagnosis-column': {
    alignSelf: 'stretch',
    alignItems: 'stretch',
    display: 'flex',
    flexDirection: 'column',
    pb: 2.5,
  },
  '& .results-diagnosis-shell': {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
    mb: 0,
    minHeight: 0,
    position: 'relative',
    width: '100%',
  },
  '& .results-diagnosis-card': {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
    minHeight: 0,
  },
} as const;

/** Empty image slot: glass-style panel for light + dark (replaces solid black placeholders). */
type ResultsImageColumnProps = {
  gridArea: 'input' | 'heatmap';
  title: string;
  subheader: React.ReactNode;
  hasMedia: boolean;
  hasBatch: boolean;
  canGoPrev: boolean;
  canGoNext: boolean;
  onGoPrev: () => void;
  onGoNext: () => void;
  mediaSrc?: string;
  mediaAlt: string;
  onMediaError?: () => void;
  onOpenZoom: () => void;
  cardBg: string;
  cardBorder: string;
  mutedText: string;
};

/** Isolated from contributions toggle so expand/collapse does not re-render images. */
const ResultsImageColumn = React.memo(function ResultsImageColumn({
  gridArea,
  title,
  subheader,
  hasMedia,
  hasBatch,
  canGoPrev,
  canGoNext,
  onGoPrev,
  onGoNext,
  mediaSrc,
  mediaAlt,
  onMediaError,
  onOpenZoom,
  cardBg,
  cardBorder,
  mutedText,
}: ResultsImageColumnProps) {
  return (
    <Box className="results-image-column" sx={{ gridArea, minWidth: 0, pb: 2.5 }}>
      <Card
        className="results-image-card"
        variant="outlined"
        sx={{
          bgcolor: cardBg,
          borderColor: cardBorder,
          color: 'text.primary',
          borderRadius: 2,
        }}
      >
        <CardHeader
          title={
            <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
              <Typography component="span" sx={{ fontSize: 14, fontWeight: 600, flexShrink: 0 }}>
                {title}
              </Typography>
              {hasBatch ? (
                <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                  <IconButton size="small" disabled={!canGoPrev} onClick={onGoPrev}>
                    <ChevronLeft fontSize="small" />
                  </IconButton>
                  <IconButton size="small" disabled={!canGoNext} onClick={onGoNext}>
                    <ChevronRight fontSize="small" />
                  </IconButton>
                </Stack>
              ) : null}
            </Stack>
          }
          subheader={subheader}
          sx={{
            pb: 1,
            '& .MuiCardHeader-content': { minWidth: 0 },
            '& .MuiCardHeader-title': { width: '100%' },
            '& .MuiCardHeader-subheader': {
              color: mutedText,
              fontSize: 12,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            },
          }}
        />
        <CardContent className="results-image-card-content" sx={{ pt: 1 }}>
          <Box
            className="results-image-media-slot"
            sx={{
              ...RESULTS_IMAGE_MEDIA_FRAME_SX,
              position: 'relative',
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {hasMedia && mediaSrc ? (
                <ImageZoomTrigger
                  src={mediaSrc}
                  alt={mediaAlt}
                  onError={onMediaError}
                  onOpen={onOpenZoom}
                />
              ) : (
                <Box
                  aria-hidden
                  sx={(theme) => ({
                    ...liquidGlassImagePlaceholderSx(theme),
                    flex: 1,
                    minHeight: 0,
                    height: '100%',
                    aspectRatio: 'unset',
                  })}
                />
              )}
            </Box>
          </Box>
          <Box className="results-image-card-stretch-spacer" aria-hidden />
        </CardContent>
      </Card>
    </Box>
  );
});

function liquidGlassImagePlaceholderSx(theme: Theme) {
  return {
    borderRadius: 2,
    width: '100%',
    position: 'relative' as const,
    overflow: 'hidden',
    border: '1px solid',
    borderColor: 'rgba(148, 163, 184, 0.38)',
    backgroundColor: 'rgba(255, 255, 255, 0.38)',
    backgroundImage:
      'linear-gradient(135deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.12) 42%, rgba(186, 230, 253, 0.18) 100%)',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    boxShadow:
      '0 10px 40px rgba(15, 23, 42, 0.07), inset 0 1px 0 rgba(255,255,255,0.75), inset 0 -1px 0 rgba(148,163,184,0.18)',
    ...theme.applyStyles('dark', {
      borderColor: 'rgba(148, 163, 184, 0.22)',
      backgroundColor: 'rgba(15, 23, 42, 0.42)',
      backgroundImage:
        'linear-gradient(145deg, rgba(255,255,255,0.14) 0%, rgba(30,41,59,0.55) 45%, rgba(59,130,246,0.1) 100%)',
      boxShadow:
        '0 12px 48px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.1), inset 0 -1px 0 rgba(0,0,0,0.35)',
    }),
  };
}

/**
 * Truncate long strings in the MIDDLE — useful for DICOM UIDs where the
 * leading scheme + trailing identifier are both informative.
 *   "raw_1.2.840.114062.2.192.168.196.13.2013.9.27.9.58.39.43467890"
 *   → "raw_1.2.840.114…39.43467890"  (max=28)
 */
function truncateMiddle(value: string, max: number): string {
  if (value.length <= max) return value;
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function formatUploadedAt(isoText: string | null): string {
  if (!isoText) return 'Uploaded • recently';
  const d = new Date(isoText);
  const now = Date.now();
  const diffMs = now - d.getTime();
  if (!Number.isFinite(diffMs)) return 'Uploaded • recently';
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'Uploaded • just now';
  if (diffMins === 1) return 'Uploaded • 1 min ago';
  return `Uploaded • ${diffMins} min ago`;
}

type FeaturesProps = {
  /** When set, this URL is shown in the Input X-Ray preview (e.g. right after upload). */
  previewImageUrl?: string | null;
  /** Local `blob:` URL from file picker — shown immediately before any Firebase upload. */
  localPreviewUrl?: string | null;
  predictUi: PredictUiState;
  onNavigateIndex?: (nextIndex: number) => void;
};

// function contributionBars(riskScore: number) {
//   const r = Math.min(100, Math.max(0, Math.round(riskScore)));
//   return [
//     { label: 'MobileNet-V2', value: Math.min(100, Math.round(r * 0.88)), barColor: '#6366f1' },
//     { label: 'ResNet-50', value: Math.min(100, Math.round(r * 1.02)), barColor: '#3b82f6' },
//     { label: 'DenseNet-121', value: r, barColor: '#22c55e' },
//   ];
// }

// function contributionBars(_riskScore: number) {
//   // Backend currently runs only MobileNetV2.
//   // Keep other planned models visible but zeroed until backend supports them.
//   return [
//     { label: 'MobileNetV2', value: 100, barColor: '#6366f1' },
//     { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
//     { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
//   ];
// }

function contributionBars(contrib?: {
  mobilenetv2: number;
  efficientnetb2: number;
  densenet121: number;
}) {
  if (!contrib) {
    return [
      { label: 'MobileNetV2', value: 0, barColor: '#6366f1' },
      { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
      { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
    ];
  }

  return [
    { label: 'MobileNetV2', value: Math.round(contrib.mobilenetv2), barColor: '#6366f1' },
    { label: 'EfficientNetB2', value: Math.round(contrib.efficientnetb2), barColor: '#f59e0b' },
    { label: 'DenseNet121', value: Math.round(contrib.densenet121), barColor: '#22c55e' },
  ];
}

export default function Features({
  previewImageUrl,
  localPreviewUrl,
  predictUi,
  onNavigateIndex,
}: FeaturesProps) {
  const pageBg = 'background.default';
  const cardBg = 'background.default';
  const cardBorder = 'divider';
  const mutedText = '#9ca3af';

  const [latestUpload, setLatestUpload] = React.useState<UploadRecord | null>(null);
  const [previewLoadFailed, setPreviewLoadFailed] = React.useState(false);

  React.useEffect(() => {
    let active = true;
    fetchLatestUpload()
      .then((record) => {
        if (active) {
          setLatestUpload(record);
        }
      })
      .catch(() => {
        if (active) {
          setLatestUpload(null);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const remotePreviewSrc = previewImageUrl ?? latestUpload?.downloadURL ?? null;
  const hasBatch = predictUi.items.length > 0;
  const safeIndex = Math.max(0, Math.min(predictUi.currentIndex, Math.max(0, predictUi.items.length - 1)));
  const selectedItem = hasBatch ? predictUi.items[safeIndex] : null;
  const previewSrc = selectedItem?.localPreviewUrl ?? localPreviewUrl ?? remotePreviewSrc;
  React.useEffect(() => {
    setPreviewLoadFailed(false);
  }, [previewSrc]);
  const showPreview = Boolean(previewSrc) && !previewLoadFailed;
  const hasRemoteImage = Boolean(remotePreviewSrc);
  const previewSubheader = hasBatch
    ? `Image ${safeIndex + 1} of ${predictUi.items.length} • ${truncateMiddle(
        selectedItem?.fileName ?? 'selected',
        30,
      )}`
    : previewLoadFailed
      ? 'No image uploaded'
    : localPreviewUrl && !previewImageUrl
      ? 'Preview • not uploaded yet'
      : previewImageUrl
        ? 'Uploaded • just now'
        : hasRemoteImage
          ? formatUploadedAt(latestUpload?.uploadedAt ?? null)
          : 'No image uploaded';

  const pred = selectedItem?.result ?? predictUi.data;
  const heatmapB64 = pred?.heatmap?.trim() ?? '';
  const hasHeatmap = heatmapB64 !== '';
  const heatmapSrc = hasHeatmap ? `data:image/png;base64,${heatmapB64}` : '';
  const riskPct =
    pred != null && Number.isFinite(Number(pred.risk_score))
      ? Math.round(Number(pred.risk_score))
      : null;
  const diagnosisLine = pred?.diagnosis?.trim() ?? '';
  const confidenceLine = pred?.confidence_label?.trim() ?? '';
  const diagnosisDisplay = formatDiagnosisLabel(diagnosisLine);
  const isHighRisk = isHighRiskFromResult(diagnosisLine, riskPct, confidenceLine);
  /** Remount diagnosis UI when a new API payload arrives so labels/scores never appear stale. */
  const analysisVersionKey = pred
    ? `${diagnosisLine}|${riskPct ?? ''}|${confidenceLine}`
    : 'no-analysis';
  const canGoPrev = hasBatch && safeIndex > 0;
  const canGoNext = hasBatch && safeIndex < predictUi.items.length - 1;
  // const modelRows = pred && riskPct != null ? contributionBars(Number(pred.risk_score)) : [
  //   { label: 'MobileNet-V2', value: 60, barColor: '#6366f1' },
  //   { label: 'ResNet-50', value: 80, barColor: '#3b82f6' },
  //   { label: 'DenseNet-121', value: 75, barColor: '#22c55e' },
  // ];
  // const modelRows = pred && riskPct != null
  const modelRows = contributionBars(pred?.model_contributions);

  const [contributionsOpen, setContributionsOpen] = React.useState(false);
  /** Stretch image card borders while contributions are open or closing (keeps bottoms aligned during animation). */
  const [contributionsStretchLayout, setContributionsStretchLayout] = React.useState(false);
  const [zoomOpen, setZoomOpen] = React.useState(false);
  const [zoomView, setZoomView] = React.useState<ZoomView>('input');
  const [zoomScale, setZoomScale] = React.useState(1);
  const zoomTouchStartX = React.useRef<number | null>(null);
  const zoomContentRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (contributionsOpen) {
      setContributionsStretchLayout(true);
      return;
    }
    const timer = window.setTimeout(() => setContributionsStretchLayout(false), 280);
    return () => window.clearTimeout(timer);
  }, [contributionsOpen]);

  const previewSrcForIndex = React.useCallback(
    (idx: number) => {
      const item = predictUi.items[idx];
      if (item?.localPreviewUrl) return item.localPreviewUrl;
      if (idx === safeIndex && previewSrc) return previewSrc;
      return null;
    },
    [predictUi.items, safeIndex, previewSrc],
  );

  const heatmapSrcForIndex = React.useCallback(
    (idx: number) => {
      const item = predictUi.items[idx];
      const b64 = item?.result?.heatmap?.trim() ?? '';
      if (b64) return `data:image/png;base64,${b64}`;
      if (idx === safeIndex && heatmapB64) return `data:image/png;base64,${heatmapB64}`;
      return '';
    },
    [predictUi.items, safeIndex, heatmapB64],
  );

  const zoomInputSrc = previewSrcForIndex(safeIndex);
  const zoomHeatmapSrc = heatmapSrcForIndex(safeIndex);
  const zoomDisplaySrc = zoomView === 'input' ? zoomInputSrc : zoomHeatmapSrc;
  const zoomCanShowHeatmap = Boolean(zoomHeatmapSrc);
  const zoomCanShowInput = Boolean(zoomInputSrc);
  const clampZoomScale = React.useCallback(
    (value: number) => Math.min(3, Math.max(0.5, value)),
    [],
  );

  const openZoom = React.useCallback((view: ZoomView) => {
    setZoomView(view);
    setZoomScale(1);
    setZoomOpen(true);
  }, []);

  const handleOpenInputZoom = React.useCallback(() => openZoom('input'), [openZoom]);
  const handleOpenHeatmapZoom = React.useCallback(() => openZoom('heatmap'), [openZoom]);
  const handlePreviewError = React.useCallback(() => setPreviewLoadFailed(true), []);
  const handleGoPrevImage = React.useCallback(() => {
    onNavigateIndex?.(safeIndex - 1);
  }, [onNavigateIndex, safeIndex]);
  const handleGoNextImage = React.useCallback(() => {
    onNavigateIndex?.(safeIndex + 1);
  }, [onNavigateIndex, safeIndex]);

  const closeZoom = React.useCallback(() => {
    setZoomOpen(false);
    setZoomScale(1);
    zoomTouchStartX.current = null;
  }, []);

  const handleZoomTouchStart = (clientX: number) => {
    zoomTouchStartX.current = clientX;
  };

  const handleZoomTouchEnd = (clientX: number) => {
    const start = zoomTouchStartX.current;
    zoomTouchStartX.current = null;
    if (start == null) return;
    const delta = clientX - start;
    if (Math.abs(delta) < 48) return;

    if (delta < 0 && zoomView === 'input' && zoomCanShowHeatmap) {
      setZoomView('heatmap');
      setZoomScale(1);
    } else if (delta > 0 && zoomView === 'heatmap' && zoomCanShowInput) {
      setZoomView('input');
      setZoomScale(1);
    }
  };

  React.useEffect(() => {
    if (!zoomOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeZoom();
        return;
      }
      if (event.key === 'ArrowLeft' && zoomView === 'heatmap' && zoomCanShowInput) {
        setZoomView('input');
        setZoomScale(1);
      } else if (event.key === 'ArrowRight' && zoomView === 'input' && zoomCanShowHeatmap) {
        setZoomView('heatmap');
        setZoomScale(1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [zoomOpen, zoomView, zoomCanShowInput, zoomCanShowHeatmap, closeZoom]);

  /** Trackpad pinch (ctrl+wheel) and scroll zoom on the dialog — block browser page zoom. */
  React.useEffect(() => {
    if (!zoomOpen) return;

    const onWheel = (event: WheelEvent) => {
      const inDialog = Boolean(
        (event.target as Element | null)?.closest?.('.MuiDialog-container'),
      );
      if (!inDialog) return;

      if (event.ctrlKey) {
        event.preventDefault();
        const intensity = 0.012;
        setZoomScale((prev) => clampZoomScale(prev - event.deltaY * intensity));
        return;
      }

      const inImageArea = zoomContentRef.current?.contains(event.target as Node);
      if (!inImageArea) return;

      event.preventDefault();
      setZoomScale((prev) => clampZoomScale(prev - event.deltaY * 0.0025));
    };

    document.addEventListener('wheel', onWheel, { passive: false, capture: true });
    return () => document.removeEventListener('wheel', onWheel, { capture: true });
  }, [zoomOpen, clampZoomScale]);

  React.useEffect(() => {
    if (!zoomOpen) return;
    if (zoomView === 'heatmap' && !zoomCanShowHeatmap && zoomCanShowInput) {
      setZoomView('input');
    }
    if (zoomView === 'input' && !zoomCanShowInput && zoomCanShowHeatmap) {
      setZoomView('heatmap');
    }
  }, [zoomOpen, zoomView, zoomCanShowHeatmap, zoomCanShowInput, safeIndex]);

  React.useEffect(() => {
    if (!hasBatch || !onNavigateIndex) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft' && canGoPrev) {
        event.preventDefault();
        onNavigateIndex(safeIndex - 1);
      } else if (event.key === 'ArrowRight' && canGoNext) {
        event.preventDefault();
        onNavigateIndex(safeIndex + 1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [hasBatch, onNavigateIndex, canGoPrev, canGoNext, safeIndex]);

  // ---- Anomaly flags (simplest opt-in, persisted to localStorage) ----
  /** Keyed by filename: stable enough for a prototype, lets re-running the same image keep its note. */
  const currentFlagKey: string | null = pred
    ? (selectedItem?.fileName ?? latestUpload?.fileName ?? null)
    : null;
  /** In-memory only — cleared on page refresh (no localStorage persistence). */
  const [flags, setFlags] = React.useState<FlagMap>({});
  const [flagFormOpen, setFlagFormOpen] = React.useState(false);
  const [flagFormCategories, setFlagFormCategories] = React.useState<FlagCategory[]>([]);
  const [flagFormNote, setFlagFormNote] = React.useState('');
  const [logOpen, setLogOpen] = React.useState(false);
  const currentFlag: AnomalyFlag | null = currentFlagKey ? (flags[currentFlagKey] ?? null) : null;
  const flagCount = Object.keys(flags).length;

  /** Close the inline form whenever the user navigates to a different image. */
  React.useEffect(() => {
    setFlagFormOpen(false);
  }, [currentFlagKey]);

  const openFlagForm = React.useCallback(() => {
    if (currentFlag) {
      setFlagFormCategories(currentFlag.categories);
      setFlagFormNote(currentFlag.note);
    } else {
      setFlagFormCategories([]);
      setFlagFormNote('');
    }
    setFlagFormOpen(true);
  }, [currentFlag]);

  const toggleFormCategory = React.useCallback((id: FlagCategory) => {
    setFlagFormCategories((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id],
    );
  }, []);

  const saveCurrentFlag = React.useCallback(() => {
    if (!currentFlagKey || !pred || flagFormCategories.length === 0) return;
    /** Preserve the user-facing order of FLAG_CATEGORIES regardless of click order. */
    const orderedCategories = FLAG_CATEGORIES.map((c) => c.id).filter((id) =>
      flagFormCategories.includes(id),
    );
    const next: FlagMap = {
      ...flags,
      [currentFlagKey]: {
        categories: orderedCategories,
        note: flagFormNote.trim(),
        savedAt: new Date().toISOString(),
        snapshot: {
          fileName: selectedItem?.fileName ?? latestUpload?.fileName ?? 'image',
          diagnosis: diagnosisDisplay,
          riskScore: riskPct,
          confidence: confidenceLine,
        },
      },
    };
    setFlags(next);
    setFlagFormOpen(false);
  }, [
    currentFlagKey,
    pred,
    flags,
    flagFormCategories,
    flagFormNote,
    selectedItem,
    latestUpload,
    diagnosisDisplay,
    riskPct,
    confidenceLine,
  ]);

  const removeFlag = React.useCallback(
    (key: string) => {
      const next = { ...flags };
      delete next[key];
      setFlags(next);
      if (key === currentFlagKey) {
        setFlagFormOpen(false);
      }
    },
    [flags, currentFlagKey],
  );

  return (
    <Box sx={{ bgcolor: pageBg }}>
      <Container id="features" maxWidth="lg" sx={{ pt: { xs: 4, md: 5 }, pb: { xs: 6, md: 7 } }}>
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          spacing={1}
          sx={{ mb: 3, flexWrap: 'wrap', rowGap: 1 }}
        >
          <Typography
            variant="h6"
            sx={{
              color: 'text.primary',
              fontWeight: 600,
              letterSpacing: 0.5,
            }}
          >
            AI-assisted TB overview
          </Typography>
          {flagCount > 0 ? (
            <Button
              size="small"
              variant="outlined"
              startIcon={<FlagOutlinedIcon fontSize="small" />}
              onClick={() => setLogOpen(true)}
            >
              Anomaly log ({flagCount})
            </Button>
          ) : null}
        </Stack>

        <Box
          data-stretch={contributionsStretchLayout ? '' : undefined}
          sx={{
            display: 'grid',
            gap: { xs: 2.5, md: 3 },
            alignItems: 'start',
            gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr minmax(300px, 380px)' },
            gridTemplateAreas: {
              xs: `
                "input"
                "heatmap"
                "diagnosis"
              `,
              lg: `"input heatmap diagnosis"`,
            },
            '& .results-image-card-stretch-spacer': {
              flex: '0 0 0',
              height: 0,
              minHeight: 0,
              overflow: 'hidden',
            },
            '&[data-stretch]': RESULTS_GRID_STRETCH_SX,
            '&:not([data-stretch]) .results-diagnosis-shell': {
              mb: 2.5,
            },
            '&:not([data-stretch]) .results-diagnosis-column': {
              alignItems: 'center',
            },
          }}
        >
          <ResultsImageColumn
            gridArea="input"
            title="Input X-Ray"
            subheader={previewSubheader}
            hasMedia={Boolean(showPreview && previewSrc)}
            hasBatch={hasBatch}
            canGoPrev={canGoPrev}
            canGoNext={canGoNext}
            onGoPrev={handleGoPrevImage}
            onGoNext={handleGoNextImage}
            mediaSrc={previewSrc ?? undefined}
            mediaAlt="Input chest X-ray"
            onMediaError={handlePreviewError}
            onOpenZoom={handleOpenInputZoom}
            cardBg={cardBg}
            cardBorder={cardBorder}
            mutedText={mutedText}
          />
          {/* Diagnosis — right column on desktop (+ contributions extension below) */}
          <Box className="results-diagnosis-column" sx={{ gridArea: 'diagnosis', minWidth: 0 }}>
          <Box className="results-diagnosis-shell">
          <Box
            className="results-diagnosis-card"
            sx={{
              width: '100%',
              border: '1px solid',
              borderColor: cardBorder,
              borderRadius: 2,
              bgcolor: cardBg,
              color: 'text.primary',
              overflow: 'visible',
            }}
          >
  <Box
    sx={{
      px: 2,
      pt: 2,
      pb: 3,
      display: 'flex',
      flexDirection: 'column',
      gap: 3, // uniform vertical spacing between blocks
    }}
  >
    {predictUi.loading ? (
      <LinearProgress sx={{ borderRadius: 1 }} />
    ) : null}
    {/* Top: diagnosis + score — key forces fresh DOM when a new /predict result lands */}
    <Box key={analysisVersionKey}>
      <Typography
        variant="overline"
        sx={{ color: mutedText, letterSpacing: 1.5 }}
      >
        ASSESSMENT
      </Typography>

      <Box sx={{ mt: 1 }}>
        <Typography
          component="h2"
          sx={{
            fontWeight: 900,
            lineHeight: 1.05,
            fontSize: { xs: '2.8rem', md: '3.4rem' },
            minHeight: { xs: 48, md: 58 },
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {diagnosisLine ? (
            diagnosisDisplay
          ) : predictUi.loading ? (
            <CircularProgress size={30} thickness={5} />
          ) : (
            '—'
          )}
        </Typography>
        {pred && !predictUi.loading ? (
          <Chip
            label=""
            size="small"
            aria-label={isHighRisk ? 'High risk' : 'Low risk'}
            sx={{
              mt: 1,
              alignSelf: 'flex-start',
              fontSize: 11,
              height: 24,
              minWidth: 48,
              borderRadius: 999,
              px: 1.5,
              '&.MuiChip-root': {
                backgroundColor: isHighRisk ? 'rgba(239,68,68,0.85)' : 'rgba(34,197,94,0.85)',
                border: isHighRisk ? '2px solid #ef4444' : '2px solid #22c55e',
                color: isHighRisk ? '#ef4444' : '#22c55e',
              },
              '& .MuiChip-label': {
                p: 0,
                width: 0,
                overflow: 'hidden',
                opacity: 0,
              },
            }}
          />
        ) : null}
      </Box>

      <Box sx={{ mt: 3 }}>
        <Typography
          variant="subtitle2"
          sx={{ color: mutedText, mb: 0.75 }}
        >
          TB risk score
        </Typography>
        <Typography
          component="p"
          sx={{
            fontWeight: 900,
            lineHeight: 1,
            fontSize: { xs: '3.8rem', md: '4.4rem' }, // big 70%
            minHeight: { xs: 64, md: 72 },
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {riskPct != null ? (
            `${riskPct}%`
          ) : predictUi.loading ? (
            <CircularProgress size={34} thickness={5} />
          ) : (
            '—'
          )}
        </Typography>
        {pred && !predictUi.loading ? (
          <Typography variant="caption" sx={{ color: mutedText, display: 'block', mt: 1 }}>
            Estimated TB probability from the screening model.
          </Typography>
        ) : null}
      </Box>

      <Box sx={{ mt: 2.5 }}>
        <HeatmapLegendBar />
        <Typography variant="caption" sx={{ color: mutedText, display: 'block', mt: 1, lineHeight: 1.45 }}>
          {HEATMAP_GUIDE_TEXT}
        </Typography>
      </Box>
    </Box>

    {/* Anomaly flag (opt-in note from the clinician for the image on screen) */}
    {pred && currentFlagKey ? (
      <Box>
        {currentFlag ? (
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap', rowGap: 1 }}>
            <FlagOutlinedIcon fontSize="small" sx={{ color: 'warning.main' }} />
            {currentFlag.categories.map((c) => (
              <Chip
                key={c}
                label={categoryLabel(c)}
                size="small"
                sx={{
                  bgcolor: 'rgba(245, 158, 11, 0.14)',
                  border: '1px solid rgba(245, 158, 11, 0.55)',
                  color: 'warning.main',
                  borderRadius: 999,
                  fontSize: 11,
                  height: 24,
                }}
              />
            ))}
            <Button size="small" variant="text" onClick={openFlagForm}>
              {flagFormOpen ? 'Close' : 'Edit note'}
            </Button>
            <Button
              size="small"
              variant="text"
              color="error"
              onClick={() => removeFlag(currentFlagKey)}
            >
              Remove
            </Button>
          </Stack>
        ) : (
          <Button
            size="small"
            variant="outlined"
            startIcon={<FlagOutlinedIcon fontSize="small" />}
            onClick={openFlagForm}
          >
            {flagFormOpen ? 'Cancel' : 'Flag anomaly'}
          </Button>
        )}

        <Collapse in={flagFormOpen} unmountOnExit>
          <Box
            sx={(theme) => ({
              mt: 1.5,
              p: 2,
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'rgba(148, 163, 184, 0.35)',
              bgcolor: 'rgba(255, 255, 255, 0.55)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              ...theme.applyStyles('dark', {
                bgcolor: 'rgba(30, 41, 59, 0.55)',
                borderColor: 'rgba(148, 163, 184, 0.22)',
              }),
            })}
          >
            <Typography
              variant="caption"
              sx={{ color: mutedText, display: 'block', mb: 1 }}
            >
              Note an anomaly
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1.5 }}>
              {FLAG_CATEGORIES.map((c) => {
                const active = flagFormCategories.includes(c.id);
                return (
                  <Chip
                    key={c.id}
                    label={c.label}
                    size="small"
                    clickable
                    /** Use outlined for unselected so MUI's filled-default bg
                     *  doesn't override our transparent background. */
                    variant={active ? 'filled' : 'outlined'}
                    onClick={() => toggleFormCategory(c.id)}
                    aria-pressed={active}
                    /**
                     * The shared theme's `MuiChip` variant for `color: 'default'`
                     * paints filled chips with `gray[800]` bg + `gray[300]` label
                     * color in dark mode (and a light gray version in light mode).
                     * That variant generates `.MuiChip-root.MuiChip-colorDefault`
                     * selectors which match this Chip and tie the specificity of
                     * a plain `sx` override — meaning the theme can win on
                     * source order, especially once focus shifts away from the
                     * chip and re-triggers its resting state.
                     *
                     * We bump specificity by anchoring rules to `&.MuiChip-root`
                     * (and `&.MuiChip-root .MuiChip-label`) so the selected
                     * white-on-dark / dark-on-white look HOLDS regardless of
                     * focus, hover, or where the user clicks next.
                     */
                    sx={{
                      borderRadius: 999,
                      fontSize: 11,
                      height: 24,
                      transition:
                        'background-color 120ms ease, color 120ms ease, border-color 120ms ease',
                      ...(active
                        ? {
                            '&.MuiChip-root': {
                              backgroundColor: 'text.primary',
                              borderColor: 'text.primary',
                              border: '1px solid',
                            },
                            '&.MuiChip-root .MuiChip-label': {
                              color: 'background.default',
                            },
                            '&.MuiChip-root:hover, &.MuiChip-root:focus, &.MuiChip-root:focus-visible, &.MuiChip-root:active':
                              {
                                backgroundColor: 'text.primary',
                                borderColor: 'text.primary',
                                opacity: 1,
                              },
                          }
                        : {
                            '&.MuiChip-root': {
                              backgroundColor: 'transparent',
                              borderColor: 'divider',
                              border: '1px solid',
                            },
                            '&.MuiChip-root .MuiChip-label': {
                              color: 'text.primary',
                            },
                            '&.MuiChip-root:hover': {
                              backgroundColor: 'action.hover',
                              borderColor: 'divider',
                            },
                          }),
                    }}
                  />
                );
              })}
            </Box>
            <TextField
              fullWidth
              multiline
              minRows={2}
              maxRows={6}
              placeholder="Note (e.g., missed nodule on right apex)"
              value={flagFormNote}
              onChange={(e) => setFlagFormNote(e.target.value)}
              sx={{
                // The shared theme pins MuiOutlinedInput's root to a fixed height
                // for `size: small`, which squeezes the multiline textarea against
                // the top edge. Override height + padding for breathing room.
                '& .MuiOutlinedInput-root': {
                  height: 'auto',
                  py: 1.25,
                  px: 1.5,
                  alignItems: 'flex-start',
                },
                '& .MuiOutlinedInput-input': {
                  padding: 0,
                },
              }}
            />
            <Stack direction="row" spacing={1} sx={{ mt: 1.5, justifyContent: 'flex-end' }}>
              <Button size="small" onClick={() => setFlagFormOpen(false)}>
                Cancel
              </Button>
              <Button
                size="small"
                variant="contained"
                onClick={saveCurrentFlag}
                disabled={flagFormCategories.length === 0}
                sx={(theme) => ({
                  // Keep the label visible when disabled — MUI's default disabled
                  // styling fades both bg and text, which on dark mode reads as a
                  // near-blank white tile. Use a clearly-disabled-but-readable look.
                  '&.Mui-disabled': {
                    opacity: 1,
                    color: 'rgba(15, 23, 42, 0.55)',
                    backgroundColor: 'rgba(148, 163, 184, 0.28)',
                    backgroundImage: 'none',
                    boxShadow: 'none',
                    borderColor: 'rgba(148, 163, 184, 0.35)',
                    ...theme.applyStyles('dark', {
                      color: 'rgba(241, 245, 249, 0.7)',
                      backgroundColor: 'rgba(148, 163, 184, 0.18)',
                      borderColor: 'rgba(148, 163, 184, 0.28)',
                    }),
                  },
                })}
              >
                {currentFlag ? 'Update note' : 'Save note'}
              </Button>
            </Stack>
          </Box>
        </Collapse>
      </Box>
    ) : null}

  </Box>

            <Box sx={{ position: 'relative', height: 0, zIndex: 2 }}>
              <Box
                aria-hidden
                sx={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  top: 0,
                  height: '1px',
                  pointerEvents: 'none',
                  transform: 'translateY(-50%)',
                  opacity: contributionsOpen ? 1 : 0,
                  transition: 'opacity 280ms cubic-bezier(0.4, 0, 0.2, 1)',
                  background: (theme) =>
                    `linear-gradient(90deg, transparent 0%, ${theme.palette.divider} 16%, ${theme.palette.divider} 50%, ${theme.palette.divider} 84%, transparent 100%)`,
                }}
              />
              <IconButton
                size="small"
                aria-label={contributionsOpen ? 'Hide model contributions' : 'Show model contributions'}
                aria-expanded={contributionsOpen}
                onClick={() => setContributionsOpen((open) => !open)}
                sx={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  width: 32,
                  height: 32,
                  p: 0,
                  borderRadius: '50%',
                  border: '1px solid',
                  borderColor: cardBorder,
                  bgcolor: cardBg,
                  boxShadow: '0 2px 8px rgba(15, 23, 42, 0.12)',
                  transition: 'transform 180ms ease',
                  transform: contributionsOpen
                    ? 'translate(-50%, -50%) rotate(180deg)'
                    : 'translate(-50%, -50%)',
                  '&:hover': { bgcolor: 'action.hover' },
                }}
              >
                <ExpandMoreIcon fontSize="small" />
              </IconButton>
            </Box>

            <Box
              sx={{
                display: 'grid',
                gridTemplateRows: contributionsOpen ? '1fr' : '0fr',
                transition: 'grid-template-rows 280ms cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              <Box sx={{ overflow: 'hidden', minHeight: 0 }}>
                <Box sx={{ px: 2, pb: 2, pt: 3.5 }}>
                  <Typography
                    variant="subtitle1"
                    sx={{ color: mutedText, mb: 1.5, fontWeight: 600, fontSize: '1rem' }}
                  >
                    Model contributions
                  </Typography>
                  {modelRows.map((item) => (
                    <Box key={item.label} sx={{ mb: 1.5 }}>
                      <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
                        <Typography variant="body2">{item.label}</Typography>
                        <Typography variant="body2" sx={{ color: mutedText }}>
                          {item.value}%
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={item.value}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          bgcolor: 'action.hover',
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 3,
                            bgcolor: item.barColor,
                          },
                        }}
                      />
                    </Box>
                  ))}
                </Box>
              </Box>
            </Box>
          </Box>
          </Box>
          </Box>

          <ResultsImageColumn
            gridArea="heatmap"
            title="Prediction Heat Map"
            subheader="Highlighted TB-suspect regions"
            hasMedia={hasHeatmap}
            hasBatch={hasBatch}
            canGoPrev={canGoPrev}
            canGoNext={canGoNext}
            onGoPrev={handleGoPrevImage}
            onGoNext={handleGoNextImage}
            mediaSrc={hasHeatmap ? heatmapSrc : undefined}
            mediaAlt="Saliency overlay on input study"
            onOpenZoom={handleOpenHeatmapZoom}
            cardBg={cardBg}
            cardBorder={cardBorder}
            mutedText={mutedText}
          />
        </Box>
        <Dialog
          open={zoomOpen}
          onClose={closeZoom}
          maxWidth={false}
          PaperProps={{
            sx: {
              bgcolor: 'background.default',
              maxWidth: 'min(96vw, 1200px)',
              width: '96vw',
              m: 1,
            },
          }}
        >
          <DialogTitle sx={{ py: 1, pr: 1 }}>
            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              alignItems={{ xs: 'stretch', sm: 'center' }}
              justifyContent="space-between"
              spacing={1}
            >
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="subtitle2">
                  {zoomView === 'input' ? 'Input X-Ray' : 'Prediction heat map'}
                </Typography>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  {zoomCanShowInput && zoomCanShowHeatmap
                    ? 'Swipe or arrow keys: X-ray ↔ heat map. Pinch or scroll on the image to zoom.'
                    : 'Pinch or scroll on the image to zoom.'}
                </Typography>
              </Box>
              <Stack direction="row" spacing={0.5} alignItems="center" justifyContent="flex-end">
                {zoomCanShowInput && zoomCanShowHeatmap ? (
                  <Stack direction="row" spacing={0.5}>
                    <Button
                      size="small"
                      variant={zoomView === 'input' ? 'contained' : 'outlined'}
                      onClick={() => {
                        setZoomView('input');
                        setZoomScale(1);
                      }}
                    >
                      X-ray
                    </Button>
                    <Button
                      size="small"
                      variant={zoomView === 'heatmap' ? 'contained' : 'outlined'}
                      onClick={() => {
                        setZoomView('heatmap');
                        setZoomScale(1);
                      }}
                    >
                      Heat map
                    </Button>
                  </Stack>
                ) : null}
                <IconButton
                  size="small"
                  onClick={() => setZoomScale((s) => clampZoomScale(s - 0.25))}
                  aria-label="Zoom out"
                  disabled={zoomScale <= 0.5}
                >
                  <ZoomOutIcon fontSize="small" />
                </IconButton>
                <Typography variant="caption" sx={{ minWidth: 40, textAlign: 'center' }}>
                  {Math.round(zoomScale * 100)}%
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => setZoomScale((s) => clampZoomScale(s + 0.25))}
                  aria-label="Zoom in"
                  disabled={zoomScale >= 3}
                >
                  <ZoomInIcon fontSize="small" />
                </IconButton>
                <IconButton size="small" onClick={closeZoom} aria-label="Close">
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Stack>
            </Stack>
          </DialogTitle>
          <DialogContent
            ref={zoomContentRef}
            dividers
            onTouchStart={(e) => {
              const touch = e.touches[0];
              if (touch) handleZoomTouchStart(touch.clientX);
            }}
            onTouchEnd={(e) => {
              const touch = e.changedTouches[0];
              if (touch) handleZoomTouchEnd(touch.clientX);
            }}
            sx={{
              overflow: 'auto',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'flex-start',
              minHeight: 280,
              maxHeight: '80vh',
              position: 'relative',
              touchAction: 'none',
              overscrollBehavior: 'contain',
            }}
          >
            {zoomCanShowInput && zoomCanShowHeatmap ? (
              <IconButton
                sx={{
                  position: 'absolute',
                  left: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  bgcolor: 'background.paper',
                  boxShadow: 1,
                  visibility: zoomView === 'heatmap' ? 'visible' : 'hidden',
                  pointerEvents: zoomView === 'heatmap' ? 'auto' : 'none',
                }}
                onClick={() => {
                  setZoomView('input');
                  setZoomScale(1);
                }}
                aria-label="Show X-ray"
              >
                <ChevronLeft />
              </IconButton>
            ) : null}
            {zoomDisplaySrc ? (
              <Box
                component="img"
                src={zoomDisplaySrc}
                alt={zoomView === 'input' ? 'Input chest X-ray' : 'Heat map overlay'}
                draggable={false}
                sx={{
                  transform: `scale(${zoomScale})`,
                  transformOrigin: 'center center',
                  transition: 'transform 80ms ease-out',
                  maxWidth: '100%',
                  maxHeight: '75vh',
                  height: 'auto',
                  objectFit: 'contain',
                }}
              />
            ) : (
              <Typography variant="body2" sx={{ color: mutedText, py: 6 }}>
                No image available for this view.
              </Typography>
            )}
            {zoomCanShowInput && zoomCanShowHeatmap ? (
              <IconButton
                sx={{
                  position: 'absolute',
                  right: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  bgcolor: 'background.paper',
                  boxShadow: 1,
                  visibility: zoomView === 'input' ? 'visible' : 'hidden',
                  pointerEvents: zoomView === 'input' ? 'auto' : 'none',
                }}
                onClick={() => {
                  setZoomView('heatmap');
                  setZoomScale(1);
                }}
                aria-label="Show heat map"
              >
                <ChevronRight />
              </IconButton>
            ) : null}
          </DialogContent>
        </Dialog>

        <Dialog open={logOpen} onClose={() => setLogOpen(false)} fullWidth maxWidth="sm">
          <DialogTitle>Anomaly log ({flagCount})</DialogTitle>
          <DialogContent dividers>
            {flagCount === 0 ? (
              <Typography variant="body2" sx={{ color: mutedText }}>
                No anomalies noted yet.
              </Typography>
            ) : (
              <Stack spacing={1.5}>
                {Object.entries(flags)
                  .sort(([, a], [, b]) => (a.savedAt < b.savedAt ? 1 : -1))
                  .map(([key, flag]) => (
                    <Box
                      key={key}
                      sx={{
                        p: 1.5,
                        borderRadius: 1.5,
                        border: '1px solid',
                        borderColor: 'divider',
                      }}
                    >
                      <Stack
                        direction="row"
                        justifyContent="space-between"
                        alignItems="flex-start"
                        sx={{ mb: 0.75 }}
                      >
                        <Box sx={{ minWidth: 0, pr: 1 }}>
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 600, wordBreak: 'break-word' }}
                          >
                            {flag.snapshot.fileName}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{ color: mutedText, display: 'block' }}
                          >
                            Model: {formatDiagnosisLabel(flag.snapshot.diagnosis || '') || '—'}
                            {flag.snapshot.riskScore != null ? ` • ${flag.snapshot.riskScore}%` : ''}
                            {flag.snapshot.confidence ? ` • ${flag.snapshot.confidence}` : ''}
                          </Typography>
                        </Box>
                        <IconButton
                          size="small"
                          aria-label="Delete note"
                          onClick={() => removeFlag(key)}
                        >
                          <DeleteOutlineIcon fontSize="small" />
                        </IconButton>
                      </Stack>
                      <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', rowGap: 0.5 }}>
                        {flag.categories.map((c) => (
                          <Chip
                            key={c}
                            label={categoryLabel(c)}
                            size="small"
                            sx={{
                              bgcolor: 'rgba(245, 158, 11, 0.14)',
                              border: '1px solid rgba(245, 158, 11, 0.55)',
                              color: 'warning.main',
                              borderRadius: 999,
                              fontSize: 11,
                              height: 22,
                            }}
                          />
                        ))}
                      </Stack>
                      {flag.note ? (
                        <Typography variant="body2" sx={{ mt: 0.75, whiteSpace: 'pre-wrap' }}>
                          {flag.note}
                        </Typography>
                      ) : null}
                      <Typography
                        variant="caption"
                        sx={{ color: mutedText, display: 'block', mt: 0.75 }}
                      >
                        {new Date(flag.savedAt).toLocaleString()}
                      </Typography>
                    </Box>
                  ))}
              </Stack>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setLogOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>
        {hasBatch ? (
          <Stack direction="row" justifyContent="center" spacing={0.75} sx={{ mt: 2 }}>
            {predictUi.items.map((item, idx) => {
              const isActive = idx === safeIndex;
              const color =
                item.status === 'error'
                  ? '#ef4444'
                  : item.status === 'done'
                    ? '#22c55e'
                    : item.status === 'processing'
                      ? '#3b82f6'
                      : '#6b7280';
              return (
                <Box
                  key={item.id}
                  component="button"
                  type="button"
                  aria-label={`Jump to image ${idx + 1}`}
                  onClick={() => onNavigateIndex?.(idx)}
                  sx={{
                    width: isActive ? 12 : 10,
                    height: isActive ? 12 : 10,
                    borderRadius: '50%',
                    border: 'none',
                    p: 0,
                    cursor: 'pointer',
                    bgcolor: color,
                    opacity: isActive ? 1 : 0.45,
                    transform: isActive ? 'scale(1.05)' : 'none',
                    transition: 'all 120ms ease',
                  }}
                />
              );
            })}
          </Stack>
        ) : null}
      </Container>
    </Box>
  );
}
