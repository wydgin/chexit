import * as React from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Container from '@mui/material/Container';
import LinearProgress from '@mui/material/LinearProgress';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { predictImagesSequential, releaseBatchPreviewUrls } from '../../api/chexit';
import type { PredictUiState } from '../../api/chexit';
import { storage, db } from '../../firebase';

const UPLOADS_COLLECTION = 'uploads';
const LATEST_DOC_ID = 'latest';
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_BATCH_IMAGES = 5;

type HeroProps = {
  /** Latest analyze state (shown in alerts + passed to Features via parent). */
  predictUi: PredictUiState;
  onUploadComplete?: (downloadUrl: string) => void;
  /** Temporary `blob:` URL for instant preview in the dashboard (no upload required). */
  onLocalPreviewChange?: (previewUrl: string | null) => void;
  onPredictUiChange?: (state: PredictUiState) => void;
};

export default function Hero({
  predictUi,
  onUploadComplete,
  onLocalPreviewChange,
  onPredictUiChange,
}: HeroProps) {
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [analyzing, setAnalyzing] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState(false);
  const [progressText, setProgressText] = React.useState<string | null>(null);
  const [progressValue, setProgressValue] = React.useState<number>(0);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const previousItemsRef = React.useRef<PredictUiState['items']>([]);

  React.useEffect(
    () => () => {
      releaseBatchPreviewUrls(previousItemsRef.current);
    },
    [],
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    if (picked.length === 0) {
      return;
    }
    const files = picked.slice(0, MAX_BATCH_IMAGES);
    if (picked.length > MAX_BATCH_IMAGES) {
      setUploadError(`Only up to ${MAX_BATCH_IMAGES} images are allowed per batch.`);
      setUploadSuccess(false);
    } else {
      setUploadError(null);
    }
    const badType = files.find((file) => {
      const isImage = file.type.startsWith('image/');
      const isDicom =
        file.type === 'application/dicom' ||
        file.type === 'application/octet-stream' ||
        /\.(dcm|dicom)$/i.test(file.name);
      return !isImage && !isDicom;
    });
    if (badType) {
      setUploadError('Please upload PNG/JPG or DICOM (.dcm).');
      setUploadSuccess(false);
      return;
    }
    const tooLarge = files.find((file) => file.size > MAX_IMAGE_BYTES);
    if (tooLarge) {
      setUploadError('Max file size is 10MB per image.');
      setUploadSuccess(false);
      return;
    }
    setSelectedFiles(files);
    setUploadSuccess(false);
    setProgressText(null);
    setProgressValue(0);
    releaseBatchPreviewUrls(previousItemsRef.current);
    previousItemsRef.current = [];
    onLocalPreviewChange?.(null);
    onPredictUiChange?.({ loading: false, error: null, data: null, items: [], currentIndex: 0 });
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (selectedFiles.length === 0) {
      onPredictUiChange?.({
        loading: false,
        error: 'Choose a chest X-ray with Browse file, then click Analyze.',
        data: null,
        items: [],
        currentIndex: 0,
      });
      return;
    }
    setAnalyzing(true);
    setProgressText(`Preparing ${selectedFiles.length} image(s)…`);
    setProgressValue(0);
    onPredictUiChange?.({ loading: true, error: null, data: null, items: [], currentIndex: 0 });
    try {
      releaseBatchPreviewUrls(previousItemsRef.current);
      const finalItems = await predictImagesSequential(selectedFiles, (nextItems, currentIndex) => {
        previousItemsRef.current = nextItems;
        const doneCount = nextItems.filter((item) => item.status === 'done' || item.status === 'error').length;
        const pct = Math.round((doneCount / Math.max(nextItems.length, 1)) * 100);
        setProgressValue(pct);
        setProgressText(`Processing image ${Math.min(currentIndex + 1, nextItems.length)} of ${nextItems.length}`);
        const selected = nextItems[currentIndex]?.result ?? null;
        onLocalPreviewChange?.(nextItems[currentIndex]?.localPreviewUrl ?? null);
        onPredictUiChange?.({
          loading: true,
          error: null,
          data: selected,
          items: nextItems,
          currentIndex,
        });
      });
      previousItemsRef.current = finalItems;
      const firstSuccessful = finalItems.findIndex((item) => item.status === 'done' && item.result);
      const finalIndex = firstSuccessful >= 0 ? firstSuccessful : 0;
      const selected = finalItems[finalIndex]?.result ?? null;
      const errors = finalItems.filter((item) => item.status === 'error').length;
      onLocalPreviewChange?.(finalItems[finalIndex]?.localPreviewUrl ?? null);
      onPredictUiChange?.({
        loading: false,
        error: errors > 0 ? `${errors} image(s) failed. Use arrows below to inspect each result.` : null,
        data: selected,
        items: finalItems,
        currentIndex: finalIndex,
      });
      setProgressValue(100);
      setProgressText(`Completed ${finalItems.length} image(s).`);
    } catch {
      onPredictUiChange?.({ loading: false, error: 'Analyze failed', data: null, items: [], currentIndex: 0 });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpload = async () => {
    const selectedFile = selectedFiles[0];
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);
    try {
      const storageRef = ref(storage, `uploads/${Date.now()}_${selectedFile.name}`);
      await uploadBytes(storageRef, selectedFile);
      const downloadURL = await getDownloadURL(storageRef);
      const latestRef = doc(db, UPLOADS_COLLECTION, LATEST_DOC_ID);
      await setDoc(latestRef, {
        downloadURL,
        fileName: selectedFile.name,
        uploadedAt: serverTimestamp(),
      });
      onUploadComplete?.(downloadURL);
      setUploadSuccess(true);
      onLocalPreviewChange?.(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(message);
      setUploadSuccess(false);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box
      id="hero"
      sx={(theme) => ({
        width: '100%',
        backgroundRepeat: 'no-repeat',

        backgroundImage:
          'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
        ...theme.applyStyles('dark', {
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
        }),
      })}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          pt: { xs: 12, sm: 16 },
          pb: { xs: 4, sm: 6 },
        }}
      >
        <Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >
          <Typography
            variant="h1"
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              alignItems: 'center',
              fontSize: 'clamp(3rem, 10vw, 3.5rem)',
            }}
          >
            Your&nbsp;latest&nbsp;
            <Typography
              component="span"
              variant="h1"
              sx={(theme) => ({
                fontSize: 'inherit',
                color: 'primary.main',
                ...theme.applyStyles('dark', {
                  color: 'primary.light',
                }),
              })}
            >
              TB assistant
            </Typography>
          </Typography>
          <Typography
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              width: { sm: '100%', md: '80%' },
            }}
          >
            Upload a chest X-ray to get AI-assisted TB risk scores and heatmaps
          </Typography>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            useFlexGap
            flexWrap="wrap"
            justifyContent="center"
            sx={{ pt: 2, width: { xs: '100%', sm: 'auto' }, maxWidth: 520 }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*,.dcm,.dicom,application/dicom"
              multiple
              style={{ display: 'none' }}
            />
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleBrowseClick}
              disabled={uploading || analyzing}
              sx={{ minWidth: 'fit-content' }}
            >
              {selectedFiles.length > 0 ? `${selectedFiles.length} file(s) selected` : 'Browse files (max 5)'}
            </Button>
            <Button
              type="button"
              variant={selectedFiles.length > 0 ? 'contained' : 'outlined'}
              color={selectedFiles.length > 0 ? 'primary' : 'inherit'}
              size="small"
              onClick={handleAnalyze}
              disabled={uploading || analyzing}
              startIcon={analyzing ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={{
                minWidth: 'fit-content',
                ...(!(selectedFiles.length > 0) && {
                  backgroundColor: 'action.disabledBackground',
                  color: 'action.disabled',
                  borderColor: 'action.disabled',
                }),
              }}
            >
              {analyzing ? 'Analyzing…' : 'Analyze'}
            </Button>
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || uploading || analyzing}
              startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={{ minWidth: 'fit-content' }}
            >
              {uploading ? 'Uploading…' : 'Upload to cloud'}
            </Button>
          </Stack>
          {uploadError && (
            <Typography variant="body2" color="error" sx={{ textAlign: 'center' }}>
              {uploadError}
            </Typography>
          )}
          {uploadSuccess && (
            <Typography variant="body2" color="success.main" sx={{ textAlign: 'center' }}>
              File uploaded. Check preview below.
            </Typography>
          )}
          {(predictUi.loading || analyzing) && (
            <Alert severity="info" sx={{ maxWidth: 560, width: '100%', textAlign: 'center' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                {progressText ?? 'Analyzing… This may take a few minutes.'}
              </Typography>
              <LinearProgress variant="determinate" value={progressValue} sx={{ borderRadius: 1 }} />
            </Alert>
          )}
          {predictUi.items.length > 0 ? (
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" justifyContent="center">
              {predictUi.items.map((item, idx) => (
                <Chip
                  key={item.id}
                  size="small"
                  label={`${idx + 1}. ${item.fileName}`}
                  color={item.status === 'error' ? 'error' : item.status === 'done' ? 'success' : 'default'}
                  variant={predictUi.currentIndex === idx ? 'filled' : 'outlined'}
                />
              ))}
            </Stack>
          ) : null}
          {!predictUi.loading && !analyzing && predictUi.error ? (
            <Alert severity="error" sx={{ maxWidth: 560, width: '100%', textAlign: 'left' }}>
              {predictUi.error}
            </Alert>
          ) : null}
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ textAlign: 'center' }}
          >
            By clicking &quot;Upload&quot; you agree to our&nbsp;
            <Link href="#" color="primary">
              Terms & Conditions
            </Link>
            .
          </Typography>
        </Stack>
        {/* removed dashboard image */}
      </Container>
    </Box>
  );
}
