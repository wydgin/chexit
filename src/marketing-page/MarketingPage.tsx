import * as React from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import type { PredictUiState } from '../api/chexit';
import AppTheme from '../../shared-theme/AppTheme';
import AppAppBar from './components/AppAppBar';
import Hero from './components/Hero';
// import LogoCollection from './components/LogoCollection';
// import Highlights from './components/Highlights';
// import Pricing from './components/Pricing';
import Features from './components/Features';
// import Testimonials from './components/Testimonials';
// import FAQ from './components/FAQ';
import Footer from './components/Footer';

const initialPredict: PredictUiState = {
  loading: false,
  error: null,
  data: null,
  items: [],
  currentIndex: 0,
};

export default function MarketingPage(props: { disableCustomTheme?: boolean }) {
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = React.useState<string | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = React.useState<string | null>(null);
  const [predictUi, setPredictUi] = React.useState<PredictUiState>(initialPredict);

  /** Clone API payload so React always sees a new `data` reference after each Analyze. */
  const handlePredictUiChange = React.useCallback((next: PredictUiState) => {
    setPredictUi({
      loading: next.loading,
      error: next.error,
      data: next.data ? { ...next.data } : null,
      items: next.items ? [...next.items] : [],
      currentIndex: next.currentIndex ?? 0,
    });
  }, []);

  const handleNavigateResult = React.useCallback((nextIndex: number) => {
    setPredictUi((prev) => {
      if (prev.items.length === 0) return prev;
      const clamped = Math.max(0, Math.min(nextIndex, prev.items.length - 1));
      const item = prev.items[clamped];
      return {
        ...prev,
        currentIndex: clamped,
        data: item?.result ? { ...item.result } : null,
        error: item?.status === 'error' ? item.error : prev.error,
      };
    });
  }, []);

  React.useEffect(() => {
    if (!predictUi.data || predictUi.loading || predictUi.error) {
      return;
    }
    document.getElementById('features')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, [predictUi.data, predictUi.loading, predictUi.error]);

  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />

      <AppAppBar />
      <Hero
        predictUi={predictUi}
        onUploadComplete={setUploadedPreviewUrl}
        onLocalPreviewChange={setLocalPreviewUrl}
        onPredictUiChange={handlePredictUiChange}
      />
      <div>
        {/* <LogoCollection /> */}
        <Features
          previewImageUrl={uploadedPreviewUrl}
          localPreviewUrl={localPreviewUrl}
          predictUi={predictUi}
          onNavigateIndex={handleNavigateResult}
        />
        {/* <Divider />
        <Testimonials />
        <Divider />
        <Highlights />
        <Divider />
        <Pricing />
        <Divider />
        <FAQ />
        <Divider /> */}
        <Footer />
      </div>
    </AppTheme>
  );
}
