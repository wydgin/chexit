import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const rootElement = document.getElementById('root');
if (!rootElement) {
  document.body.innerHTML =
    '<div style="padding: 20px;"><h1>Error</h1><p>Missing #root element.</p></div>';
} else {
  try {
    ReactDOM.createRoot(rootElement).render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error('Error rendering React:', error);
    }
    rootElement.innerHTML = `<div style="padding: 20px;">
      <h1>Error loading app</h1>
      <p>${error instanceof Error ? error.message : String(error)}</p>
    </div>`;
  }
}
