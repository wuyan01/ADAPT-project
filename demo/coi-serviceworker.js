// Enables cross-origin isolation (SharedArrayBuffer → multi-threaded ONNX Runtime WASM) on static hosts such as
// GitHub Pages that cannot send COOP/COEP headers. The same file is both the page-side registration script and the
// service worker. Scope is this directory, so the project page one level up is unaffected.
//
// Note: with COEP in place every cross-origin resource on this page needs a CORP/CORS header. The demo only loads
// same-origin files, so nothing changes; keep it that way (no third-party CDN scripts or iframes in demo/).
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', event => {
    const request = event.request;
    if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
    event.respondWith(fetch(request).then(response => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
      return new Response(response.body, {status: response.status, statusText: response.statusText, headers});
    }));
  });
} else if (!window.crossOriginIsolated && 'serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register(document.currentScript.src, {scope: './'}).then(registration => {
    // First visit: the worker is not controlling the page yet, so reload once after it activates.
    if (navigator.serviceWorker.controller || sessionStorage.getItem('coi-reloaded')) return;
    sessionStorage.setItem('coi-reloaded', '1');
    const reload = () => window.location.reload();
    if (registration.active) reload(); else navigator.serviceWorker.addEventListener('controllerchange', reload, {once: true});
  }).catch(e => console.warn('coi-serviceworker registration failed; CPU inference stays single-threaded:', e));
}
