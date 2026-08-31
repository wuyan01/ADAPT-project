/*
 * Lazy video loading.
 *
 * The page carries a lot of video, so <source> tags use `data-src` instead of
 * `src`. A video only fetches and plays once it scrolls into view, and pauses
 * again when it leaves. This keeps the initial page load to the figures only.
 *
 * bulma-carousel deep-clones slides to fill its track when `infinite` is on,
 * so `refresh()` is exposed for the carousel setup to call once it has built
 * those clones. The clones are fresh elements that the observer below has
 * never seen, and they inherit `data-loaded` from the slide they were copied
 * from, which would otherwise leave them permanently blank.
 */
(function () {
  'use strict';

  var observer = null;
  var seen = typeof WeakSet === 'function' ? new WeakSet() : null;

  function load(video) {
    if (video.dataset.loaded === 'true') return;
    var sources = video.querySelectorAll('source[data-src]');
    for (var i = 0; i < sources.length; i++) {
      sources[i].src = sources[i].dataset.src;
    }
    video.dataset.loaded = 'true';
    video.load();
  }

  // True when a clone claims to be loaded but carries no usable src.
  function isStale(video) {
    var sources = video.querySelectorAll('source[data-src]');
    for (var i = 0; i < sources.length; i++) {
      if (!sources[i].getAttribute('src')) return true;
    }
    return false;
  }

  function track(video) {
    if (seen) {
      if (seen.has(video)) return;
      seen.add(video);
    }
    if (isStale(video)) video.dataset.loaded = '';
    if (observer) {
      observer.observe(video);
    } else {
      load(video);
    }
  }

  function refresh() {
    var videos = document.querySelectorAll('video.lazy-video');
    Array.prototype.forEach.call(videos, track);
  }

  function init() {
    // Fall back to loading everything if IntersectionObserver is unavailable.
    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var video = entry.target;
          if (entry.isIntersecting) {
            load(video);
            var p = video.play();
            if (p && p.catch) p.catch(function () { /* autoplay blocked */ });
          } else if (!video.paused) {
            video.pause();
          }
        });
      }, { rootMargin: '200px 0px', threshold: 0.1 });
    }

    refresh();
  }

  window.adaptRefreshLazyVideos = refresh;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
