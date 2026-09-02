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

/*
 * Enhanced page navigation and restrained reveal motion.
 * This is intentionally dependency-free and honors prefers-reduced-motion.
 */
(function () {
  'use strict';

  function initPageChrome() {
    var body = document.body;
    var nav = document.querySelector('.site-nav');
    var menu = document.querySelector('.nav-links');
    var toggle = document.querySelector('.nav-toggle');
    var progress = document.querySelector('.scroll-progress span');
    var navLinks = document.querySelectorAll('.nav-links a');
    var reduceMotion = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function updateScrollChrome() {
      var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      var scrollRange = document.documentElement.scrollHeight - window.innerHeight;
      var ratio = scrollRange > 0 ? Math.min(scrollTop / scrollRange, 1) : 0;

      if (progress) progress.style.transform = 'scaleX(' + ratio + ')';
      if (nav) nav.classList.toggle('is-scrolled', scrollTop > 32);
    }

    function closeMenu() {
      if (!menu || !toggle) return;
      menu.classList.remove('is-open');
      toggle.classList.remove('is-active');
      toggle.setAttribute('aria-expanded', 'false');
      if (nav) nav.classList.remove('menu-open');
    }

    if (toggle && menu) {
      toggle.addEventListener('click', function () {
        var open = !menu.classList.contains('is-open');
        menu.classList.toggle('is-open', open);
        toggle.classList.toggle('is-active', open);
        toggle.setAttribute('aria-expanded', String(open));
        if (nav) nav.classList.toggle('menu-open', open);
      });
    }

    Array.prototype.forEach.call(navLinks, function (link) {
      link.addEventListener('click', closeMenu);
    });

    Array.prototype.forEach.call(document.querySelectorAll('a[aria-disabled="true"]'), function (link) {
      link.addEventListener('click', function (event) {
        event.preventDefault();
      });
    });

    var reveals = document.querySelectorAll('.reveal');
    if (!reduceMotion && 'IntersectionObserver' in window) {
      body.classList.add('motion-ready');
      var revealObserver = new IntersectionObserver(function (entries, observer) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });

      Array.prototype.forEach.call(reveals, function (element) {
        revealObserver.observe(element);
      });
    } else {
      Array.prototype.forEach.call(reveals, function (element) {
        element.classList.add('is-visible');
      });
    }

    var sections = document.querySelectorAll('#overview, #method, #results, #goal-reaching, #BibTeX');
    if ('IntersectionObserver' in window) {
      var sectionObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          Array.prototype.forEach.call(navLinks, function (link) {
            link.classList.toggle('is-active', link.getAttribute('href') === '#' + entry.target.id);
          });
        });
      }, { rootMargin: '-28% 0px -62% 0px', threshold: 0 });

      Array.prototype.forEach.call(sections, function (section) {
        sectionObserver.observe(section);
      });
    }

    updateScrollChrome();
    window.addEventListener('scroll', updateScrollChrome, { passive: true });
    window.addEventListener('resize', function () {
      if (window.innerWidth > 768) closeMenu();
      updateScrollChrome();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPageChrome);
  } else {
    initPageChrome();
  }
})();
