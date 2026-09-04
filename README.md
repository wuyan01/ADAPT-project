# ADAPT — project page

Project website for **ADAPT: Agile Diffusion Action Priors for Robust and Steerable Online
Text-Driven Humanoid Control** (Yan Wu, Chenhao Li, Kaifeng Zhao\*, Gen Li\*, Marco Hutter,
Siyu Tang — ETH Zurich).

Built on the [Nerfies](https://github.com/nerfies/nerfies.github.io) template, same as the
UniPhys project page.

## Layout

```
index.html                   the whole page
static/css/adapt.css         ADAPT-specific styling (captions, video cards, placeholders)
static/js/adapt.js           lazy video loading via IntersectionObserver
static/images/               figures extracted from the paper
static/videos/               web-sized clips (see below)
static/paper.pdf             the paper
```

## Videos

`static/videos/` holds web-sized clips committed to the repo, about 51 MB in total:

- `atomic-clips/*.mp4` — single-skill real-world clips (the carousel)
- `sequential-clips/*.mp4` — real-world prompt-switching sequences (the carousel)
- `long-rollouts/*.mp4` — uncut long-horizon rollouts, one in simulation and one on hardware
- `goal-reaching/*.mp4` — goal reaching with a commanded motion style

They were transcoded from the raw capture footage in `../assets/video` (1.7 GB, some files
over GitHub's 100 MB per-file limit) down to 720p H.264, audio stripped, with a faststart
header. To regenerate or add clips, re-encode with `ffmpeg`:

```sh
ffmpeg -i in.mp4 -map 0:v:0 -vf "scale=-2:'min(720,ih)'" -r 30 \
  -c:v libx264 -preset slow -crf 30 -pix_fmt yuv420p \
  -profile:v high -level 4.0 -movflags +faststart -an out.mp4
```

`-map 0:v:0` matters for `.mov` captures, which carry extra metadata tracks, and those are
often HEVC, which most browsers cannot decode — so always write H.264 `.mp4`.

Videos load lazily and only play when scrolled into view, so page weight stays low. Because
`bulma-carousel` clones slides in `infinite` mode, `static/js/index.js` calls
`window.adaptRefreshLazyVideos()` after attaching the carousels so the clones load too.

The teaser is a YouTube embed rather than a local file.

## TODOs before publishing

Search `index.html` for `TODO`:

- arXiv link and id (header button + BibTeX)
- code repository link

## Local preview

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy

GitHub Pages from the `main` branch, root directory. `.nojekyll` is present so paths
beginning with `_` and `static/` are served verbatim.
