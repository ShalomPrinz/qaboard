#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pillow>=10.1", "pyyaml", "tifffile"]
# ///
"""
Makes the stand-ins of the pipeline shots (commit/pipeline, commit/pipeline-step, commit/pipeline-running,
commit/pipeline-failed), without the demo stack: no demo project has a pipeline yet.

1. It runs the camera ISP example (examples/image-quality, its qa/main.py's run()) on its charts, like the steps of a
   `calibration` pipeline would: calibration-images (the color chart and the flat gray at the 3 ISOs), calibrate (the
   sensor's noise profile: a fit of the raw noise in the gray patches, per ISO), evaluation-images (4 charts at 2 ISOs) and report (the means).
2. It takes pictures of the web app's production build (vite preview) with the API mocked by Playwright, on those
   runs' metrics and plots: webapp/e2e/standins/pipelines.standin.js.
3. It saves them as standins/commit/<shot>.webp (and -dark.webp) and prints their size: update `width`, `height`
   and the numbers of `alt` and `caption` in ../shots.yaml.

  cd webapp && npm ci && npm run build && cd ..
  uv run docs-media/standins/pipelines.py            # about 2 minutes
  uv run docs-media/standins/pipelines.py --reuse    # only the pictures, from the last runs
  (PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chromium uses a browser that's already installed)

The runs, their durations and the steps' commands are this machine's, the pipeline's status files are made like
qaboard/pipelines.py makes them. Runs are kept in docs-media/out/standins/pipelines/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXAMPLE = REPO / 'examples' / 'image-quality'
WEBAPP = REPO / 'webapp'
OUT = HERE.parent / 'out' / 'standins' / 'pipelines'
# Where the runs' paths say they are: the example's storage (qaboard.yaml)
STORAGE = '/mnt/qaboard/kitefin/camera-isp/main'

CALIBRATION = [(chart, config) for config in ('base', 'low-light', 'night') for chart in ('color-chart', 'flat-gray')]
EVALUATION = [(chart, config) for config in ('base', 'low-light') for chart in ('slanted-edge', 'siemens-star', 'color-chart', 'zone-plate')]
ISOS = {'base': 100, 'low-light': 3200, 'night': 12800}


def load_example():
  """The example's qa/main.py, without downloading the photos or making the 113 MP wall on import"""
  sys.path.insert(0, str(EXAMPLE / 'qa'))
  import fetch_photos
  import make_charts
  fetch_photos.ensure_photos = lambda *args, **kwargs: None
  make_charts.ensure_charts = lambda *args, **kwargs: None
  import main
  return main, make_charts


def run_one(main, make_charts, chart, config, output_dir: Path, database: Path):
  path = database / 'charts' / f'{chart}.png'
  if not path.exists():
    make_charts.save(*make_charts.GENERATORS[chart](), path)
  import yaml
  shutil.rmtree(output_dir, ignore_errors=True)
  output_dir.mkdir(parents=True)
  context = SimpleNamespace(
    input_path=path, rel_input_path=Path('charts') / path.name, database=database, configs=[config], params={},
    output_dir=output_dir, platform='linux', input_metadata=yaml.safe_load(path.with_suffix('.metadata.yaml').read_text()),
  )
  start = time.time()
  metrics = main.run(context)
  metrics['runtime'] = round(time.time() - start, 2)
  (output_dir / 'metrics.json').write_text(json.dumps(metrics))
  return metrics


def fit_noise(runs):
  """
  The calibrate step: the sensor's noise profile (photon transfer curve), per ISO. On the green pixels of the raw
  image in the gray patches (camera.capture(), with the run's seed): variance = k * signal + read_noise^2, in DN
  above the black level. k is the conversion gain, in DN per electron.
  """
  import camera
  profile, traces = {}, []
  for config in ISOS:
    signal, variance = [], []
    for run in runs:
      if run['config'] != config:
        continue
      path = OUT / 'inputs' / 'charts' / f"{run['chart']}.png"
      scene = np.asarray(Image.open(path).convert('RGB'))
      raw = camera.capture(scene, config, f'0:charts/{path.name}')
      green = camera.cfa_channel(*raw.mosaic.shape) == 1
      meta = run['metadata']['measure']
      boxes = [(p, 0.25) for p in meta.get('patches', []) if p['name'] in ('white', 'black') or p['name'].startswith('neutral')]
      boxes += [(meta['flat'], 0)] if 'flat' in meta else []
      for box, inset in boxes:
        dx, dy = int(box['w'] * inset), int(box['h'] * inset)
        window = (slice(box['y'] + dy, box['y'] + box['h'] - dy), slice(box['x'] + dx, box['x'] + box['w'] - dx))
        values = raw.mosaic[window][green[window]].astype(np.float64) - camera.BLACK_LEVEL
        signal.append(float(values.mean()))
        variance.append(float(values.var()))
    order = np.argsort(signal)
    s, v = np.array(signal)[order], np.array(variance)[order]
    k, b = (float(x) for x in np.polyfit(s, v, 1))
    b = max(b, 0.0)
    profile[config] = {'iso': ISOS[config], 'conversion_gain': round(k, 4), 'read_noise': round(float(np.sqrt(b)), 2)}
    traces.append((config, s, v, k, b))
  colors = {'base': '#238551', 'low-light': '#2d72d2', 'night': '#c87619'}
  data = []
  for config, s, v, k, b in traces:
    xs = np.geomspace(s.min() * 0.8, s.max() * 1.15, 40)
    data.append({'type': 'scatter', 'mode': 'markers', 'name': f'ISO {ISOS[config]}', 'legendgroup': config,
                 'x': [round(float(x), 1) for x in s], 'y': [round(float(np.sqrt(y)), 2) for y in v], 'marker': {'color': colors[config], 'size': 8}})
    data.append({'type': 'scatter', 'mode': 'lines', 'name': f'fit, ISO {ISOS[config]}', 'legendgroup': config, 'showlegend': False,
                 'x': [round(float(x), 1) for x in xs], 'y': [round(float(np.sqrt(k * x + b)), 2) for x in xs],
                 'line': {'color': colors[config], 'width': 2, 'dash': 'dot'}})
  plot = {'data': data, 'layout': {
    'margin': {'l': 60, 'r': 20, 't': 40, 'b': 50}, 'height': 340,
    'title': {'text': 'Noise profile: raw noise in the gray patches, and its fit per ISO'},
    'xaxis': {'title': {'text': 'Signal, green pixels (DN above black)'}, 'type': 'log'},
    'yaxis': {'title': {'text': 'Noise, standard deviation (DN)'}, 'type': 'log'},
  }}
  return profile, plot


def measure():
  main, make_charts = load_example()
  database = OUT / 'inputs'
  runs = {}
  for step, todo in (('calibration-images', CALIBRATION), ('evaluation-images', EVALUATION)):
    runs[step] = []
    for chart, config in todo:
      folder = OUT / 'runs' / step / f'{chart}-{config}'
      print(f'{step}: charts/{chart}.png --config {config}', flush=True)
      metrics = run_one(main, make_charts, chart, config, folder, database)
      import yaml
      meta = yaml.safe_load((database / 'charts' / f'{chart}.metadata.yaml').read_text())
      runs[step].append({'chart': chart, 'config': config, 'dir': str(folder), 'metrics': metrics, 'metadata': meta,
                         'files': sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file())})
  start = time.time()
  profile, plot = fit_noise(runs['calibration-images'])
  calibrate_duration = time.time() - start
  ok = [r['metrics'] for r in runs['evaluation-images'] if not r['metrics'].get('is_failed')]
  def mean(key):
    values = [m[key] for m in ok if key in m]
    return round(float(np.mean(values)), 4)
  report = {
    'psnr': mean('psnr'), 'worst_psnr': round(min(m['psnr'] for m in ok), 3), 'ssim': mean('ssim'), 'mtf50': mean('mtf50'),
    'overshoot': mean('overshoot'), 'delta_e': mean('delta_e'), 'noise_flat': mean('noise_flat'),
  }
  import yaml
  # the example's metrics, with the one the report adds
  qatools_metrics = yaml.safe_load((EXAMPLE / 'qa' / 'metrics.yaml').read_text())
  qatools_metrics['available_metrics']['worst_psnr'] = {
    **qatools_metrics['available_metrics']['psnr'], 'label': 'Worst PSNR', 'short_label': 'Worst PSNR',
    'description': 'The lowest PSNR of the evaluation runs.', 'target': 18}
  qatools_metrics['summary_metrics'] = ['psnr', 'worst_psnr', 'ssim', 'mtf50', 'overshoot', 'noise_flat']
  qatools_metrics['main_metrics'] = ['psnr', 'worst_psnr', 'ssim', 'overshoot', 'mtf50', 'runtime']
  data = {'runs': runs, 'profile': profile, 'noise_profile_plot': plot, 'report': report, 'qatools_metrics': qatools_metrics,
          'durations': {step: round(sum(r['metrics']['runtime'] for r in rs), 1) for step, rs in runs.items()} | {'calibrate': round(calibrate_duration, 2)}}
  (OUT / 'data.json').write_text(json.dumps(data, indent=1))
  # the pipeline's own files: what calibrate and report write in {output_dir}
  pipeline = OUT / 'pipeline'
  pipeline.mkdir(exist_ok=True)
  (pipeline / 'calibration.yaml').write_text(yaml.safe_dump({'noise_profile': profile}, sort_keys=False))
  (pipeline / 'noise-profile.plotly.json').write_text(json.dumps(plot, separators=(',', ':')))
  (pipeline / 'metrics.json').write_text(json.dumps(report))
  print(json.dumps({'profile': profile, 'report': report, 'durations': data['durations']}, indent=1))


def pictures():
  shutil.rmtree(OUT / 'pictures', ignore_errors=True)
  env = {**os.environ, 'STANDINS': '1', 'STANDINS_DATA': str(OUT), 'E2E_PORT': os.environ.get('E2E_PORT', '4183')}
  subprocess.run(['npx', 'playwright', 'test', 'e2e/standins/pipelines.standin.js'], cwd=WEBAPP, env=env, check=True)
  for png in sorted((OUT / 'pictures').glob('*.png')):
    image = Image.open(png)
    webp = HERE / 'commit' / f'{png.stem}.webp'
    image.save(webp, quality=90, method=6)
    print(f'{webp.relative_to(REPO)}: width {image.width // 2}, height {image.height // 2}, {webp.stat().st_size // 1000} kB')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--reuse', action='store_true', help='only the pictures, from the last runs')
  args = parser.parse_args()
  if not args.reuse:
    measure()
  pictures()
