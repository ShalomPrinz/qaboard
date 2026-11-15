// The docs' pictures of pipelines on the commit page (stand-ins, see docs-media/standins/README.md): the production
// build with the API and the files mocked, on real runs of the camera ISP example made by docs-media/standins/pipelines.py.
//   uv run docs-media/standins/pipelines.py      (runs this file with STANDINS=1, then saves the pictures as .webp)
import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';

const DATA = process.env.STANDINS_DATA ?? path.resolve('../docs-media/out/standins/pipelines');
const PICTURES = path.join(DATA, 'pictures');
const data = JSON.parse(fs.readFileSync(path.join(DATA, 'data.json'), 'utf8'));

const project = 'kitefin/camera-isp';
const hexsha = '3f9c2e1a7b64d05e8c21f4a9be3d7c560a18e2f4';
const reference = 'b81d04c95e2f7a3361c0d8e4f9a25b7c0e6d1a93';
const storage =`/mnt/qaboard/${project}/${hexsha.slice(0, 8)}`;
const pipeline_dir = `${storage}/calibration`;
const qatools_metrics = data.qatools_metrics;
const qatools_config = {
  project: { name: project, reference_branch: 'main' },
  outputs: {
    visualizations: [
      { name: 'Noise profile', path: 'noise-profile.plotly.json', side_by_side: false },
      { name: 'MTF', path: 'mtf.plotly.json', side_by_side: false },
      { name: 'Color errors', path: 'color.plotly.json', side_by_side: false },
    ],
  },
};
const steps = [
  { batch: 'calibration-images', label: 'calibration' },
  { run: 'python qa/calibrate.py {steps.calibration-images.output_dirs} --output {output_dir}/calibration.yaml' },
  { batch: 'evaluation-images', configs: ['{output_dir}/calibration.yaml'] },
  { run: 'python qa/report.py {steps.evaluation-images.runs_file} {output_dir}' },
];
// with a step that publishes the calibration, allowed to fail
const steps_publish = [...steps, { run: 'python qa/publish_profile.py {output_dir}/calibration.yaml', allow_failure: true }];

// the runs: their folder in the mocked storage -> their folder here
const folders = { [pipeline_dir]: path.join(DATA, 'pipeline') };
const run_dir = (label, run) => `${storage}/${label}/charts/${run.chart}.png/${run.config}`;
const output = (id, fields) => ({
  id, platform: 'linux', configurations: ['base'], extra_parameters: {}, metrics: {}, data: {},
  is_failed: false, is_pending: false, is_running: false, deleted: false,
  created_date: '2026-11-12T09:41:00Z',
  test_input_database: `/mnt/qaboard/${project}/inputs`, test_input_metadata: {},
  ...fields,
});
const run_output = (id, label, batch, run, fields = {}) => {
  const dir = run_dir(label, run);
  folders[dir] = run.dir;
  return output(id, {
    test_input_path: `charts/${run.chart}.png`, configurations: batch === 'evaluation-images' ? [run.config, `${pipeline_dir}/calibration.yaml`] : [run.config],
    output_dir_url: `/s${dir}`, metrics: run.metrics, data: { batch }, ...fields,
  });
};

// state: done | running | publish-failed
const scenario = state => {
  const calibration = data.runs['calibration-images'];
  const evaluation = data.runs['evaluation-images'];
  const order = ['slanted-edge-low-light', 'color-chart-low-light']; // shown right after the pipeline
  const rank = run => { const i = order.indexOf(`${run.chart}-${run.config}`); return i === -1 ? 99 : i; };
  const sorted = [...evaluation].sort((a, b) => rank(a) - rank(b));
  const total = Object.values(data.durations).reduce((a, b) => a + b, 0);
  const pipeline = output(2000, {
    output_type: 'pipeline', test_input_database: '.', test_input_path: 'PIPELINE',
    configurations: state === 'publish-failed' ? steps_publish : steps, data: { batch: 'calibration' },
    output_dir_url: `/s${pipeline_dir}`,
    metrics: state === 'running' ? {} : { ...data.report, runtime: total + 2.1, is_failed: state === 'publish-failed' },
    is_running: state === 'running', is_failed: state === 'publish-failed',
  });
  const evaluation_outputs = Object.fromEntries(sorted.map((run, i) => {
    // while the pipeline runs: 3 runs done, 2 running, the others waiting
    const fields = state !== 'running' ? {} : i < 3 ? {} : i < 5 ? { is_running: true, metrics: {} } : { is_pending: true, metrics: {} };
    return [1999 - i, run_output(1999 - i, 'default', 'evaluation-images', run, fields)];
  }));
  const calibration_dirs = calibration.map(run => run_dir('calibration', run));
  const count = outputs => ({
    valid_outputs: Object.values(outputs).filter(o => !o.is_running && !o.is_pending && !o.is_failed).length,
    running_outputs: Object.values(outputs).filter(o => o.is_running).length,
    pending_outputs: Object.values(outputs).filter(o => o.is_pending).length,
    failed_outputs: Object.values(outputs).filter(o => o.is_failed).length,
    deleted_outputs: 0,
  });
  const defaults = { 2000: pipeline, ...evaluation_outputs };
  const calibrations = Object.fromEntries(calibration.map((run, i) => [1900 - i, run_output(1900 - i, 'calibration', 'calibration-images', run)]));
  const commit = {
    id: hexsha, hexsha, project_id: project, branch: 'noise-profile', committer_name: 'Ines Marchetti',
    message: 'calibration: fit the noise profile per ISO from the gray patches',
    authored_datetime: '2026-11-12T09:32:00Z', committed_datetime: '2026-11-12T09:32:00Z',
    latest_output_datetime: '2026-11-12T09:44:00Z', parents: [], deleted: false,
    data: { qatools_config, qatools_metrics },
    batches: {
      default: { id: 1, label: 'default', data: {}, ...count(defaults), outputs: defaults },
      calibration: { id: 2, label: 'calibration', data: {}, ...count(calibrations), outputs: calibrations },
    },
  };
  const d = data.durations;
  const done = (fields) => ({ status: 'success', ...fields });
  const pipeline_steps = [
    done({ name: 'calibration-images', type: 'batch', batches: ['calibration-images'], label: 'calibration', runs: 6, failed_runs: 0, cached_runs: 0, duration: d['calibration-images'] + 3.4, runs_file: 'steps/calibration-images/runs.json',
      command: 'qa --label calibration batch calibration-images --batches-file qa/batches.yaml --action-on-existing=run --action-on-pending=sync' }),
    done({ name: 'calibrate', type: 'run', returncode: 0, duration: d.calibrate,
      command: `python qa/calibrate.py ${calibration_dirs.join(' ')} --output ${pipeline_dir}/calibration.yaml` }),
    done({ name: 'evaluation-images', type: 'batch', batches: ['evaluation-images'], label: 'default', runs: 8, failed_runs: 0, cached_runs: 0, duration: d['evaluation-images'] + 3.6, runs_file: 'steps/evaluation-images/runs.json',
      command: `qa --label default --tuning '["${pipeline_dir}/calibration.yaml"]' batch evaluation-images --batches-file qa/batches.yaml --action-on-existing=run --action-on-pending=sync` }),
    done({ name: 'report', type: 'run', returncode: 0, duration: 0.4,
      command: `python qa/report.py ${pipeline_dir}/steps/evaluation-images/runs.json ${pipeline_dir}` }),
  ];
  if (state === 'running') {
    // what a batch step has before its runs are done
    const { name, type, batches, label, command } = pipeline_steps[2];
    pipeline_steps[2] = { name, type, batches, label, command, status: 'running', started_at: '2026-11-12T09:43:51Z' };
    pipeline_steps[3] = { name: 'report', type: 'run', status: 'pending', commands: [steps[3].run] };
  }
  if (state === 'publish-failed')
    pipeline_steps.push({ name: 'publish_profile', type: 'run', status: 'failed', allow_failure: true, returncode: 1, duration: 30.1,
      command: `python qa/publish_profile.py ${pipeline_dir}/calibration.yaml` });
  const pipeline_json = {
    version: 1, status: state === 'running' ? 'running' : state === 'publish-failed' ? 'failed' : 'success',
    started_at: '2026-11-12T09:43:30Z', ...(state === 'running' ? {} : { ended_at: '2026-11-12T09:44:12Z' }),
    steps: pipeline_steps,
  };
  return { commit, pipeline_json };
};

const manifest = dir => {
  const files = {};
  const walk = (folder, prefix) => {
    if (!fs.existsSync(folder)) return;
    for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(path.join(folder, entry.name), rel);
      else files[rel] = { st_size: fs.statSync(path.join(folder, entry.name)).size, md5: rel };
    }
  };
  walk(dir, '');
  return files;
};

const errors = [];
const setup = async (page, state, { running_files = false } = {}) => {
  const { commit, pipeline_json } = scenario(state);
  const user = { is_authenticated: true, user_id: 1, user_name: 'ines', full_name: 'Ines Marchetti', email: 'ines@kitefin.example', login_type: 'local' };
  const api = {
    '/api/v1/config': {},
    '/api/v1/user/me/': user,
    '/api/v1/projects': { [project]: { data: { qatools_config, qatools_metrics }, latest_commit_datetime: commit.latest_output_datetime, total_commits: 214 } },
    '/api/v1/project': { qatools_config, qatools_metrics },
    '/api/v1/project/branches': ['main', 'noise-profile'],
    [`/api/v1/commit/${hexsha}`]: commit,
    // the reference, on main: the pipeline is new, main has no run to compare with
    '/api/v1/commit': {
      ...commit, id: reference, hexsha: reference, branch: 'main', committer_name: 'Tobias M. Wendt', message: 'fxp: 2 guard bits in bilateral LUT',
      authored_datetime: '2026-11-11T14:05:00Z', committed_datetime: '2026-11-11T14:05:00Z', latest_output_datetime: '2026-11-11T14:20:00Z', batches: {},
    },
  };
  await page.addInitScript(() => localStorage.setItem('qaboard.release-notes.last-seen', '9999-12-31'));
  await page.route('**/api/**', route => {
    const { pathname } = new URL(route.request().url());
    const key = Object.keys(api).sort((a, b) => b.length - a.length).find(k => pathname.startsWith(k));
    return route.fulfill({ json: key ? api[key] : {} });
  });
  await page.route('**/s/**', route => {
    const pathname = decodeURIComponent(new URL(route.request().url()).pathname).replace(/^\/s/, '');
    const dir = Object.keys(folders).sort((a, b) => b.length - a.length).find(d => pathname.startsWith(`${d}/`));
    if (!dir) return route.fulfill({ status: 404 });
    const file = pathname.slice(dir.length + 1);
    if (dir === pipeline_dir && file === 'pipeline.json') return route.fulfill({ json: pipeline_json });
    const local = folders[dir];
    if (file === 'manifest.outputs.json') {
      const files = manifest(local);
      if (dir === pipeline_dir) files['pipeline.json'] = { st_size: JSON.stringify(pipeline_json).length, md5: 'pipeline' };
      if (dir === pipeline_dir && state === 'running' && !running_files) delete files['metrics.json'];
      return route.fulfill({ json: files });
    }
    const target = path.join(local, file);
    if (!fs.existsSync(target)) return route.fulfill({ status: 404 });
    return route.fulfill({ path: target });
  });
  await page.goto(`/${project}/commit/${hexsha}?selected_views=output-list`);
  const cards = page.locator('.output-card[data-output-id]');
  await expect(cards.first()).toHaveAttribute('data-output-id', '2000');
  await expect(cards.first().getByTestId('pipeline-step').first()).toBeVisible();
  // the plots are drawn, nothing is loading
  await expect(page.locator('.output-card .js-plotly-plot').first()).toBeVisible({ timeout: 20000 });
  // (while the pipeline runs, it and its runs have theirs)
  if (state !== 'running')
    await expect(page.locator('.output-card .bp6-spinner')).toHaveCount(0, { timeout: 20000 });
  await page.waitForTimeout(800);
  return cards;
};

// a picture of the union of boxes, with a margin, at most the page's width
const shoot = async (page, name, boxes, margin = 12) => {
  const all = (await Promise.all(boxes.map(b => b.boundingBox()))).filter(Boolean);
  const x = Math.max(0, Math.min(...all.map(b => b.x)) - margin);
  const y = Math.max(0, Math.min(...all.map(b => b.y)) - margin);
  const right = Math.max(...all.map(b => b.x + b.width)) + margin;
  const bottom = Math.max(...all.map(b => b.y + b.height)) + Math.min(margin, 3);
  fs.mkdirSync(PICTURES, { recursive: true });
  await page.screenshot({ path: path.join(PICTURES, `${name}.png`), clip: { x, y, width: right - x, height: bottom - y }, animations: 'disabled' });
};

// the card's width, from its top down to the bottom of `below`
const shoot_card = async (page, name, card, below, margin = 12) => {
  const box = await card.boundingBox();
  const bottom = Math.max(...(await Promise.all(below.map(b => b.boundingBox()))).map(b => b.y + b.height)) + margin;
  fs.mkdirSync(PICTURES, { recursive: true });
  await page.screenshot({ path: path.join(PICTURES, `${name}.png`), clip: { x: box.x - margin, y: box.y - margin, width: box.width + 2 * margin, height: bottom - box.y + margin }, animations: 'disabled' });
};

for (const theme of ['light', 'dark']) {
  const suffix = theme === 'dark' ? '-dark' : '';
  test.describe(theme, () => {
    test.use({ viewport: { width: 1280, height: 1800 }, deviceScaleFactor: 2, colorScheme: theme });
    test.beforeEach(async ({ page }) => {
      errors.length = 0;
      page.on('console', msg => msg.type() === 'error' && errors.push(msg.text()));
      page.on('pageerror', error => errors.push(error.message));
    });
    test.afterEach(() => {
      // the runs without some files answer 404, which the browser logs
      expect(errors.filter(e => !e.includes('404')), 'errors in the browser console').toEqual([]);
    });

    test(`pipeline ${theme}`, async ({ page }) => {
      const cards = await setup(page, 'done');
      await shoot(page, `pipeline${suffix}`, [cards.nth(0), cards.nth(1)], 6);
    });

    test(`pipeline-step ${theme}`, async ({ page }) => {
      const cards = await setup(page, 'done');
      await cards.first().getByTestId('pipeline-step').nth(0).click();
      const runs = page.locator('.bp6-popover').filter({ hasText: 'Show its 6 runs' });
      await expect(runs).toBeVisible();
      await page.mouse.move(0, 0);
      await page.waitForTimeout(300);
      await shoot_card(page, `pipeline-step${suffix}`, cards.first(), [runs]);
    });

    test(`pipeline-running ${theme}`, async ({ page }) => {
      const cards = await setup(page, 'running');
      await shoot_card(page, `pipeline-running${suffix}`, cards.first(), [cards.first().getByTestId('pipeline-steps')]);
    });

    test(`pipeline-failed ${theme}`, async ({ page }) => {
      const cards = await setup(page, 'publish-failed');
      await cards.first().getByTestId('pipeline-step').nth(4).click();
      const popover = page.locator('.bp6-popover').filter({ hasText: 'publish_profile' });
      await expect(popover).toBeVisible();
      await page.mouse.move(0, 0);
      await page.waitForTimeout(300);
      await shoot_card(page, `pipeline-failed${suffix}`, cards.first(), [popover]);
    });
  });
}
