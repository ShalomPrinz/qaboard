// Pipelines on the commit page: their card comes first and shows their steps, against a mocked API.
import { test, expect } from '@playwright/test';

const project = 'group/repo';
const qatools_config = { outputs: { visualizations: [{ name: 'Report', path: 'report.txt' }] } };

const output = (id, fields) => ({
  id, platform: 'linux', configurations: ['base'], extra_parameters: {}, metrics: {}, data: {},
  is_failed: false, is_pending: false, is_running: false, deleted: false,
  created_date: '2026-09-01T10:00:00Z',
  output_dir_url: `/s/mnt/qaboard/out/${id}`,
  test_input_database: '/mnt/database', test_input_metadata: {},
  ...fields,
});
const steps = [
  { batch: 'calibration-images', label: 'calibration' },
  { name: 'calibrate', run: 'python calibrate.py {steps.calibration-images.output_dirs} --output {output_dir}/calibration.yaml' },
  { batch: 'evaluation', configs: ['{output_dir}/calibration.yaml'] },
  { run: 'python report.py {steps.evaluation.runs_file}' },
];
const outputs = {
  // the pipeline has the largest id: it's first all the same
  101: output(101, { test_input_path: 'images/A.png', data: { batch: 'evaluation' } }),
  102: output(102, { test_input_path: 'images/B.png', data: { batch: 'evaluation' } }),
  103: output(103, { output_type: 'pipeline', test_input_database: '.', test_input_path: 'PIPELINE', configurations: steps, data: { batch: 'calibration-pipeline' } }),
};
const commit = {
  id: 'abc123', hexsha: 'abc123', project_id: project, branch: 'master', committer_name: 'Alice', message: 'Tune the denoiser',
  authored_datetime: '2026-09-01T10:00:00Z', committed_datetime: '2026-09-01T10:00:00Z',
  latest_output_datetime: '2026-09-01T10:00:00Z', parents: [], deleted: false,
  data: { qatools_config },
  batches: {
    default: { id: 1, label: 'default', data: {}, valid_outputs: 3, pending_outputs: 0, running_outputs: 0, failed_outputs: 0, deleted_outputs: 0, outputs },
    calibration: { id: 2, label: 'calibration', data: {}, valid_outputs: 1, pending_outputs: 0, running_outputs: 0, failed_outputs: 0, deleted_outputs: 0, outputs: {
      201: output(201, { test_input_path: 'charts/C.png', data: { batch: 'calibration-images' } }),
    } },
  },
};
const pipeline = {
  version: 1,
  status: 'success',
  steps: [
    { name: 'calibration-images', type: 'batch', status: 'success', batches: ['calibration-images'], label: 'calibration', runs: 1, failed_runs: 0, cached_runs: 1, duration: 3.2, command: 'qa --label "calibration" batch "calibration-images" --action-on-existing=skip --action-on-pending=wait' },
    { name: 'calibrate', type: 'run', status: 'success', returncode: 0, duration: 12, command: 'python calibrate.py /mnt/qaboard/out/201 --output /mnt/qaboard/out/103/calibration.yaml' },
    { name: 'evaluation', type: 'batch', status: 'success', batches: ['evaluation'], label: 'default', runs: 2, failed_runs: 0, cached_runs: 0, duration: 95, command: 'qa batch "evaluation"' },
    { name: 'report', type: 'run', status: 'success', returncode: 0, duration: 0.4, command: 'python report.py /mnt/qaboard/out/103/steps/evaluation/runs.json' },
  ],
};

const errors = [];
test.beforeEach(async ({ page }) => {
  errors.length = 0;
  page.on('console', msg => msg.type() === 'error' && errors.push(msg.text()));
  page.on('pageerror', error => errors.push(error.message));
  const api = {
    '/api/v1/config': {},
    '/api/v1/user/me/': { is_authenticated: true, user_id: 1, user_name: 'alice', full_name: 'Alice', email: 'alice@example.com', login_type: 'local' },
    '/api/v1/projects': { [project]: { data: { qatools_config }, latest_commit_datetime: '2026-09-01T10:00:00Z', total_commits: 1 } },
    '/api/v1/project': { qatools_config },
    '/api/v1/project/branches': ['master'],
    '/api/v1/commit': commit,
  };
  await page.addInitScript(() => localStorage.setItem('qaboard.release-notes.last-seen', '9999-12-31'));
  await page.route('**/api/**', route => {
    const { pathname } = new URL(route.request().url());
    const key = Object.keys(api).sort((a, b) => b.length - a.length).find(k => pathname.startsWith(k));
    return route.fulfill({ json: key ? api[key] : {} });
  });
  await page.route('**/s/**', route => {
    const { pathname } = new URL(route.request().url());
    if (pathname.endsWith('/103/pipeline.json')) return route.fulfill({ json: pipeline });
    if (pathname.endsWith('/103/manifest.outputs.json')) return route.fulfill({ json: { 'report.txt': { st_size: 20, md5: 'a' }, 'pipeline.json': { st_size: 900, md5: 'b' } } });
    if (pathname.endsWith('/103/report.txt')) return route.fulfill({ body: 'PSNR +1.6 dB after calibration', contentType: 'text/plain' });
    if (pathname.endsWith('/manifest.outputs.json')) return route.fulfill({ json: {} });
    return route.fulfill({ status: 404 });
  });
});
test.afterEach(() => {
  // the runs without files answer 404, which the browser logs
  expect(errors.filter(e => !e.includes('404')), 'errors in the browser console').toEqual([]);
});


test('a pipeline is the first card, with its steps', async ({ page }) => {
  await page.goto(`/${project}/commit/abc123?selected_views=output-list`);
  const cards = page.locator('.output-card[data-output-id]');
  await expect(cards).toHaveCount(3);
  await expect(cards.first()).toHaveAttribute('data-output-id', '103');
  await expect(cards.first()).toContainText('calibration-pipeline (pipeline)');

  const pipeline_steps = cards.first().getByTestId('pipeline-step');
  await expect(pipeline_steps).toHaveCount(4);
  await expect(pipeline_steps.nth(0)).toContainText('calibration-images');
  await expect(pipeline_steps.nth(0)).toContainText('1 run · all reused');
  await expect(pipeline_steps.nth(2)).toContainText('2 runs');
  await expect(pipeline_steps.nth(2)).toContainText('1m 35s');
  // the other runs have no steps
  await expect(cards.nth(1).getByTestId('pipeline-step')).toHaveCount(0);

  // a step's details: its command, and its runs
  await pipeline_steps.nth(1).click();
  await expect(page.getByText('python calibrate.py /mnt/qaboard/out/201 --output /mnt/qaboard/out/103/calibration.yaml')).toBeVisible();
  await page.keyboard.press('Escape');
  await pipeline_steps.nth(0).click();
  await page.getByRole('button', { name: 'Show its 1 run' }).click();
  await expect(page).toHaveURL(/batch=calibration/);
  await expect(page).toHaveURL(/filter=calibration-images/);
  await expect(page.locator('.output-card[data-output-id="201"]')).toBeVisible();
});
