// Several references shown together: the image viewer shows the new image and every ticked reference, each in its
// tile, compared with an anchor. Against a mocked API and IIIF image server.
import { test, expect } from '@playwright/test';

import { info, tile } from './fixtures/iiif';

const project = 'group/repo';
const qatools_config = { outputs: { visualizations: [{ name: 'Output', path: 'image.png' }] } };

// A gradient: "new" has a dark square, "old" is brighter, "alt" is the same file as "ref"
const size = { width: 512, height: 384 };
const gradient = (x, y) => [x >> 1, y >> 1, 128];
const images = {
  new: { ...size, pixel: (x, y) => x >= 100 && x < 140 && y >= 100 && y < 140 ? [20, 20, 20] : gradient(x, y) },
  ref: { ...size, pixel: gradient },
  old: { ...size, pixel: (x, y) => gradient(x, y).map(v => Math.min(255, v + 30)) },
  alt: { ...size, pixel: gradient },
};
const data_hash = { new: 'new', ref: 'ref', old: 'old', alt: 'ref' };
const runs = ['A', 'B'];
const output = (id, hexsha, run) => ({
  id, platform: 'linux', configurations: ['base'], extra_parameters: {}, metrics: {}, data: {},
  is_failed: false, is_pending: false, is_running: false, deleted: false, created_date: '2026-09-01T10:00:00Z',
  output_dir_url: `/s/mnt/qaboard/${hexsha}-${run}/output`,
  test_input_database: '/mnt/database', test_input_path: `images/${run}.png`, test_input_metadata: {},
});
const commit = (hexsha, batch_id, message) => ({
  id: hexsha, hexsha, project_id: project, branch: 'master', committer_name: 'Alice', message,
  authored_datetime: '2026-09-01T10:00:00Z', committed_datetime: '2026-09-01T10:00:00Z',
  latest_output_datetime: '2026-09-01T10:00:00Z', parents: [], deleted: false, data: { qatools_config },
  batches: { default: {
    id: batch_id, label: 'default', data: {},
    valid_outputs: runs.length, pending_outputs: 0, running_outputs: 0, failed_outputs: 0, deleted_outputs: 0,
    outputs: Object.fromEntries(runs.map((run, i) => [batch_id * 100 + i, output(batch_id * 100 + i, hexsha, run)])),
  } },
});
const commits = { new: commit('new', 1, 'New'), ref: commit('ref', 2, 'Reference'), old: commit('old', 3, 'Older'), alt: commit('alt', 4, 'Alternative') };
const hexshaOf = folder => folder.split('-')[0];

const errors = [];
test.beforeEach(async ({ page }) => {
  errors.length = 0;
  page.on('console', msg => msg.type() === 'error' && errors.push(msg.text()));
  page.on('pageerror', error => errors.push(error.message));
  const api = {
    '/api/v1/config': {},
    '/api/v1/user/me/': { is_authenticated: true, user_id: 1, user_name: 'alice', full_name: 'Alice', email: 'alice@example.com', login_type: 'local' },
    '/api/v1/projects': { [project]: { data: { qatools_config }, latest_commit_datetime: '2026-09-01T10:00:00Z', total_commits: 4 } },
    '/api/v1/project': { qatools_config },
    '/api/v1/project/branches': ['master'],
  };
  await page.addInitScript(() => localStorage.setItem('qaboard.release-notes.last-seen', '9999-12-31'));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const id = url.pathname.match(/^\/api\/v1\/commit\/(\w+)$/)?.[1];
    if (id && commits[id]) {
      // with its runs, the older reference takes a while: its tiles wait for it
      if (id === 'old' && !url.searchParams.has('summary') && !/outputs=(false|0)/.test(url.search)) await new Promise(resolve => setTimeout(resolve, 700));
      return route.fulfill({ json: commits[id] });
    }
    const key = Object.keys(api).sort((a, b) => b.length - a.length).find(k => url.pathname.startsWith(k));
    return route.fulfill({ json: key ? api[key] : {} });
  });
  await page.route('**/s/**', route => {
    const { pathname } = new URL(route.request().url());
    const hexsha = hexshaOf(pathname.split('/')[4]);
    if (pathname.endsWith('/manifest.outputs.json'))
      return route.fulfill({ json: { 'image.png': { st_size: 1000, md5: hexsha, md5_data: data_hash[hexsha] } } });
    return route.fulfill({ status: 404 });
  });
  await page.route('**/iiif/2/**', route => {
    const { pathname } = new URL(route.request().url());
    const [identifier, ...request] = pathname.replace(/^\/iiif\/2\//, '').split('/');
    const [, , folder] = decodeURIComponent(identifier).split('/');
    const image = images[hexshaOf(folder)];
    if (request.join('/') === 'info.json') return route.fulfill({ json: info(identifier, image) });
    return route.fulfill({ body: tile(request.join('/'), image), contentType: 'image/png' });
  });
});
test.afterEach(() => {
  expect(errors, 'errors in the browser console').toEqual([]);
});

test('ticks references to see them together in the image viewer, compared with an anchor', async ({ page }) => {
  await page.setViewportSize({ width: 1500, height: 1000 });
  await page.goto(`/${project}/commit/new?reference=ref&compare=old&compare=alt&selected_views=output-list`);
  const viewer = page.locator('.qiv').first();
  await expect(viewer.locator('.qiv-pane')).toHaveCount(2);
  await expect(viewer.locator('.qiv-tile')).toHaveCount(0);
  await expect(page.getByRole('tab')).toHaveCount(3);

  // One more reference: its tile is there before its runs
  await page.getByRole('checkbox', { name: 'Show ref 2 together with ref' }).click();
  await expect(page).toHaveURL(/together/);
  await expect(viewer.locator('.qiv-tile')).toHaveCount(2);
  await expect(viewer.getByText('Loading ref 2…')).toBeVisible();
  const stage = await viewer.locator('.qiv-stage').boundingBox();
  await expect(viewer.getByText('Loading ref 2…')).toHaveCount(0);
  // ...and nothing moved when they came
  expect(await viewer.locator('.qiv-stage').boundingBox()).toEqual(stage);

  // All of them: new (the anchor), ref, ref 2, and ref 3, which says it's the same file as ref
  await page.getByRole('button', { name: 'All 3 together' }).click();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(3);
  await expect(viewer.locator('.qiv-label')).toHaveText([/^newanchor/, /^ref\d/, /^ref 2\d/, /^ref 3 = ref\d/]);
  // how far each image is from the anchor: the square, the brightness
  await expect(viewer.locator('.qiv-tile-note')).toHaveText([/^Δ new: 0\.\d$/, /^Δ new: \d\d\.\d$/, /^Δ new: 0\.\d$/]);
  await expect(viewer.getByTitle('ref and ref 3 = ref: the same in this view. Click to compare them alone')).toHaveText('=');

  // The modes compare each image with the anchor
  await page.mouse.move(stage.x + 50, stage.y + 50);
  await page.keyboard.press('4');
  await expect(viewer.locator('.qiv-tile .qiv-diff')).toHaveCount(3);
  await page.keyboard.press('2');
  await expect(viewer.getByRole('separator')).toHaveCount(3);
  // one at a time, in every run's viewer
  await page.keyboard.press('3');
  await expect(viewer.getByRole('radio', { name: /^1 new/ })).toBeChecked();
  await page.keyboard.press('t');
  await expect(viewer.getByRole('radio', { name: /^2 ref/ })).toBeChecked();
  await expect(page.locator('.qiv').nth(1).getByRole('radio', { name: /^2 ref/ })).toBeChecked();
  await page.keyboard.press('Shift+Digit3');
  await expect(viewer.getByRole('radio', { name: /^3 ref 2/ })).toBeChecked();
  await page.keyboard.press('1');

  // Another anchor, for every run
  await viewer.getByRole('button', { name: /^ref 2/ }).click();
  await expect(viewer.locator('.qiv-pane .qiv-label')).toHaveText(/^ref 2anchor/);
  await expect(page.locator('.qiv').nth(1).locator('.qiv-pane .qiv-label')).toHaveText(/^ref 2anchor/);
  await expect(viewer.locator('.qiv-tile-note').first()).toHaveText(/^Δ ref 2: \d\d\.\d$/);

  // A pair alone: the viewer for two images, then all of them again
  await viewer.getByTitle(/^new and ref[.:]/).click();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(0);
  await expect(viewer.getByRole('button', { name: 'Blind test' })).toBeVisible();
  await viewer.getByRole('button', { name: 'All 4 images' }).click();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(3);
  await page.screenshot({ path: test.info().outputPath('together.png') });

  // One at a time again: the reference alone, like before
  await page.getByRole('button', { name: 'One at a time' }).click();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(0);
  await expect(page).not.toHaveURL(/together/);
  await page.waitForLoadState('networkidle');
});

test('shows the images of each run in one row, as a contact sheet', async ({ page }) => {
  await page.setViewportSize({ width: 1500, height: 1000 });
  await page.goto(`/${project}/commit/new?reference=ref&compare[0][commit]=old&compare[0][together]=1&selected_views=output-list`);
  const viewer = page.locator('.qiv').first();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(2);
  await expect(viewer.getByRole('radiogroup', { name: 'Compare' })).toBeVisible();

  await page.getByRole('button', { name: 'Show the view options' }).click();
  await page.getByText('Images as a contact sheet').click();
  // one row, without the toolbar until users ask for it
  await expect(viewer.getByRole('radiogroup', { name: 'Compare' })).toBeHidden();
  const tops = await viewer.locator('.qiv-tile').evaluateAll(tiles => tiles.map(tile => tile.getBoundingClientRect().top));
  expect(new Set(tops).size).toBe(1);
  await page.screenshot({ path: test.info().outputPath('contact-sheet.png') });
  await viewer.getByRole('button', { name: 'Show the toolbar' }).click();
  await expect(viewer.getByRole('radiogroup', { name: 'Compare' })).toBeVisible();
  // remembered
  await page.reload();
  await expect(viewer.locator('.qiv-tile')).toHaveCount(2);
  await expect(viewer.getByRole('radiogroup', { name: 'Compare' })).toBeHidden();
  await page.waitForLoadState('networkidle');
});
