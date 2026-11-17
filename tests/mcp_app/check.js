// Drives the app of `qa mcp` in a host that uses the official AppBridge, and fails if the protocol breaks:
// the handshake, sizing, links, messages to the chat, and tool calls from the app. Screenshots in ./screenshots/.
// Start `python harness.py` first (see README.md).
const { chromium } = require('playwright');
const assert = require('assert');
const fs = require('fs');
const HOST = process.env.HOST || 'http://127.0.0.1:5199';
const NEW = 'a1b2c3d4' + '0'.repeat(32);

(async () => {
  const fake = await (await fetch(`${HOST}/fake`)).text();
  fs.mkdirSync('screenshots', { recursive: true });
  const browser = await chromium.launch();
  async function open(tool, args, theme = 'light') {
    const page = await browser.newPage({ viewport: { width: 780, height: 900 } });
    const errors = [];
    page.on('console', m => { if (['error', 'warning'].includes(m.type())) errors.push(m.text()); });
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(`${HOST}/host.html?tool=${tool}&theme=${theme}&args=${encodeURIComponent(JSON.stringify(args))}`);
    await page.waitForFunction(() => window.events?.some(e => e.type === 'result-sent'), null, { timeout: 30000 });
    await page.waitForTimeout(500);
    const events = () => page.evaluate(() => window.events);
    return { page, frame: page.frames()[1], errors, events };
  }

  // compare: links, a question to the chat, then the app asks the server for a test's images
  let { page, frame, errors, events } = await open('compare', { url: `${fake}/group/repo/commit/${NEW}` });
  assert.ok((await events()).some(e => e.type === 'initialized'), 'the app initializes with the host');
  assert.ok((await events()).some(e => e.type === 'size'), 'the app tells the host its size');
  await frame.click('text=Open the Metrics Diff');
  await frame.click('text=Ask about the worst regression');
  await frame.click('td.name >> text=night/02');
  await frame.waitForSelector('#image figure img', { timeout: 30000 });
  await page.screenshot({ path: 'screenshots/compare.png', fullPage: true });
  let seen = await events();
  assert.ok(seen.some(e => e.type === 'open-link' && e.url.includes('selected_views=table-compare')));
  assert.ok(seen.some(e => e.type === 'message' && e.content[0].text.includes('night/02')));
  assert.deepStrictEqual(seen.filter(e => e.type === 'tools/call').map(e => e.arguments), [{ run_id: 2 }, { run_id: 2, path: 'output.png', reference_run_id: 12 }]);
  assert.deepStrictEqual(errors, []);

  // metric_history, in the dark
  ({ page, frame, errors, events } = await open('metric_history', { project: 'group/repo', label: 'nightly' }, 'dark'));
  assert.strictEqual(await frame.locator('circle').count(), 3);
  await frame.click('text=/Ask about a1b2c3d4/');
  await page.screenshot({ path: 'screenshots/history-dark.png', fullPage: true });
  assert.ok((await events()).some(e => e.type === 'message' && e.content[0].text.includes('compare')));
  assert.deepStrictEqual(errors, []);

  // look_at_output: new and reference, whole and where they differ
  ({ page, frame, errors } = await open('look_at_output', { run_id: 2, path: 'output.png' }));
  assert.strictEqual(await frame.locator('figure img').count(), 4);
  await page.screenshot({ path: 'screenshots/images.png', fullPage: true });
  assert.deepStrictEqual(errors, []);

  // errors are shown
  ({ frame } = await open('look_at_output', { run_id: 999 }));
  assert.match(await frame.locator('pre').innerText(), /No run 999/);

  await browser.close();
  console.log('OK: the app works with the official AppBridge');
})().catch(e => { console.error(e); process.exit(1); });
