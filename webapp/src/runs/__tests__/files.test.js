/**
 * The run panel's files, configuration and links (runs/files.js). Run with: cd webapp && npm test -- runs/__tests__/files
 */
import { run_params, varying_params } from '../../viewers/series/comparison';
import { configuration_json, configuration_rows, file_url, integration_links, is_checkpoint, load_snippet, run_files, run_links } from '../files';

describe('checkpoints', () => {
  it('are saved models, whatever the framework, and what is in checkpoint folders', () => {
    for (const path of ['model.pt', 'checkpoints/epoch=3-step=900.ckpt', 'best.safetensors', 'export/detector.onnx', 'keras/model.h5', 'checkpoints/epoch-3/config.json', 'weights/backbone.npy', 'pytorch_model.bin', 'best_model.pkl'])
      expect([path, is_checkpoint(path)]).toEqual([path, true]);
    for (const path of ['log.txt', 'series.json', 'predictions.json', 'images/0001.png', 'data.bin', 'results.pkl', 'final/report.html'])
      expect([path, is_checkpoint(path)]).toEqual([path, false]);
  });

  it('come first, the listing itself is left out, sizes add up', () => {
    const files = run_files({
      'log.txt': { st_size: 300 }, 'checkpoints/last.pt': { st_size: 2000 }, 'checkpoints/best.pt': { st_size: 1000 },
      'manifest.outputs.json': { st_size: 50 }, 'series.json': { st_size: 10 },
    });
    expect(files.checkpoints.map(f => f.path)).toEqual(['checkpoints/best.pt', 'checkpoints/last.pt']);
    // the best, the last, the exports, then the latest first
    expect(run_files({ 'checkpoints/epoch-2.pt': {}, 'checkpoints/epoch-10.pt': {}, 'export/m.onnx': {}, 'checkpoints/last.pt': {}, 'best.ckpt': {} }).checkpoints.map(f => f.path))
      .toEqual(['best.ckpt', 'checkpoints/last.pt', 'export/m.onnx', 'checkpoints/epoch-10.pt', 'checkpoints/epoch-2.pt']);
    expect(files.others.map(f => f.path)).toEqual(['log.txt', 'series.json']);
    expect(files.count).toBe(4);
    expect(files.total_size).toBe(3310);
  });

  it('have URLs and code that load them', () => {
    expect(file_url('/s/runs/1', 'checkpoints/epoch=3 #1.pt')).toBe('/s/runs/1/checkpoints/epoch%3D3%20%231.pt');
    const code = load_snippet('https://qa/p/runs?run=1', 'checkpoints/best.pt');
    expect(code).toContain('qb.runs("https://qa/p/runs?run=1")[0]');
    expect(code).toContain('run.output_dir / "checkpoints/best.pt"');
    expect(code).toContain('torch.load');
    expect(load_snippet('u', 'model.onnx')).not.toContain('torch');
  });
});

describe('the configuration', () => {
  it('lists the configurations files, then the parameters, nested ones with dots, without what is not configuration', () => {
    const run = {
      configurations: ['configs/detector.yaml', { lr: 1e-3 }], data: { params: { optimizer: { name: 'adamw', betas: [0.9, 0.99] }, badges: [] } },
      params: { lr: 1e-3, optimizer: { name: 'adamw', betas: [0.9, 0.99] }, badges: [{ text: 'x' }], empty: {} },
    };
    expect(configuration_rows(run)).toEqual({
      texts: ['configs/detector.yaml'],
      params: [['lr', 1e-3, 'configs'], ['optimizer.name', 'adamw', 'logged'], ['optimizer.betas', [0.9, 0.99], 'logged'], ['empty', {}, 'configs']],
    });
    expect(JSON.parse(configuration_json(run))).toEqual({ lr: 1e-3, optimizer: { name: 'adamw', betas: [0.9, 0.99] }, empty: {} });
  });
});

describe('links', () => {
  it("are the run's badges with an href", () => {
    expect(run_links({ params: { badges: [{ text: 'Model card', href: 'https://hub/m' }, { text: 'no link' }, 'text'] } }))
      .toEqual([{ text: 'Model card', href: 'https://hub/m', icon: undefined }]);
  });

  it("and the project's integrations filled for the run's commit, without actions nor what needs a branch", () => {
    const integrations = [
      { text: 'Checks', href: '${git.web_url}/commit/${commit.id}/checks', icon: 'tick' },
      { text: 'On the branch', href: '/b/${branch}' },
      { text: 'Rebuild', webhook: 'https://ci/hook', href: 'https://ci' },
      { divider: true },
      { text: 'Docs', sub: [{ text: 'Model zoo', href: 'https://zoo/${commit.id.slice(0, 7)}' }, { text: 'Broken', href: '${nope}' }] },
    ];
    const links = integration_links({
      integrations, project: 'kitefin/detector', project_data: { data: { git: { web_url: 'https://git/kitefin/detector' } } },
      commit: { id: 'abcdef123456', branch: 'main' }, batch: { label: '@alice| lr-sweep' },
    });
    expect(links).toEqual([
      { text: 'Checks', href: 'https://git/kitefin/detector/commit/abcdef123456/checks', icon: 'tick' },
      { text: 'Model zoo', href: 'https://zoo/abcdef1', icon: undefined },
    ]);
  });
});

describe("the run's links", () => {
  it("never tell runs apart: they are not parameters, even when each run's differ", () => {
    const runs = [1, 2].map(id => ({ configurations: [{ lr: id }], data: { params: { badges: [{ text: 'Model card', href: `https://hub/${id}` }], device: 'cuda' } } }));
    expect(run_params(runs[0])).toEqual({ lr: 1, device: 'cuda' });
    expect(varying_params(runs)).toEqual(['lr']);
  });
});
