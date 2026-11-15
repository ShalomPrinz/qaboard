// The steps shown on a pipeline's card. Run with: cd webapp && npm test -- PipelineSteps
import { screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

import { renderWithProviders } from '../../test-utils';
import { PipelineSteps, defined_steps, duration_label, runs_summary } from '../PipelineSteps';

const pipeline = {
  version: 1,
  status: 'failed',
  steps: [
    { name: 'calibration-images', type: 'batch', status: 'success', batches: ['calibration-images'], label: 'my-label', runs: 12, failed_runs: 0, cached_runs: 12, duration: 75, command: 'qa batch calibration-images' },
    { name: 'calibrate', type: 'run', status: 'failed', returncode: 2, duration: 0.2, command: 'python calibrate.py /out/a /out/b', allow_failure: true },
    { name: 'evaluation', type: 'batch', status: 'running', batches: ['evaluation'], label: 'my-label' },
    { name: 'report', type: 'run', status: 'pending', commands: ['python report.py'] },
  ],
};
const output = { id: 1, output_type: 'pipeline', test_input_path: 'PIPELINE', output_dir_url: '/s/out/1', is_pending: false, is_running: false };

const render = (output_, files = { '/s/out/1/pipeline.json': pipeline }) => {
  vi.stubGlobal('fetch', vi.fn(async url => {
    const file = files[String(url).split('?')[0]];
    return file ? new Response(JSON.stringify(file), { headers: { 'Content-Type': 'application/json' } }) : new Response('not found', { status: 404 });
  }));
  return renderWithProviders(<MemoryRouter><PipelineSteps output={output_}/></MemoryRouter>);
};

afterEach(() => vi.unstubAllGlobals());


test('durations and summaries of runs', () => {
  expect(duration_label(0.2)).toBe('<1s');
  expect(duration_label(42.4)).toBe('42s');
  expect(duration_label(75)).toBe('1m 15s');
  expect(duration_label(3 * 3600 + 5 * 60)).toBe('3h 05m');
  expect(runs_summary({ type: 'batch', runs: 12, failed_runs: 2, cached_runs: 3 })).toBe('12 runs · 2 failed · 3 reused');
  expect(runs_summary({ type: 'batch', runs: 1, failed_runs: 0, cached_runs: 1 })).toBe('1 run · all reused');
  expect(runs_summary({ type: 'run' })).toBeNull();
  expect(runs_summary({ type: 'batch' })).toBeNull();
});

test('shows the steps in order, with their status', async () => {
  render({ ...output, is_running: true });
  await waitFor(() => expect(screen.getAllByTestId('pipeline-step')).toHaveLength(4));
  const steps = screen.getAllByTestId('pipeline-step');
  expect(steps.map(s => s.dataset.status)).toEqual(['success', 'failed', 'running', 'pending']);
  expect(steps[0]).toHaveTextContent('calibration-images');
  expect(steps[0]).toHaveTextContent('12 runs · all reused');
  expect(steps[0]).toHaveTextContent('1m 15s');
  expect(steps[1]).toHaveTextContent('calibrate');
});

test('a pipeline that is over has no step still running', async () => {
  // e.g. its job was killed
  render(output);
  await waitFor(() => expect(screen.getAllByTestId('pipeline-step')).toHaveLength(4));
  expect(screen.getAllByTestId('pipeline-step').map(s => s.dataset.status)).toEqual(['success', 'failed', 'failed', 'skipped']);
});

test('a step shows its command and details', async () => {
  render(output);
  await waitFor(() => expect(screen.getAllByTestId('pipeline-step')).toHaveLength(4));
  fireEvent.click(screen.getAllByTestId('pipeline-step')[1]);
  expect(await screen.findByText('python calibrate.py /out/a /out/b')).toBeInTheDocument();
  expect(screen.getByText('Allowed to fail')).toBeInTheDocument();
  expect(screen.getByText(/exited with the code 2/)).toBeInTheDocument();
});

test('a batch step links to its runs', async () => {
  render(output);
  await waitFor(() => expect(screen.getAllByTestId('pipeline-step')).toHaveLength(4));
  fireEvent.click(screen.getAllByTestId('pipeline-step')[0]);
  expect(await screen.findByRole('button', { name: /Show its 12 runs/ })).toBeInTheDocument();
});

test('the steps of the definition, in both syntaxes', () => {
  expect(defined_steps([
    'base', { gain: 2 },
    { batch: 'images', label: 'x' },
    { batch: ['a', { label: 'y' }, 'b', { 'allow-failure': true }] },
    { name: 'calibrate', run: 'python calibrate.py' },
    { run: ['cd /tmp', 'ls'] },
  ])).toEqual([
    { name: 'images', type: 'batch', status: 'unknown', batches: ['images'] },
    { name: 'a+b', type: 'batch', status: 'unknown', batches: ['a', 'b'], allow_failure: true },
    { name: 'calibrate', type: 'run', status: 'unknown', commands: ['python calibrate.py'] },
    { name: 'step-4', type: 'run', status: 'unknown', commands: ['cd /tmp', 'ls'] },
  ]);
  expect(defined_steps(undefined)).toEqual([]);
});

test('without pipeline.json, shows the steps of the definition', async () => {
  // pipelines from before the file, or done by the project's own code
  render({ ...output, configurations: [{ batch: 'images' }, { run: 'python report.py' }] }, {});
  await waitFor(() => expect(screen.getAllByTestId('pipeline-step')).toHaveLength(2));
  expect(screen.getAllByTestId('pipeline-step').map(s => s.dataset.status)).toEqual(['unknown', 'unknown']);
});

test('shows nothing without steps', async () => {
  const { container } = render({ ...output, configurations: ['base'] }, {});
  await waitFor(() => expect(fetch).toHaveBeenCalled());
  expect(container.querySelector('[data-testid="pipeline-steps"]')).toBeNull();
});
