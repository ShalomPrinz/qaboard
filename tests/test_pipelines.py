"""
Pipelines (qaboard/pipelines.py): their definition, what commands can use, and whole pipelines run with the CLI.
"""
import os
import sys
import json
import shutil
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest import mock

from qaboard import pipelines
from qaboard.pipelines import PipelineError, ShellList, StepResults, interpolate, parse_steps, validate


def names(configurations):
  return [s.name for s in parse_steps(configurations)]


class TestSteps(unittest.TestCase):
  def test_steps_are_the_configurations_with_run_or_batch(self):
    steps = parse_steps(["base", {"noise": 1}, {"batch": "a"}, {"run": "echo ok"}])
    self.assertEqual([(s.index, s.kind) for s in steps], [(1, 'batch'), (2, 'run')])

  def test_options_next_to_the_step(self):
    step, = parse_steps([{"batch": "a", "label": "my-label", "use-cache": True, "allow_failure": True, "batches-file": "x.yaml"}])
    self.assertEqual(step.batches, ["a"])
    self.assertEqual(step.options, {"label": "my-label", "use_cache": True, "allow_failure": True, "batches_file": ["x.yaml"]})
    self.assertTrue(step.allow_failure)

  def test_options_in_the_list_like_the_first_syntax(self):
    step, = parse_steps([{"batch": ["a", {"label": "my-label"}, "b", {"use-cache": True, "tuning": False}]}])
    self.assertEqual(step.batches, ["a", "b"])
    self.assertEqual(step.options, {"label": "my-label", "use_cache": True, "tuning": False})
    step, = parse_steps([{"run": ["cd /tmp", "ls", {"allow_failure": True}], "runner": "windows"}])
    self.assertEqual(step.commands, ["cd /tmp", "ls"])
    self.assertEqual(step.options, {"allow_failure": True, "runner": "windows"})

  def test_mistakes_are_explained(self):
    with self.assertRaisesRegex(PipelineError, "must be followed by a text or a list"):
      parse_steps([{"batch": {"name": "a"}}])
    with self.assertRaisesRegex(PipelineError, "needs the name of a batch"):
      parse_steps([{"batch": [{"label": "x"}]}])
    with self.assertRaisesRegex(PipelineError, "needs a command"):
      parse_steps([{"run": None}])
    with self.assertRaisesRegex(PipelineError, "same name"):
      parse_steps([{"batch": "a", "name": "x"}, {"run": "ls", "name": "x"}])
    with self.assertRaisesRegex(PipelineError, "can only have letters"):
      parse_steps([{"batch": "a", "name": "my step"}])

  def test_unknown_options(self):
    # a typo would change what the step does: an error
    with self.assertRaisesRegex(PipelineError, 'unknown option "use_cahce". Did you mean "use_cache"'):
      parse_steps([{"batch": "a", "use_cahce": True}])
    with self.assertRaisesRegex(PipelineError, 'Did you mean "label"'):
      parse_steps([{"batch": ["a", {"lable": "x"}]}])
    # others are ignored with a warning
    with mock.patch('click.secho') as secho:
      step, = parse_steps([{"batch": "a", "owner": "me"}])
    self.assertEqual(step.options, {})
    self.assertIn('unknown option "owner"', secho.call_args[0][0])
    with self.assertRaisesRegex(PipelineError, 'both "run:" and "batch:"'):
      parse_steps([{"batch": "a", "run": "x"}])

  def test_names(self):
    self.assertEqual(names([{"batch": "calibration"}]), ["calibration"])
    self.assertEqual(names([{"batch": "a", "name": "first"}]), ["first"])
    self.assertEqual(names([{"batch": ["a", "b"]}]), ["a+b"])
    # the same batch twice
    self.assertEqual(names([{"batch": "a"}, {"batch": "a"}, {"batch": "a"}]), ["a", "a-2", "a-3"])
    # explicit names are kept, whatever their position
    self.assertEqual(names([{"batch": "a"}, {"batch": "b", "name": "a"}]), ["a-2", "a"])

  def test_run_steps_are_named_after_their_script(self):
    for command, name in [
      ("python calibrate.py --fast", "calibrate"),
      ("python $REPO_DIR/scripts/my_script.py {output_dir}/batches.json", "my_script"),
      ("python3 -u ./ci/create_noise_matrix.py --config x", "create_noise_matrix"),
      ("./ci/report.sh", "report"),
      ("DEBUG=1 uv run tools/plot.py", "plot"),
      ("python -m mypackage.evaluate --all", "evaluate"),
      ("calibration.exe", "calibration"),
      ("my-tool --fast input.py", "my-tool"),
      ("echo Start of PIPELINE", "step-1"),
      ("non-existing-command", "non-existing-command"),
      ('python -c "print(1)"', "step-1"),
    ]:
      self.assertEqual(names([{"run": command}]), [name], command)
    # the first command that says something
    self.assertEqual(names([{"run": ["cd /tmp", "touch new_input.txt", "python script1.py /tmp/new_input.txt"]}]), ["script1"])
    self.assertEqual(names([{"run": "echo a"}, {"run": "python a.py"}, {"run": "python a.py"}, {"run": "ls"}]), ["step-1", "a", "a-2", "step-4"])

  def test_references_to_steps_are_checked_before_anything_runs(self):
    validate([{"batch": "a"}, {"run": "x.py {steps.a.output_dirs} {steps[a].runs_file} {not_a_step}"}])
    with self.assertRaisesRegex(PipelineError, 'no step is named "calib"'):
      validate([{"batch": "a"}, {"run": "x.py {steps.calib.output_dirs}"}])
    with self.assertRaisesRegex(PipelineError, 'runs after it'):
      validate([{"run": "x.py {steps.a.output_dirs}"}, {"batch": "a"}])
    with self.assertRaisesRegex(PipelineError, 'its own results'):
      validate([{"batch": "a", "database": "{steps.a.output_dir}"}])
    with self.assertRaisesRegex(PipelineError, 'no step is named "b"'):
      validate([{"batch": "a"}, {"batch": "c", "tuning": {"path": ["{steps.b.output_dir}/x"]}}])
    # what follows the name too
    with self.assertRaisesRegex(PipelineError, 'Did you mean "output_dirs"'):
      validate([{"batch": "a"}, {"run": "x.py {steps.a.outputs_dirs}"}])
    with self.assertRaisesRegex(PipelineError, '"calibrate" is a command'):
      validate([{"run": "calibrate.py"}, {"run": "x.py {steps.calibrate.output_dirs}"}])
    with self.assertRaisesRegex(PipelineError, 'needs what you want from these runs'):
      validate([{"batch": "a"}, {"run": "x.py {steps.a.ok}"}])
    with self.assertRaisesRegex(PipelineError, r'\{steps.a.ok.runs_file\}: "runs_file" is unknown here'):
      validate([{"batch": "a"}, {"run": "x.py {steps.a.ok.runs_file}"}])
    validate([{"batch": "a"}, {"run": "calibrate.py {steps.a.files[o.png]} {steps.a.ok.output_dirs:,}"}, {"run": "x {steps.calibrate.status}"}])

  def test_steps_of_a_batch(self):
    self.assertEqual(pipelines.steps_of({"inputs": []}), [])
    self.assertEqual(pipelines.steps_of({"steps": [{"batch": "a"}]}), [{"batch": "a"}])
    with self.assertRaisesRegex(PipelineError, "must be a list"):
      pipelines.steps_of({"steps": {"a": {"batch": "a"}}}, "my-pipeline")
    with self.assertRaisesRegex(PipelineError, 'Step 2 of "my-pipeline" must have'):
      pipelines.steps_of({"steps": [{"batch": "a"}, "python x.py"]}, "my-pipeline")


class TestInterpolation(unittest.TestCase):
  def test_fields_and_environment_variables(self):
    with mock.patch.dict(os.environ, {"MY_DIR": "/data"}):
      self.assertEqual(interpolate("ls $MY_DIR/{name} ${MY_DIR}", {"name": "x"}), "ls /data/x /data")

  def test_values_can_use_fields(self):
    self.assertEqual(interpolate("{a}", {"a": "{b}/x", "b": "y"}), "y/x")

  def test_nested_values(self):
    params = {"configurations": ["davis", {"noise": 2}], "params": {"noise": 2}}
    self.assertEqual(interpolate("--config {configurations[0]} --noise {params[noise]}", params), "--config davis --noise 2")

  def test_braces_that_are_not_fields_dont_stop_the_others(self):
    # it used to leave everything as is
    params = {"output_dir": "/out"}
    self.assertEqual(interpolate("""echo '{"a": 1}' > {output_dir}/x.json""", params), """echo '{"a": 1}' > /out/x.json""")
    self.assertEqual(interpolate("awk '{print $1}' {output_dir}/x {unknown}", params), "awk '{print $1}' /out/x {unknown}")
    self.assertEqual(interpolate("echo } {output_dir} {", params), "echo } /out {")

  def test_commands_that_worked_give_the_same_result(self):
    params = {"output_dir": "/out", "nb": 2}
    self.assertEqual(interpolate("x {{literal}} {output_dir} {nb:03d}", params), "x {literal} /out 002")

  def test_values_that_use_themselves(self):
    self.assertTrue(interpolate("{a}", {"a": "{a} x"}).startswith("{a} x"))

  def test_lists_of_paths(self):
    paths = ShellList(["/out/a", "/out/with space/b"])
    self.assertEqual(f"{paths}", "/out/a '/out/with space/b'")
    self.assertEqual(f"{paths:,}", "/out/a,'/out/with space/b'")
    self.assertEqual(str(paths), "/out/a '/out/with space/b'")

  def test_results_of_steps(self):
    runs = [{"output_dir": "/out/a", "input_path": "/db/a.jpg", "is_failed": False}, {"output_dir": "/out/b", "input_path": "/db/b.jpg", "is_failed": True}]
    steps = pipelines._Steps(later=["calib", "report", "later"])
    steps.add(StepResults("calib", "batch", runs, Path("/pipeline/steps/calib/runs.json"), label="my-label"))
    steps.add(StepResults("report", "run"))
    params = {"steps": steps}
    self.assertEqual(interpolate("{steps.calib.output_dirs}", params), "/out/a /out/b")
    self.assertEqual(interpolate("{steps[calib].output_dirs:,}", params), "/out/a,/out/b")
    self.assertEqual(interpolate("{steps.calib.files[sub/output.png]}", params), "/out/a/sub/output.png /out/b/sub/output.png")
    self.assertEqual(interpolate("{steps.calib.input_paths}", params), "/db/a.jpg /db/b.jpg")
    self.assertEqual(interpolate("{steps.calib.ok.output_dirs} | {steps.calib.failed.output_dirs}", params), "/out/a | /out/b")
    self.assertEqual(interpolate("{steps.calib.ok.output_dir}", params), "/out/a")
    self.assertEqual(interpolate("{steps.calib.runs_file} {steps.calib.label} {steps.report.status}", params), "/pipeline/steps/calib/runs.json my-label success")
    # with other braces around
    self.assertEqual(interpolate("echo '{}' {steps.calib.ok.output_dirs}", params), "echo '{}' /out/a")
    for text, error in [
      ("{steps.calib.output_dir}", "has 2 runs, not one"),
      ("{steps.calib.outputs}", 'steps have no "outputs"'),
      ("{steps.report.output_dirs}", "is a run step"),
      ("{steps.later.output_dirs}", "did not run yet"),
      ("{steps.typo.output_dirs}", 'no step is named "typo"'),
      ("echo '{}' {steps.typo.output_dirs}", 'no step is named "typo"'),
    ]:
      with self.assertRaisesRegex(PipelineError, error):
        interpolate(text, params)


class TestRun(unittest.TestCase):
  """Pipelines of commands, with a fake shell: what pipeline.json says when steps fail or are interrupted"""
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp(prefix='qa-pipeline-run-'))
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

  def run_pipeline(self, configurations, run_command):
    from qaboard.run import RunContext
    context = RunContext(type='pipeline', input_path=self.tmp / 'PIPELINE', database=self.tmp, platform='linux',
                         configurations=configurations, output_dir=self.tmp / 'pipeline')
    with mock.patch.dict(os.environ, {"QA_BATCHES_FILES": "[]"}):
      return pipelines.Pipeline(context, run_command=run_command).run(), context.output_dir

  def state(self, output_dir):
    return json.loads((output_dir / 'pipeline.json').read_text())

  def test_commands_get_the_step_and_the_pipeline_folder(self):
    seen = []
    metrics, output_dir = self.run_pipeline([{"run": "first"}, {"name": "second", "run": "x"}], lambda command, cwd, env: seen.append((command, env['PIPELINE_STEP'], env['PIPELINE_OUTPUT_DIR'])) or 0)
    self.assertEqual(seen, [('first', 'first', str(output_dir)), ('x', 'second', str(output_dir))])
    self.assertEqual(metrics, {"is_failed": False})
    self.assertEqual(self.state(output_dir)['status'], 'success')

  def test_an_error_fails_the_step_and_says_why(self):
    def run_command(command, cwd, env):
      if command == 'broken':
        raise OSError("No space left on device")
      return 0
    with mock.patch('click.secho'):
      metrics, output_dir = self.run_pipeline([{"run": "broken"}, {"run": "after"}], run_command)
    self.assertTrue(metrics['is_failed'])
    state = self.state(output_dir)
    self.assertEqual((state['status'], [s['status'] for s in state['steps']]), ('failed', ['failed', 'skipped']))
    self.assertEqual(state['steps'][0]['error'], 'OSError: No space left on device')

  def test_an_interrupted_pipeline_is_not_running_forever(self):
    def run_command(command, cwd, env):
      if command == 'long':
        raise KeyboardInterrupt
      return 0
    with self.assertRaises(KeyboardInterrupt):
      self.run_pipeline([{"run": "first"}, {"run": "long"}, {"run": "after"}], run_command)
    state = self.state(self.tmp / 'pipeline')
    self.assertEqual(state['status'], 'failed')
    self.assertEqual([s['status'] for s in state['steps']], ['success', 'failed', 'skipped'])
    self.assertEqual(state['steps'][1]['error'], 'Interrupted')
    self.assertIn('ended_at', state)


class TestEngine(unittest.TestCase):
  def test_engine(self):
    self.assertEqual(pipelines.engine({}, environ={}), 'builtin')
    self.assertEqual(pipelines.engine({"pipelines": {"engine": "entrypoint"}}, environ={}), 'entrypoint')
    self.assertEqual(pipelines.engine({"pipelines": {"engine": "entrypoint"}}, environ={"QABOARD_PIPELINES_ENGINE": "builtin"}), 'builtin')
    self.assertEqual(pipelines.engine({}, environ={"QABOARD_PIPELINES_ENGINE": "entrypoint"}), 'entrypoint')
    with mock.patch('click.secho') as secho:
      self.assertEqual(pipelines.engine({"pipelines": {"engine": "buildin"}}, environ={}), 'builtin')
    self.assertIn('unknown pipelines engine "buildin"', secho.call_args[0][0])

  def test_hw_alg_keeps_its_own_pipelines_until_it_opts_in(self):
    # its old branches keep running in CI, with their qaboard.yaml
    hw_alg = {"project": {"name": "CDE-Users/HW_ALG/CIS"}}
    self.assertEqual(pipelines.engine(hw_alg, environ={}), 'entrypoint')
    self.assertEqual(pipelines.engine({**hw_alg, "pipelines": {"engine": "builtin"}}, environ={}), 'builtin')
    self.assertEqual(pipelines.engine(hw_alg, environ={"QABOARD_PIPELINES_ENGINE": "builtin"}), 'builtin')
    with mock.patch('click.secho'):
      self.assertEqual(pipelines.engine({**hw_alg, "pipelines": {"engine": "typo"}}, environ={}), 'entrypoint')
    self.assertEqual(pipelines.engine({"project": {"name": "group/other"}}, environ={}), 'builtin')
    with mock.patch.dict(os.environ, {"QABOARD_PIPELINES_ENGINE": ""}):
      self.assertTrue(pipelines.is_builtin('pipeline', {}))
      self.assertFalse(pipelines.is_builtin('default', {}))
      self.assertFalse(pipelines.is_builtin('pipeline', {"pipelines": {"engine": "entrypoint"}}))


BATCHES = """
images:
  globs: ['*.jpg']
  inputs: [cli_tests]
  database: .

one-image:
  inputs: [cli_tests/a.jpg]
  database: .

two-batches:
  type: pipeline
  steps:
  - batch: [images, one-image]
    label: pipeline-two-batches
  - run: echo {steps.images+one-image.output_dirs} > {output_dir}/dirs.txt

matrix-configs:
  type: pipeline
  matrix:
    configs: [low, high]
  steps:
  - run: echo {configurations[0]} > {output_dir}/config.txt

wrong-type:
  type: images
  steps:
  - run: echo never

my-pipeline:
  type: pipeline
  steps:
  - run: echo "start" > {output_dir}/start.txt
  - batch: images
    label: pipeline-images
  - name: report
    run: >-
      python -c "import sys, json; json.dump(sys.argv[1:], open(sys.argv[1], 'w'))"
      {output_dir}/report.json {steps.images.runs_file} {steps.images.output_dirs}
  - batch: one-image
    label: pipeline-one-image
    configs: ["{output_dir}/start.txt", {threshold: 3}]
  - run: >-
      echo '{"accuracy": 0.9}' > $PIPELINE_OUTPUT_DIR/metrics.json

first-syntax:
  type: pipeline
  configurations:
  - param: 5
  - batch:
    - images
    - label: first-syntax-images
  - run: echo "{param} $PIPELINE_OUTPUT_DIR" > {output_dir}/echo.txt
  - run:
    - exit 3
    - allow_failure: true
  - run: echo after > {output_dir}/after.txt

failing:
  type: pipeline
  steps:
  - run: exit 1
  - run: echo after > {output_dir}/after.txt

matrix-pipeline:
  type: pipeline
  matrix:
    value: [1, 2]
  steps:
  - run: echo ${matrix.value} > {output_dir}/value.txt

cached:
  steps:
  - batch: images
    label: cached-images
    use-cache: true

with-inputs:
  type: pipeline
  database: .
  inputs:
  - cli_tests/a.jpg
  - cli_tests/b.jpg
  steps:
  - run: echo {rel_input_path} > {output_dir}/input.txt

typo-batch:
  type: pipeline
  steps:
  - batch: imagez
  - run: echo after > {output_dir}/after.txt

parameter-typo:
  type: pipeline
  configs:
  - threshold: 3
  steps:
  - run: >-
      echo {treshold} '{"a": 1}' ${HOME} > {output_dir}/out.txt

bad-reference:
  type: pipeline
  steps:
  - run: echo {steps.missing.output_dirs}
"""


class TestPipelinesCli(unittest.TestCase):
  """Whole pipelines in the sample project, run with the CLI"""
  @classmethod
  def setUpClass(cls):
    cls.project_dir = Path(__file__).resolve().parent.parent / 'qaboard/sample_project'
    cls.tmp = Path(tempfile.mkdtemp(prefix='qa-pipelines-'))
    cls.batches_file = cls.tmp / 'batches.yaml'
    cls.batches_file.write_text(BATCHES)
    cls.created = set()
    cls.clean()

  @classmethod
  def clean(cls):
    # local runs are saved in the project, under output/: other tests use it too, we remove what is ours
    for label in ('pipeline-images', 'pipeline-one-image', 'pipeline-two-batches', 'first-syntax-images', 'cached-images', 'entrypoint-engine'):
      shutil.rmtree(cls.project_dir / 'output' / label, ignore_errors=True)
    for output_dir in cls.created:
      shutil.rmtree(output_dir.parent, ignore_errors=True)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.tmp, ignore_errors=True)
    cls.clean()

  def qa(self, *args, env=None):
    environ = {
      **os.environ,
      "QA_TESTING": "true",
      "QA_OFFLINE": "true",
      "QA_DATABASE": ".",
      "QA_STORAGE": str(self.tmp / 'storage'),
      "QABOARD_IGNORE_SITE": "1",
      "QA_BATCH_QUIET": "1",
      "PYTHONPATH": str(self.project_dir.parent.parent),
      **(env or {}),
    }
    if not env or 'QABOARD_PIPELINES_ENGINE' not in env:
      environ.pop('QABOARD_PIPELINES_ENGINE', None)
    return subprocess.run([sys.executable, '-m', 'qaboard', *args], cwd=str(self.project_dir), env=environ,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

  def batch(self, *args, env=None):
    return self.qa('batch', '--batches-file', str(self.batches_file), *args, env=env)

  def pipeline_dirs(self, name, env=None):
    result = self.batch(name, '--list', env=env)
    self.assertEqual(result.returncode, 0, result.stderr)
    output_dirs = [Path(run['output_dir']) for run in json.loads(result.stdout)]
    self.created.update(output_dirs)
    return output_dirs

  def run_pipeline(self, name, expect_success=True, env=None):
    result = self.batch(name, env=env)
    self.assertEqual(result.returncode == 0, expect_success, result.stdout + result.stderr)
    return self.pipeline_dirs(name, env=env), result

  def test_list(self):
    result = self.batch('my-pipeline', '--list')
    self.assertEqual(result.returncode, 0, result.stderr)
    run, = json.loads(result.stdout)
    self.assertEqual(run['type'], 'pipeline')
    self.assertEqual(Path(run['input_path']).name, 'PIPELINE')
    self.assertEqual([c for c in run['configurations'] if 'batch' in c][0], {"batch": "images", "label": "pipeline-images"})

  def test_pipeline(self):
    (output_dir,), result = self.run_pipeline('my-pipeline')
    self.assertEqual((output_dir / 'start.txt').read_text().strip(), 'start')

    state = json.loads((output_dir / 'pipeline.json').read_text())
    self.assertEqual(state['status'], 'success')
    self.assertEqual([s['name'] for s in state['steps']], ['step-1', 'images', 'report', 'one-image', 'step-5'])
    self.assertEqual({s['status'] for s in state['steps']}, {'success'})
    images = state['steps'][1]
    self.assertEqual((images['type'], images['label'], images['runs'], images['failed_runs'], images['cached_runs']), ('batch', 'pipeline-images', 3, 0, 0))
    self.assertEqual(images['runs_file'], 'steps/images/runs.json')

    runs = json.loads((output_dir / 'steps/images/runs.json').read_text())
    self.assertEqual(sorted(r['input'] for r in runs), ['cli_tests/a.jpg', 'cli_tests/b.jpg', 'cli_tests/dir/c.jpg'])
    for run in runs:
      self.assertEqual((run['batch'], run['is_failed'], run['metrics']['is_failed']), ('images', False, False))
      self.assertTrue(Path(run['input_path']).is_absolute())
      self.assertTrue((Path(run['output_dir']) / 'metrics.json').exists())

    # the command got the runs file and the output directories
    report = json.loads((output_dir / 'report.json').read_text())
    self.assertEqual(report[1], str(output_dir / 'steps/images/runs.json'))
    self.assertEqual(sorted(report[2:]), sorted(r['output_dir'] for r in runs))

    # the first format, for the scripts that read it
    batches = json.loads((output_dir / 'batches.json').read_text())
    self.assertEqual(sorted(batches), ['images', 'one-image'])
    self.assertEqual(sorted(r['output_dir'] for r in batches['images']), sorted(r['output_dir'] for r in runs))
    self.assertNotIn('job_options', batches['images'][0])

    # configs are added after the batch's
    one_image, = json.loads((output_dir / 'steps/one-image/runs.json').read_text())
    self.assertEqual(one_image['extra_parameters'], {"_configs": [str(output_dir / 'start.txt'), {"threshold": 3}]})

    # metrics.json in the pipeline's directory: the pipeline's metrics
    metrics = json.loads((output_dir / 'metrics.json').read_text())
    self.assertEqual((metrics['accuracy'], metrics['is_failed']), (0.9, False))
    self.assertFalse([k for k in metrics if k.startswith('step ')])

  def test_first_syntax(self):
    (output_dir,), result = self.run_pipeline('first-syntax', expect_success=False)
    self.assertEqual((output_dir / 'echo.txt').read_text().strip(), f'5 {output_dir}')
    # allow_failure: the next steps run, and the pipeline is failed
    self.assertEqual((output_dir / 'after.txt').read_text().strip(), 'after')
    state = json.loads((output_dir / 'pipeline.json').read_text())
    self.assertEqual([s['status'] for s in state['steps']], ['success', 'success', 'failed', 'success'])
    self.assertEqual((state['status'], state['steps'][2]['returncode'], state['steps'][2]['allow_failure']), ('failed', 3, True))
    self.assertEqual(state['steps'][0]['label'], 'first-syntax-images')
    metrics = json.loads((output_dir / 'metrics.json').read_text())
    self.assertTrue(metrics['is_failed'])
    self.assertEqual(sorted(json.loads((output_dir / 'batches.json').read_text())), ['images'])

  def test_failure_stops_the_pipeline(self):
    (output_dir,), result = self.run_pipeline('failing', expect_success=False)
    self.assertFalse((output_dir / 'after.txt').exists())
    state = json.loads((output_dir / 'pipeline.json').read_text())
    self.assertEqual((state['status'], [s['status'] for s in state['steps']]), ('failed', ['failed', 'skipped']))

  def test_matrix(self):
    output_dirs, result = self.run_pipeline('matrix-pipeline')
    self.assertEqual(sorted((d / 'value.txt').read_text().strip() for d in output_dirs), ['1', '2'])

  def test_pipelines_with_inputs_run_once_per_input(self):
    output_dirs, result = self.run_pipeline('with-inputs')
    self.assertEqual(sorted((d / 'input.txt').read_text().strip() for d in output_dirs), ['cli_tests/a.jpg', 'cli_tests/b.jpg'])

  def test_a_step_with_several_batches(self):
    (output_dir,), result = self.run_pipeline('two-batches')
    batches = json.loads((output_dir / 'batches.json').read_text())
    # the key of the first implementation, and one per batch
    self.assertEqual(sorted(batches), ['--batch "images" --batch "one-image"', 'images', 'one-image'])
    self.assertEqual((len(batches['images']), len(batches['one-image'])), (3, 1))
    self.assertEqual(len((output_dir / 'dirs.txt').read_text().split()), 4)

  def test_matrix_of_configurations(self):
    # the matrix's configurations replace the default ones, not the steps
    output_dirs, result = self.run_pipeline('matrix-configs')
    self.assertEqual(sorted((d / 'config.txt').read_text().strip() for d in output_dirs), ['high', 'low'])

  def test_a_batch_without_runs_fails(self):
    # often a typo in the batch's name
    (output_dir,), result = self.run_pipeline('typo-batch', expect_success=False)
    self.assertIn('imagez has no runs. Is the name of the batch right?', result.stdout + result.stderr)
    self.assertFalse((output_dir / 'after.txt').exists())

  def test_misspelled_parameters_are_left_with_a_warning(self):
    (output_dir,), result = self.run_pipeline('parameter-typo')
    self.assertIn('{treshold} is not a parameter or field', result.stdout + result.stderr)
    self.assertNotIn('{a', result.stderr.split('{treshold}')[1].split('\n')[0])
    self.assertEqual((output_dir / 'out.txt').read_text().split()[0], '{treshold}')

  def test_the_end_of_the_log_says_what_happened(self):
    (output_dir,), result = self.run_pipeline('failing', expect_success=False)
    out = result.stdout + result.stderr
    self.assertIn('Pipeline FAILED', out)
    self.assertIn('To run the failed command again by hand:', out)
    self.assertIn(f'PIPELINE_OUTPUT_DIR={output_dir} exit 1', out)

  def test_steps_need_the_pipeline_type(self):
    result = self.batch('wrong-type', '--list')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('must have type: pipeline', result.stdout + result.stderr)

  def test_cache(self):
    # `steps:` is enough to say it's a pipeline
    (output_dir,), result = self.run_pipeline('cached')
    step, = json.loads((output_dir / 'pipeline.json').read_text())['steps']
    self.assertEqual((step['runs'], step['cached_runs']), (3, 0))
    (output_dir,), result = self.run_pipeline('cached')
    step, = json.loads((output_dir / 'pipeline.json').read_text())['steps']
    self.assertEqual((step['runs'], step['cached_runs'], step['status']), (3, 3, 'success'))
    self.assertIn('--action-on-existing=skip', step['command'])
    runs = json.loads((output_dir / 'steps/images/runs.json').read_text())
    self.assertEqual([r['is_cached'] for r in runs], [True, True, True])

  def test_mistakes_are_said_before_anything_runs(self):
    result = self.batch('bad-reference')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('no step is named "missing"', result.stdout + result.stderr)

  def test_tuning(self):
    result = self.qa('--tuning', '{"gain": 2}', 'batch', '--batches-file', str(self.batches_file), 'first-syntax')
    self.assertIn('exit 3', result.stdout + result.stderr) # its step allowed to fail: the pipeline ran until the end
    listed = self.qa('--tuning', '{"gain": 2}', 'batch', '--batches-file', str(self.batches_file), 'first-syntax', '--list')
    output_dir = Path(json.loads(listed.stdout)[0]['output_dir'])
    self.created.add(output_dir)
    self.assertEqual(json.loads((output_dir / 'tuning.json').read_text()), {"gain": 2})
    # batch steps get the tuning
    runs = json.loads((output_dir / 'steps/images/runs.json').read_text())
    self.assertEqual([r['extra_parameters'] for r in runs], [{"gain": 2}] * 3)

  def test_projects_can_keep_doing_pipelines_in_their_entrypoint(self):
    # the sample project's run() does nothing with the steps
    (output_dir,), result = self.run_pipeline('my-pipeline', env={"QABOARD_PIPELINES_ENGINE": "entrypoint", "QA_LABEL": "entrypoint-engine"})
    self.assertFalse((output_dir / 'pipeline.json').exists())


if __name__ == '__main__':
  unittest.main()
