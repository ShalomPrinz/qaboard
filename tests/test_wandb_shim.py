"""
Tests for qaboard/integrations/wandb.py: training scripts written for Weights & Biases, with their import changed to
`from qaboard.integrations import wandb`, make QA-Board runs.

Like tests/test_active_run.py, each test runs a script in a temporary project, against a small HTTP server that
records what it gets.
"""
import json
import os
import unittest
from pathlib import Path

from tests.test_active_run import ActiveRunTestCase

PLATFORM = 'linux' if os.name != 'nt' else 'windows'

TRAINING = """
from qaboard.integrations import wandb

wandb.init(project="detector", group="lr-sweep", config={"model": "resnet50", "lr": 0.001}, tags=["baseline"], job_type="train")
wandb.config.update({"batch_size": 32})          # before the first log: part of the run's identity
wandb.define_metric("val/loss", summary="min", goal="minimize")
for epoch in range(3):
  for step in range(10):
    wandb.log({"train/loss": 1 / (1 + 10 * epoch + step), "epoch": epoch}, commit=False)
    wandb.log({"lr": 0.001})
  wandb.log({"val": {"loss": [0.5, 0.2, 0.3][epoch]}})
wandb.config.epochs = 3                            # after: a parameter
wandb.summary["mAP"] = 0.41
print("training done")
wandb.finish()
"""


class WandbTestCase(ActiveRunTestCase):
  def script(self, code, name='train_wandb.py', **options):
    (self.project / name).write_text(code)
    return self.python(script=name, **options)

  def run_dir(self, batch, configs, input, root=None):
    from qaboard.conventions import run_dir_in_commit
    relative = run_dir_in_commit(batch, PLATFORM, configs, {}, input, Path('.'), {})
    if root is not None:
      return root / relative
    return self.storage / 'kitefin/detector' / self.commit[:2] / self.commit[2:16] / relative


class TestTraining(WandbTestCase):
  def test_a_wandb_training_is_a_run_like_start_runs(self):
    server = self.server()
    result = self.script(TRAINING, server=server)
    self.assertEqual(result.returncode, 0, result.stderr)
    started, *_, finished = server.outputs()
    configuration = {'model': 'resnet50', 'lr': 0.001, 'batch_size': 32}
    run_dir = self.run_dir('@alice| detector/lr-sweep', [configuration], 'train_wandb')

    # what start_run(input="train_wandb", configs=[...], batch="detector/lr-sweep") sends
    for body in (started, finished):
      self.assertEqual(body['project'], 'kitefin/detector')  # the repository's, not wandb's project
      self.assertEqual(body['commit_sha'], self.commit)
      self.assertEqual(body['batch_label'], '@alice| detector/lr-sweep')
      self.assertEqual(body['input_path'], 'train_wandb')  # no name: the script's
      self.assertEqual(body['configurations'], [configuration])
      self.assertEqual(body['output_directory'], str(run_dir))
    self.assertEqual(finished['data']['params'], {'epochs': 3, 'wandb': {'job_type': 'train', 'tags': ['baseline']}})
    self.assertEqual((finished['is_pending'], finished['is_failed']), (False, False))

    # each series' last value is a metric, val/loss its minimum, and the summary
    metrics = finished['metrics']
    self.assertEqual({k: metrics[k] for k in ('train/loss', 'epoch', 'lr', 'val/loss', 'mAP')},
                     {'train/loss': 1 / 30, 'epoch': 2, 'lr': 0.001, 'val/loss': 0.2, 'mAP': 0.41})
    self.assertEqual(finished['metrics_definitions']['val/loss'], {'smaller_is_better': True})

    # wandb's steps: commit=False adds to the step, each committed call is one
    from qaboard.series import read_series
    self.assertEqual([p.step for p in read_series(run_dir, 'val/loss')], [10, 21, 32])
    self.assertEqual([p.step for p in read_series(run_dir, 'train/loss')], [*range(0, 10), *range(11, 21), *range(22, 32)])
    self.assertEqual([p.step for p in read_series(run_dir, 'lr')], [p.step for p in read_series(run_dir, 'train/loss')])
    self.assertIn('training done', (run_dir / 'log.txt').read_text())
    self.assertNotIn('WARNING', result.stderr)

  def test_steps_like_wandb(self):
    result = self.script(QA_OFFLINE='1', code="""
from qaboard.integrations import wandb
from qaboard.series import read_series

run = wandb.init()
wandb.log({"a": 1})                       # step 0
wandb.log({"a": 2}, step=5)               # step 5, not committed yet
wandb.log({"b": 3})                       # also step 5, then commits
wandb.log({"a": 4}, commit=False)         # step 6
wandb.log({"a": 5}, commit=False)         # step 6: the last value wins
wandb.log({"b": 6}, step=7)               # commits step 6
wandb.log({"b": 7}, step=3)               # back in time: skipped, with a warning
wandb.log({"b": 8}, step=9, commit=True)  # step 9, committed: the next is 10
assert run.step == 10, run.step
wandb.log({"c": "a text", "d": None, "_private": 1, "e": 1.5})
wandb.log({"f": 1}, step=12)              # not committed: finish() logs it
directory = run.dir
wandb.finish()
points = lambda key: [(p.step, p.value) for p in read_series(directory, key)]
assert points("a") == [(0, 1), (5, 2), (6, 5)], points("a")
assert points("b") == [(5, 3), (7, 6), (9, 8)], points("b")
assert points("e") == [(10, 1.5)] and points("f") == [(12, 1)], (points("e"), points("f"))
assert wandb.run is None
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stderr.count('steps only go up'), 1)
    self.assertEqual(result.stderr.count("str values aren't kept"), 1)
    self.assertEqual(result.stderr.count("'_private'"), 1)

  def test_the_name_and_the_configuration_identify_the_run(self):
    result = self.script(QA_OFFLINE='1', code="""
from qaboard.integrations import wandb

for fold in range(2):
  with wandb.init(project="detector", group="cv", name=f"fold-{fold}", config={"lr": 0.1}) as run:
    wandb.log({"loss": fold})
    assert run.qaboard.input == f"fold-{fold}", run.qaboard
    assert run.qaboard.configs == ({"lr": 0.1},)
    assert run.name == f"fold-{fold}"

# no name: the script's. A name set before the first log() counts; after, it is only recorded
run = wandb.init(project="detector", config={"lr": 0.1})
assert run.name.startswith("run-")
run.name = "renamed"
wandb.log({"loss": 1})
run.name = "too-late"
assert run.qaboard.input == "renamed" and run.qaboard.batch == "detector", run.qaboard
assert run.qaboard.params["wandb"] == {"name": "too-late"}, run.qaboard.params
wandb.finish()

run = wandb.init(project="detector")
assert run.qaboard.input == "train_wandb" and run.qaboard.configs == (), run.qaboard
wandb.finish()

# start_run()'s arguments win, wandb's configuration is then a parameter
run = wandb.init(name="ignored", config={"lr": 0.1}, qaboard={"input": "train/coco-mini", "configs": ["resnet50"], "batch": "mine"})
assert (run.qaboard.input, run.qaboard.configs, run.qaboard.batch) == ("train/coco-mini", ("resnet50",), "mine"), run.qaboard
assert run.qaboard.params["lr"] == 0.1
wandb.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stderr.count('wandb.run.name'), 1)
    for fold in range(2):
      run_dir = self.run_dir(f'detector/cv', [{'lr': 0.1}], f'fold-{fold}', root=self.project)
      self.assertEqual(json.loads((run_dir / 'metrics.json').read_text())['loss'], fold)

  def test_config(self):
    result = self.script(QA_OFFLINE='1', code="""
import argparse
from qaboard.integrations import wandb

try:
  wandb.config.lr
except wandb.Error as e:
  assert "wandb.init()" in str(e)
try:
  wandb.log({"loss": 1})
except wandb.Error:
  pass
else:
  raise AssertionError("log before init")

args = argparse.Namespace(lr=0.1, layers=[64, 64])
run = wandb.init(config=args, config_exclude_keys=["layers"])
wandb.config.momentum = 0.9
wandb.config["optimizer"] = "sgd"
wandb.config.setdefaults({"lr": 1.0, "warmup": 5})
assert wandb.config.lr == 0.1 and wandb.config["warmup"] == 5 and "momentum" in wandb.config
assert getattr(wandb.config, "missing", None) is None
assert dict(wandb.config) == {"lr": 0.1, "momentum": 0.9, "optimizer": "sgd", "warmup": 5}, dict(wandb.config)
wandb.log({"loss": 1})
wandb.config.update({"lr": 0.05}, allow_val_change=True)
assert run.qaboard.configs == ({"lr": 0.1, "momentum": 0.9, "optimizer": "sgd", "warmup": 5},), run.qaboard.configs
assert run.qaboard.params["lr"] == 0.05 and wandb.config.lr == 0.05

wandb.summary["best"] = 0.5
wandb.summary.update({"val": {"top1": 0.7}, "image": wandb.Image(None), "verdict": "ship"})
assert wandb.summary["loss"] == 1 and wandb.summary.best == 0.5 and wandb.summary["val/top1"] == 0.7, dict(wandb.summary)
assert "image" not in wandb.summary and wandb.summary.get("verdict") == "ship"
wandb.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stderr.count("Image values aren't kept"), 1)


class TestModes(WandbTestCase):
  def test_disabled(self):
    server = self.server()
    result = self.script(server=server, code="""
from qaboard.integrations import wandb
run = wandb.init(project="detector", mode="disabled", config={"lr": 0.1})
assert wandb.config.lr == 0.1 and run.disabled and run.qaboard is None
for step in range(3):
  wandb.log({"loss": step})
wandb.summary["best"] = 1
assert wandb.summary["best"] == 1
(__import__("pathlib").Path(run.dir) / "notes.txt").write_text("somewhere")
wandb.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(server.requests, [])
    self.assertFalse(any(self.storage.iterdir()))
    self.assertFalse((self.project / 'output').exists())

  def test_disabled_from_the_environment(self):
    server = self.server()
    result = self.script("from qaboard.integrations import wandb\nwandb.init()\nwandb.log({'loss': 1})\nwandb.finish()", server=server, WANDB_MODE='disabled')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(server.requests, [])

  def test_offline(self):
    server = self.server()
    result = self.script(TRAINING.replace('job_type="train"', 'job_type="train", mode="offline"'), server=server)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(server.requests, [])
    # like a local `qa run`: in the project's output/, in the batch as named
    run_dir = self.run_dir('detector/lr-sweep', [{'model': 'resnet50', 'lr': 0.001, 'batch_size': 32}], 'train_wandb', root=self.project)
    self.assertEqual(json.loads((run_dir / 'metrics.json').read_text())['val/loss'], 0.2)
    self.assertFalse(any(self.storage.iterdir()))

  def test_resume_continues_the_steps(self):
    code = """
from qaboard.integrations import wandb
run = wandb.init(project="detector", config={"lr": 0.1}, resume="allow")
for _ in range(3):
  wandb.log({"loss": run.step})
wandb.finish()
"""
    self.assertEqual(self.script(code, QA_OFFLINE='1').returncode, 0)
    result = self.script(code, QA_OFFLINE='1')
    self.assertEqual(result.returncode, 0, result.stderr)
    from qaboard.series import read_series
    run_dir = self.run_dir('detector', [{'lr': 0.1}], 'train_wandb', root=self.project)
    self.assertEqual([(p.step, p.value) for p in read_series(run_dir, 'loss')], [(s, s) for s in range(6)])


class TestLifecycle(WandbTestCase):
  def test_an_exception_fails_the_run(self):
    server = self.server()
    result = self.script(server=server, code="""
from qaboard.integrations import wandb
with wandb.init(project="detector") as run:
  wandb.log({"loss": 1.0})
  raise RuntimeError("the loss diverged")
""")
    self.assertNotEqual(result.returncode, 0)
    finished = server.outputs()[-1]
    self.assertEqual((finished['is_failed'], finished['metrics']['loss']), (True, 1.0))

  def test_an_exit_code_fails_the_run(self):
    server = self.server()
    result = self.script(server=server, code="from qaboard.integrations import wandb\nwandb.init()\nwandb.finish(exit_code=1)\nwandb.finish()")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual([o['is_failed'] for o in server.outputs() if not o['is_pending']], [True])

  def test_what_is_logged_last_is_kept_without_finish(self):
    result = self.script(QA_OFFLINE='1', code="""
from qaboard.integrations import wandb
wandb.init(project="detector")
wandb.log({"loss": 1.0})
wandb.log({"loss": 0.5}, step=10)  # not committed when the process ends
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    run_dir = self.run_dir('detector', [], 'train_wandb', root=self.project)
    self.assertEqual(json.loads((run_dir / 'metrics.json').read_text())['loss'], 0.5)

  def test_init_again_returns_the_active_run_unless_reinit(self):
    result = self.script(QA_OFFLINE='1', code="""
from qaboard.integrations import wandb
first = wandb.init(project="detector", name="first")
assert wandb.init(project="detector", name="second") is first
second = wandb.init(project="detector", name="second", reinit=True)
assert second is not first and first._finished and wandb.run is second
wandb.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stderr.count('returns it, like wandb'), 1)


class TestFilesAndMedia(WandbTestCase):
  def test_files(self):
    (self.project / 'checkpoints').mkdir()
    (self.project / 'checkpoints' / 'epoch-1.pt').write_bytes(b'one')
    (self.project / 'model.onnx').write_bytes(b'model')
    (self.project / 'sample.png').write_bytes(b'png')
    result = self.script(QA_OFFLINE='1', code="""
import numpy
from pathlib import Path
from qaboard.integrations import wandb

class PILLike:  # what wandb.Image() takes from Pillow
  mode = "RGB"
  def save(self, f, format):
    f.write(b"from-pil")

run = wandb.init(project="detector")
wandb.save("checkpoints/*.pt")
Path("checkpoints/epoch-2.pt").write_bytes(b"two")   # saved when the run finishes, like wandb's policy="live"
(Path(run.dir) / "written-in-place.txt").write_text("ok")
wandb.save(str(Path(run.dir) / "written-in-place.txt"))
artifact = wandb.Artifact("predictions", type="dataset")
artifact.add_file("model.onnx", name="weights.onnx")
with artifact.new_file("readme.md") as f:
  f.write("hello")
wandb.log_artifact(artifact)
wandb.log_model("model.onnx", name="detector")
wandb.log({"examples": wandb.Image("sample.png"), "pil": wandb.Image(PILLike()), "html": wandb.Html("<p>hi</p>")})
wandb.log({"grid": [wandb.Image("sample.png"), wandb.Image("sample.png")]})
wandb.log({"array": wandb.Image(numpy.zeros((4, 4, 3)))})
wandb.log({"array": wandb.Image(numpy.zeros((4, 4, 3)))})
wandb.log({"missing": wandb.Image("missing.png")})
wandb.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    run_dir = self.run_dir('detector', [], 'train_wandb', root=self.project)
    files = {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
    self.assertEqual({k: v for k, v in files.items() if not k.endswith(('.json', '.jsonl', '.txt'))}, {
      'checkpoints/epoch-1.pt': b'one',
      'checkpoints/epoch-2.pt': b'two',
      'artifacts/predictions/weights.onnx': b'model',
      'artifacts/predictions/readme.md': b'hello',
      'artifacts/detector/model.onnx': b'model',
      'media/examples/0.png': b'png',
      'media/pil/0.png': b'from-pil',
      'media/html/0.html': b'<p>hi</p>',
      'media/grid/1_0.png': b'png',
      'media/grid/1_1.png': b'png',
    })
    self.assertEqual(files['written-in-place.txt'], b'ok')
    self.assertEqual(result.stderr.count("wandb.Image('missing.png'): no such file"), 1)
    self.assertNotIn("Image values aren't kept", result.stderr)
    try:
      import PIL  # noqa
    except ImportError:
      self.assertEqual(result.stderr.count("Pillow, which isn't installed"), 1)


class TestUnsupported(WandbTestCase):
  def test_unsupported_calls_warn_once_and_dont_raise(self):
    result = self.script(QA_OFFLINE='1', code="""
from qaboard.integrations import wandb

assert wandb.login(key="x") is True
run = wandb.init(project="detector", entity="team", sync_tensorboard=True, not_a_wandb_argument=1, settings=wandb.Settings(start_method="thread"))
for _ in range(2):
  wandb.watch(object(), log="all", log_freq=10)
  run.watch(object())
  wandb.define_metric("val/*", step_metric="epoch")
  table = wandb.Table(columns=["a", "b"])
  table.add_data(1, 2)
  wandb.log({"table": table, "hist": wandb.Histogram([1, 2, 3]), "curve": wandb.plot.line(table, "a", "b"), "values": [1, 2]})
  assert not wandb.use_artifact("dataset:latest").download()
  for r in wandb.Api().runs("team/detector"):
    raise AssertionError("no runs")
  run.log_code(".")
  wandb.alert(title="Loss", text="it diverged")
  wandb.Artifact("x").add_reference("s3://bucket/x")
wandb.unwatch()
wandb.mark_preempting()
wandb.define_metric("loss", summary="min,max")
wandb.log({"loss": 2})
wandb.log({"loss": 1})
assert wandb.summary["loss"] == 1
wandb.finish()
wandb.finish()
run.log({"loss": 0})
run.finish()
""")
    self.assertEqual(result.returncode, 0, result.stderr)
    for warning in ("wandb.watch() isn't supported", "sync_tensorboard=...", "not_a_wandb_argument=...", 'step_metric=...',
                    "Table values aren't kept", "Histogram values aren't kept", "wandb.plot isn't supported",
                    "list values aren't kept", "wandb.use_artifact() isn't supported", "wandb.Api isn't supported",
                    "wandb.log_code()", "QA-Board doesn't send alerts", "add_reference()", "one summary per series",
                    'wandb.log() after the run finished'):
      self.assertEqual(result.stderr.count(warning), 1, f"{warning!r} in:\n{result.stderr}")
    self.assertEqual(result.stderr.count('it diverged'), 2)  # alerts are printed each time
    self.assertNotIn('entity', result.stderr)
    self.assertNotIn('settings', result.stderr)

  def test_it_works_without_wandb(self):
    result = self.python(code="""
import sys
sys.modules["wandb"] = None  # import wandb fails
from qaboard.integrations import wandb
assert wandb.__name__ == "qaboard.integrations.wandb"
""")
    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
  unittest.main()
