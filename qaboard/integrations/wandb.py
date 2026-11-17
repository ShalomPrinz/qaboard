"""
A Weights & Biases-compatible front for `qaboard.start_run()`: try QA-Board with a training script written for
wandb by changing its import, and keep the rest.

    from qaboard.integrations import wandb       # instead of: import wandb

    wandb.init(project="detector", group="lr-sweep", config={"model": "resnet50", "lr": 1e-3})
    for step, batch in enumerate(loader):
      wandb.log({"train/loss": train_step(batch)})
    wandb.summary["mAP"] = evaluate(model)
    wandb.finish()

Experimental, to evaluate QA-Board with the scripts you have. It is a thin mapping onto `start_run()` and `ActiveRun`
(qaboard/active_run.py); user docs: website/docs/logging-from-code.mdx, "Coming from Weights & Biases". Nothing here
imports wandb, which doesn't need to be installed. What QA-Board doesn't do warns once and does nothing: an
unsupported call never ends a training. Only `init()` raises, when QA-Board can't make the run (no project, a wrong
`qaboard={...}`), and what would give files that don't exist (`use_artifact().download()`). When the run can't start
later, it says why once, and the training goes on without it.

A QA-Board run is identified by its project, commit, batch, input and configurations. A wandb run becomes one so that
the next commit's run of the same training is compared with it, and so that two runs of a script don't replace each
other's files:
- `project` and `group` make the batch, "project/group", where the web app lists the run. QA-Board's project is the
  repository, from qaboard.yaml: wandb's `entity` isn't used.
- The input is the run's `name` when you give one, else the script's name and the `job_type`, "train/eval". wandb's
  random names are not used. Ranks of a distributed training other than 0 add "rank-N" (WORLD_SIZE, RANK).
- The configuration is `config`'s plain data: objects, like argparse's `func`, are only recorded as parameters, as
  are the keys in `init(qaboard={"params": [...]})`.
- So stable names, groups and configurations compare across commits. Those with a date, a time, a job id or a
  generated id make each run new, with nothing to compare with: they are warned about. The run's identity is printed
  when it starts. Running the same run again on the same commit is a new attempt of it, which replaces its files:
  within a process, that is warned about too.
- The run starts in QA-Board at the first call that needs it (`log()`, `summary`, files, `run.dir`, `run.url`,
  `finish()`), with the configuration and name set until then: `wandb.config.update(args)` right after `init()` is
  part of the run's identity. What the configuration gets after that is recorded as parameters (`log_params()`).
  A run that logs nothing and isn't finished explicitly is not created.
- The name, `id`, `job_type`, `tags` and `notes` you give are recorded as the parameters `wandb.*`.
- `init(qaboard={...})` passes arguments to `start_run()`, and they win: `qaboard={"input": "train/coco-mini"}`.
- A process forked from the run's (a data loader's, a pool's worker) doesn't log to it, start it or finish it: it
  warns once. `wandb.init()` there starts a run of its own.

How the calls map:
- `log(data, step=, commit=)` -> `run.log()`, with wandb's steps: without `step` each call is a step, `commit=False`
  adds to the step until a call commits it, a step that goes back is skipped (warned once, counted at the end), and
  a key logged twice in a step keeps its last value. Numbers make series. `Image`s (also of matplotlib figures, or
  figures logged as they are) and media made from a file are saved as media/<key>/<step>.<extension>, lists of them
  as media/<key>/<index>/<step>.<extension>; other media, texts and lists are skipped with a warning.
- `config` -> the configuration, then `log_params()`. `summary[key] = value` -> `log_metrics()`; reading `summary`
  gives the run's metrics, where each series' last value is one, like wandb's summary.
- `define_metric(name, summary=, goal=)` -> `run.set_summary()` and which way is better (`qaboard.define_metric()`).
- `save()`, `log_artifact()`, `log_model()` and `Artifact` -> files of the run, under their own name or artifacts/<name>/.
- `finish(exit_code=)` -> `finish()`, or `fail()` for an exit code other than 0; also leaving `with wandb.init():`.
- `mode="disabled"` logs nothing, `mode="offline"` -> `share=False`, `resume="allow"|"must"|True` -> `resume=True`.
  The WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_NAME, WANDB_MODE... environment variables are read like wandb does.
- `login()` does nothing: QA-Board uses `qa auth login`. `alert()` prints. `watch()`, `use_artifact()`, tables,
  plots, sweeps and the rest of wandb's API warn once and do nothing.
- `run.qaboard` is the `ActiveRun`, for what only QA-Board does.
"""
import atexit
import inspect
import io
import json
import math
import os
import random
import re
import string
import sys
import threading
import time
from contextlib import contextmanager
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

from ..active_run import MAX_NAME_LENGTH, RESERVED_METRICS, SUMMARIES, ActiveRun, _default_input, _jsonable, start_run
from ..series import as_number, flatten

__all__ = [
  'init', 'log', 'finish', 'define_metric', 'save', 'log_artifact', 'log_model', 'use_artifact', 'watch', 'unwatch',
  'alert', 'login', 'mark_preempting', 'log_code', 'setup', 'teardown', 'run', 'config', 'summary', 'util', 'Run',
  'Config', 'Summary', 'Artifact', 'Image', 'Video', 'Audio', 'Html', 'Table', 'Histogram', 'Object3D', 'Molecule',
  'Plotly', 'Settings', 'AlertLevel', 'Error',
]
# Not wandb's: scripts that compare versions take their oldest code path
__version__ = '0.0.0+qaboard'

ONLINE_MODES = (None, '', 'online', 'run', 'shared')
OFFLINE_MODES = ('offline', 'dryrun')
RESUME_VALUES = (True, 'allow', 'must', 'auto')
TRUE_VALUES = ('1', 'true', 'yes', 'on')
PATTERN = re.compile(r'[*?\[]')   # in define_metric()'s names
# What changes at every run: dates, times, long numbers like job ids, objects' addresses
VOLATILE = re.compile(r'(19|20)\d\d[-_./]?[01]\d[-_./]?[0-3]\d|\b[0-2]?\d[:h][0-5]\d|\d{6,}| at 0x[0-9a-f]+')
JOB_IDS = ('SLURM_JOB_ID', 'LSB_JOBID', 'PBS_JOBID', 'JOB_ID', 'CI_JOB_ID', 'GITHUB_RUN_ID')
# What QA-Board, Python or numpy ask objects, which the stubs below must not pretend to have
PROTOCOL = frozenset(('tolist', 'item', 'numpy', 'detach', 'cpu', 'shape', 'ndim', 'dtype', 'save', 'savefig', 'mode',
                      'read', 'keys', 'items', 'values', 'flag_values_dict', 'to_dict', 'to_json', 'as_posix'))

_lock = threading.RLock()
_warned: Set[str] = set()
_unfinished: List['Run'] = []    # started in QA-Board, not finished: what is still buffered is saved at exit
_exit_hook = False


def _say(message: str) -> None:
  print(f"qaboard: {message}", file=sys.stderr)


def _warn_once(kind: str, message: str) -> None:
  with _lock:
    if kind in _warned:
      return
    _warned.add(kind)
  _say(message)


class Error(Exception):
  """`wandb.Error`: what wandb raises for calls in the wrong order, like `wandb.log()` before `wandb.init()`"""


class _Unsupported:
  """
  A wandb feature QA-Board doesn't have: calling it, or anything it gives, does nothing and returns it. What would
  give files (`download()`, `get_path()`, a path) raises an Error that says so, rather than giving a path to nothing.
  """
  FILES = ('download', 'get_path', 'file', 'get_entry', 'checkout')

  def __init__(self, name: str):
    self._name = name

  def __call__(self, *args: Any, **kwargs: Any) -> '_Unsupported':
    if self._name.rsplit('.', 1)[-1] in self.FILES:
      raise Error(f"{self._name}() isn't supported by QA-Board: there are no files to give. Read them from where they are.")
    return self

  def __getattr__(self, name: str) -> '_Unsupported':
    if name.startswith('_') or name in PROTOCOL:
      raise AttributeError(name)
    return _Unsupported(f"{self._name}.{name}")

  def __fspath__(self) -> str:
    raise Error(f"{self._name} isn't supported by QA-Board: it is no file or folder")

  def __bool__(self) -> bool:
    return False

  def __iter__(self) -> Iterator[Any]:
    return iter(())

  def __len__(self) -> int:
    return 0

  def __enter__(self) -> '_Unsupported':
    return self

  def __exit__(self, *args: Any) -> None:
    return None

  def __repr__(self) -> str:
    return f"<{self._name}: not supported by QA-Board>"


class Settings:
  """`wandb.Settings(...)`: accepted, and ignored"""
  def __init__(self, *args: Any, **kwargs: Any):
    self.__dict__.update(kwargs)


class AlertLevel:
  """`wandb.AlertLevel`, for `alert()`"""
  INFO = 'INFO'
  WARN = 'WARN'
  ERROR = 'ERROR'


class _Util:
  """`wandb.util`: `generate_id()`, the rest warns once and does nothing"""
  generated: Set[str] = set()  # random: names made with them change at every run

  @classmethod
  def generate_id(cls, length: int = 8) -> str:
    generated = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    cls.generated.add(generated)
    return generated

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
      raise AttributeError(name)
    return _unsupported(f"wandb.util.{name}")


util = _Util()


def _unsupported(name: str, why: str = '') -> _Unsupported:
  _warn_once(f"unsupported:{name}", f"WARNING: {name} isn't supported by QA-Board: ignored{why}.")
  return _Unsupported(name)


def _ignore_arguments(call: str, kwargs: Mapping[str, Any]) -> None:
  for key in kwargs:
    _warn_once(f"argument:{call}:{key}", f"WARNING: {call}({key}=...) isn't supported by QA-Board: ignored.")


def _as_mapping(values: Any, call: str) -> Dict[str, Any]:
  """A configuration as wandb takes it: a dict, an argparse.Namespace, absl flags, or a YAML file's path"""
  if values is None:
    return {}
  if isinstance(values, Mapping):
    return dict(values)
  if isinstance(values, (str, os.PathLike)):
    import yaml
    try:
      with open(values, encoding='utf-8') as f:
        loaded = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
      _warn_once(f"config-file:{values}", f"WARNING: {call}: could not read the configuration {values}: {e}")
      return {}
    if not isinstance(loaded, Mapping):
      _warn_once(f"config-file:{values}", f"WARNING: {call}: {values} is not a mapping: ignored.")
      return {}
    # wandb's config-defaults.yaml: {"lr": {"desc": "...", "value": 0.001}}
    return {str(k): v['value'] if isinstance(v, Mapping) and 'value' in v and set(v) <= {'value', 'desc'} else v for k, v in loaded.items()}
  if hasattr(values, 'flag_values_dict'):  # absl flags
    return dict(values.flag_values_dict())
  if hasattr(values, '__dict__'):  # argparse.Namespace, simple objects
    return {k: v for k, v in vars(values).items() if not k.startswith('_')}
  _warn_once(f"config-type:{type(values).__name__}", f"WARNING: {call}: a configuration is a dict or an argparse.Namespace, not {type(values).__name__}: ignored.")
  return {}


def _as_step(step: Any) -> Optional[int]:
  if isinstance(step, int) and not isinstance(step, bool):
    return step
  try:
    number = as_number(step) if not isinstance(step, bool) else math.nan
  except TypeError:
    number = math.nan
  if math.isfinite(number) and number == int(number):
    return int(number)
  _warn_once('step-type', f"WARNING: wandb.log(step={step!r}): steps are integers. The step is ignored.")
  return None


def _input_of(name: Optional[str], job_type: Optional[str]) -> str:
  """
  The run's input in QA-Board: its name, as a relative path without '..', else the script's name and the job type,
  "train/eval". Ranks of a distributed training other than 0 add "rank-N": else they would be the same run.
  """
  parts = [p for p in str(name or '').replace('\\', '/').split('/') if p.strip() and p not in ('.', '..')]
  if not parts:
    parts = [_default_input(), *([str(job_type).replace('/', '-')] if job_type else [])]
  rank = _rank()
  if rank:
    parts.append(f"rank-{rank}")
  return '/'.join(parts)


def _rank() -> int:
  """This process' rank in a distributed training (torchrun, Lightning, Accelerate, DeepSpeed, MPI...), else 0"""
  try:
    if int(os.environ.get('WORLD_SIZE') or os.environ.get('OMPI_COMM_WORLD_SIZE') or 1) <= 1:
      return 0
    return int(os.environ.get('RANK') or os.environ.get('OMPI_COMM_WORLD_RANK') or os.environ.get('LOCAL_RANK') or 0)
  except ValueError:
    return 0


def _volatile(value: Any) -> bool:
  """Whether a value looks like it changes at every run: a date, a time, a job id..."""
  if not isinstance(value, (str, int, os.PathLike)) or isinstance(value, bool):
    return False
  job_ids = [j for j in (os.environ.get(v) for v in JOB_IDS) if j and len(j) >= 3]
  if isinstance(value, int):  # a number of steps is no date
    return str(value) in job_ids
  text = str(value)
  return bool(VOLATILE.search(text)) or any(j in text for j in job_ids) or any(g in text for g in _Util.generated)


def _identifies(value: Any) -> bool:
  """
  Whether a configuration's value can identify a run: plain data. Objects (functions, devices, modules) would only
  be saved as their text, often with their address in memory, different at every run.
  """
  if value is None or isinstance(value, (bool, int, float, Path)):
    return True
  if isinstance(value, str):
    return ' at 0x' not in value
  if isinstance(value, (list, tuple)):
    return all(_identifies(v) for v in value)
  if isinstance(value, Mapping):
    return all(isinstance(k, str) and _identifies(v) for k, v in value.items())
  return type(value).__module__ == 'numpy' and hasattr(value, 'item') and getattr(value, 'ndim', 1) == 0


# --- configuration and summary ----------------------------------------------------------------------------------------
class Config:
  """
  `wandb.config` and `run.config`: the run's configuration. Read it like a dict or with attributes, `wandb.config.lr`.
  What it has when the run starts in QA-Board identifies the run; what you set after is recorded as parameters.
  """
  def __init__(self, run: Optional['Run'] = None, values: Optional[Mapping[str, Any]] = None):
    object.__setattr__(self, '_run', run)
    object.__setattr__(self, '_values', dict(values or {}))

  def _bound(self, call: str) -> 'Run':
    if self._run is None:
      raise Error(f"call wandb.init() before wandb.config{call}")
    return self._run

  def update(self, values: Any, allow_val_change: Optional[bool] = None) -> None:
    """Sets several values: a dict, an argparse.Namespace, absl flags or a YAML file's path"""
    run = self._bound('.update()')
    run._set_config(_as_mapping(values, 'wandb.config.update()'))

  def setdefaults(self, values: Any) -> None:
    """Sets the values not set yet"""
    run = self._bound('.setdefaults()')
    run._set_config({k: v for k, v in _as_mapping(values, 'wandb.config.setdefaults()').items() if k not in self._values})

  def setdefault(self, key: str, default: Any = None) -> Any:
    if key not in self._bound('.setdefault()')._config_values():
      self.update({key: default})
    return self._values[key]

  def as_dict(self) -> Dict[str, Any]:
    self._bound('.as_dict()')
    return dict(self._values)

  def keys(self) -> Any:
    return self.as_dict().keys()

  def values(self) -> Any:
    return self.as_dict().values()

  def items(self) -> Any:
    return self.as_dict().items()

  def get(self, key: str, default: Any = None) -> Any:
    return self.as_dict().get(key, default)

  def persist(self) -> None:
    pass

  def __getitem__(self, key: str) -> Any:
    self._bound(f"[{key!r}]")
    return self._values[key]

  def __setitem__(self, key: str, value: Any) -> None:
    self.update({key: value})

  def __getattr__(self, key: str) -> Any:
    if key.startswith('_'):
      raise AttributeError(key)
    self._bound(f".{key}")
    try:
      return self._values[key]
    except KeyError:
      raise AttributeError(f"wandb.config has no {key!r}") from None

  def __setattr__(self, key: str, value: Any) -> None:
    self.update({key: value})

  def __contains__(self, key: object) -> bool:
    return key in self.as_dict()

  def __iter__(self) -> Iterator[str]:
    return iter(self.as_dict())

  def __len__(self) -> int:
    return len(self.as_dict())

  def __repr__(self) -> str:
    return f"Config({self._values!r})"


class Summary:
  """
  `wandb.summary` and `run.summary`: the run's metrics. Each series' last value is one, or the summary chosen with
  `define_metric()`. What you set is a metric of the run (`log_metrics()`), and replaces a series' summary.
  """
  def __init__(self, run: Optional['Run'] = None):
    object.__setattr__(self, '_run', run)

  def _bound(self, call: str) -> 'Run':
    if self._run is None:
      raise Error(f"call wandb.init() before wandb.summary{call}")
    return self._run

  def update(self, values: Mapping[str, Any]) -> None:
    self._bound('.update()')._set_summary(dict(values))

  def as_dict(self) -> Dict[str, Any]:
    return self._bound('')._summary_values()

  def keys(self) -> Any:
    return self.as_dict().keys()

  def values(self) -> Any:
    return self.as_dict().values()

  def items(self) -> Any:
    return self.as_dict().items()

  def get(self, key: str, default: Any = None) -> Any:
    return self.as_dict().get(key, default)

  def __getitem__(self, key: str) -> Any:
    return self.as_dict()[key]

  def __setitem__(self, key: str, value: Any) -> None:
    self.update({key: value})

  def __delitem__(self, key: str) -> None:
    _warn_once('summary-delete', "WARNING: del wandb.summary[...] isn't supported by QA-Board: the metric stays.")

  def __getattr__(self, key: str) -> Any:
    if key.startswith('_'):
      raise AttributeError(key)
    try:
      return self.as_dict()[key]
    except KeyError:
      raise AttributeError(f"wandb.summary has no {key!r}") from None

  def __setattr__(self, key: str, value: Any) -> None:
    self.update({key: value})

  def __contains__(self, key: object) -> bool:
    return key in self.as_dict()

  def __iter__(self) -> Iterator[str]:
    return iter(self.as_dict())

  def __len__(self) -> int:
    return len(self.as_dict())

  def __repr__(self) -> str:
    values = self.as_dict() if self._run is not None else {}
    return f"Summary({values!r})"


# --- media and artifacts ----------------------------------------------------------------------------------------------
Source = Tuple[Union[Path, bytes], str]   # a file to copy or bytes to write, and the file's extension


class _NotSaved(Exception):
  """A medium that can't be saved, for a reason already told"""


def _is_file(data: Any) -> bool:
  if not isinstance(data, (str, os.PathLike)):
    return False
  try:
    return Path(data).is_file()
  except (OSError, ValueError):  # e.g. a text too long to be a path
    return False


class _Media:
  """
  What wandb logs that isn't a number. QA-Board keeps the file it was made from, if any, as a file of the run: in
  `wandb.log({"examples": wandb.Image(path)})`, at media/examples/<step>.png. Other media are skipped with a warning.
  """
  def __init__(self, data_or_path: Any = None, *args: Any, caption: Optional[str] = None, **kwargs: Any):
    self._data = data_or_path
    self.caption = caption

  def _source(self) -> Optional[Source]:
    data = self._data
    if _is_file(data):
      return Path(data), Path(data).suffix
    if isinstance(data, (str, os.PathLike)):
      _warn_once(f"no-file:{data}", f"WARNING: wandb.{type(self).__name__}({str(data)!r}): no such file, skipped.")
      raise _NotSaved()
    return None

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_') or name in PROTOCOL:
      raise AttributeError(name)
    return _Unsupported(f"wandb.{type(self).__name__}.{name}")


def _image_array(data: Any) -> Any:
  """An image as an array of uint8 Pillow takes, like wandb.Image: channels last, [0, 1] and [-1, 1] floats scaled"""
  for method in ('detach', 'cpu', 'numpy'):  # PyTorch and TensorFlow tensors
    if hasattr(data, method) and callable(getattr(data, method)):
      data = getattr(data, method)()
  array = data
  if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
    array = array.transpose(1, 2, 0)
  if array.ndim == 3 and array.shape[-1] == 1:
    array = array[:, :, 0]
  if array.dtype.kind == 'f':
    if array.min() < 0:
      array = (array + 1) * 127.5
    elif array.max() <= 1:
      array = array * 255
  if array.dtype.kind == 'b':
    array = array * 255
  return array.clip(0, 255).astype('uint8')


class Image(_Media):
  """`wandb.Image(path | PIL image | matplotlib figure | array | tensor, caption=...)`, saved as a file of the run when logged"""
  def _source(self) -> Optional[Source]:
    found = super()._source()
    if found is not None:
      return found
    data = self._data
    if hasattr(data, 'savefig'):  # a matplotlib figure
      buffer = io.BytesIO()
      data.savefig(buffer, format='png')
      return buffer.getvalue(), '.png'
    if hasattr(data, 'save') and hasattr(data, 'mode'):  # a PIL image
      buffer = io.BytesIO()
      data.save(buffer, format='PNG')
      return buffer.getvalue(), '.png'
    if hasattr(data, 'shape'):
      try:
        from PIL import Image as PILImage  # type: ignore[import-not-found,unused-ignore]
      except ImportError:
        _warn_once('pillow', "WARNING: wandb.Image() of an array is saved with Pillow, which isn't installed: skipped. pip install pillow, or log images from files.")
        raise _NotSaved() from None
      buffer = io.BytesIO()
      PILImage.fromarray(_image_array(data)).save(buffer, format='PNG')
      return buffer.getvalue(), '.png'
    return None


class Html(_Media):
  """`wandb.Html(text | file | path)`, saved as an .html file of the run when logged"""
  def _source(self) -> Optional[Source]:
    data = self._data
    if hasattr(data, 'read'):
      data = data.read()
    if _is_file(data):
      return super()._source()
    if isinstance(data, str):
      data = data.encode('utf-8')
    return (data, '.html') if isinstance(data, bytes) else None


class Video(_Media):
  """`wandb.Video(path)`: the file is saved with the run. Videos made from arrays are skipped."""


class Audio(_Media):
  """`wandb.Audio(path)`: the file is saved with the run. Audio made from arrays is skipped."""


class Object3D(_Media):
  """`wandb.Object3D(path)`: the file is saved with the run, e.g. a point cloud QA-Board's 3D viewer opens"""


class Molecule(_Media):
  """`wandb.Molecule(path)`: the file is saved with the run"""


class Table(_Media):
  """`wandb.Table`: not kept by QA-Board. Logging one warns once. Save a CSV with `log_artifact()` instead."""


class Histogram(_Media):
  """`wandb.Histogram`: not kept by QA-Board. Logging one warns once."""


class Plotly(_Media):
  """`wandb.Plotly`: not kept by QA-Board. Save the figure's HTML with `wandb.Html(figure.to_html())`."""


class Artifact:
  """
  `wandb.Artifact(name, type)`: files to keep with the run. Logged with `log_artifact()`, they are saved in the run's
  folder, under artifacts/<name>/. QA-Board has no artifact versions, aliases, references or lineage.
  """
  def __init__(self, name: str, type: str = 'unspecified', description: Optional[str] = None,
               metadata: Optional[Mapping[str, Any]] = None, **kwargs: Any):
    self.name = str(name).replace('/', '-').replace(':', '-')
    self.type = type
    self.description = description
    self.metadata = dict(metadata or {})
    self._files: List[Tuple[str, Union[Path, bytes]]] = []

  def add_file(self, local_path: Union[str, Path], name: Optional[str] = None, **kwargs: Any) -> None:
    path = Path(local_path)
    self._files.append((name or path.name, path))

  def add_dir(self, local_path: Union[str, Path], name: Optional[str] = None, **kwargs: Any) -> None:
    root = Path(local_path)
    for path in sorted(root.rglob('*')):
      if path.is_file():
        relative = path.relative_to(root)
        self._files.append(((Path(name) / relative if name else relative).as_posix(), path))

  @contextmanager
  def new_file(self, name: str, mode: str = 'x', encoding: Optional[str] = None) -> Iterator[Any]:
    buffer: Any = io.BytesIO() if 'b' in mode else io.StringIO()
    yield buffer
    content = buffer.getvalue()
    self._files.append((name, content if isinstance(content, bytes) else content.encode(encoding or 'utf-8')))

  def add(self, obj: Any, name: str) -> None:
    try:
      source = obj._source() if isinstance(obj, _Media) else None
    except _NotSaved:
      return
    if source is None:
      _warn_once(f"artifact-add:{type(obj).__name__}", f"WARNING: Artifact.add() of a {type(obj).__name__} isn't supported by QA-Board: skipped.")
      return
    content, extension = source
    self._files.append((name if name.endswith(extension) else f"{name}{extension}", content))

  def add_reference(self, uri: Any, name: Optional[str] = None, **kwargs: Any) -> None:
    _warn_once('artifact-reference', "WARNING: Artifact.add_reference() isn't supported by QA-Board: the reference isn't saved.")

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
      raise AttributeError(name)
    return _unsupported(f"wandb.Artifact.{name}")

  def __repr__(self) -> str:
    return f"<Artifact {self.name} ({self.type}), {len(self._files)} files>"


# --- runs -------------------------------------------------------------------------------------------------------------
class Run:
  """
  What `wandb.init()` returns, also `wandb.run`. Its QA-Board run starts with `qaboard.start_run()` at the first call
  that needs it: `run.qaboard` is that `ActiveRun`, for what only QA-Board does.
  """
  def __init__(self, *, project: Optional[str], entity: Optional[str], group: Optional[str], name: Optional[str],
               notes: Optional[str], tags: Optional[List[str]], job_type: Optional[str], config: Mapping[str, Any],
               id: str, id_given: bool, resume: bool, offline: bool, disabled: bool, options: Dict[str, Any],
               overrides: Dict[str, Any]):
    self.project = project
    self.entity = entity
    self.group = group
    self.job_type = job_type
    self.id = id
    self.resumed = resume
    self.offline = offline
    self.disabled = disabled
    self.sweep_id: Optional[str] = None
    self.start_time = time.time()
    self.config = Config(self, config)
    self.summary = Summary(self)

    self._name = name
    self._tags = tuple(tags or ())
    self._notes = notes
    self._recorded: Dict[str, Any] = {k: v for k, v in {
      'name': name, 'id': id if id_given else None, 'job_type': job_type, 'tags': list(self._tags) or None, 'notes': notes,
    }.items() if v is not None}
    self._options = options        # start_run()'s arguments, without the configuration
    self._overrides = overrides    # init(qaboard={...})
    self._active: Optional[ActiveRun] = None
    self._lock = threading.RLock()
    self._finished = False
    self._step = 0                 # wandb's: the step being logged
    self._pending: Dict[str, Any] = {}   # its values, until it is committed
    self._summaries: Dict[str, str] = {}  # define_metric(summary=) before the run started
    self._goals: Dict[str, bool] = {}     # define_metric(goal=): smaller is better, for names and patterns
    self._defined: Set[str] = set()       # series given their goal
    self._saved: List[Tuple[str, Optional[str]]] = []  # save(): copied again at the end
    self._copied: Dict[Path, Tuple[int, int]] = {}      # save(): the files copied, when they were as (mtime, size)
    self._steps_back = 0                  # log() calls skipped because their step went back
    self._disabled_summary: Dict[str, Any] = {}
    self._disabled_dir: Optional[str] = None
    self._pid = os.getpid()               # forked workers don't log to their parent's run
    self._broken = False                  # start_run() failed: the training goes on, without QA-Board

  # --- what the run is ---------------------------------------------------------------------------------------------
  @property
  def qaboard(self) -> Optional[ActiveRun]:
    """The QA-Board run, started if it wasn't. None with mode="disabled"."""
    return self._started()

  @property
  def name(self) -> str:
    return self._name or f"run-{self.id}"

  @name.setter
  def name(self, value: str) -> None:
    with self._lock:
      if self._active is not None and value != self._name:
        _warn_once('rename', "WARNING: wandb.run.name: the run started in QA-Board, where its name is its input: the new name is only recorded as a parameter. Set it before the first wandb.log().")
      self._name = value
      if value is not None:
        self._record({'name': value})

  @property
  def tags(self) -> Tuple[str, ...]:
    return self._tags

  @tags.setter
  def tags(self, value: Any) -> None:
    self._tags = tuple(value or ())
    self._record({'tags': list(self._tags)})

  @property
  def notes(self) -> Optional[str]:
    return self._notes

  @notes.setter
  def notes(self, value: Optional[str]) -> None:
    self._notes = value
    self._record({'notes': value})

  @property
  def step(self) -> int:
    """The step the next `log()` goes to. A resumed run starts in QA-Board to know it: steps go on from its last."""
    if self._options.get('resume') and not self._finished:
      self._started()
    return self._step

  @property
  def dir(self) -> str:
    """The run's folder: files written there are the run's, shown in the web app"""
    active = self._started()
    if active is None:
      if self._disabled_dir is None:
        import tempfile
        self._disabled_dir = tempfile.mkdtemp(prefix='wandb-disabled-')
      return self._disabled_dir
    return str(active.output_dir)

  @property
  def url(self) -> Optional[str]:
    """The run in the web app, when it is shared"""
    active = self._started()
    return active.url if active is not None else None

  def get_url(self) -> Optional[str]:
    return self.url

  def get_project_url(self) -> Optional[str]:
    return None

  def project_name(self) -> Optional[str]:
    return self.project

  @property
  def path(self) -> str:
    return '/'.join(p for p in (self.entity, self.project, self.id) if p)

  @property
  def mode(self) -> str:
    return 'disabled' if self.disabled else 'offline' if self.offline else 'online'

  def __repr__(self) -> str:
    return f"<wandb Run {self.name} -> {self._active!r}>" if not self.disabled else f"<wandb Run {self.name} (disabled)>"

  # --- logging -----------------------------------------------------------------------------------------------------
  def log(self, data: Mapping[str, Any], step: Optional[int] = None, commit: Optional[bool] = None, sync: Optional[bool] = None) -> None:
    """
    Logs values at a step, like wandb: without `step`, each call is a step; `commit=False` adds to the step until a
    call commits it; with `step`, values add to that step until a later step is logged. Numbers make series; images
    are saved as files of the run; other media, texts and lists are skipped with a warning.
    """
    if not isinstance(data, Mapping):
      raise ValueError(f"wandb.log() takes a dict, like wandb.log({{'loss': 0.3}}), not {type(data).__name__}")
    if self._done('log'):
      return
    with self._lock:
      if self._started() is None:
        return
      if step is not None:
        step = _as_step(step)
      if step is not None:
        if step < self._step:
          self._steps_back += 1
          _warn_once('step-back', f"WARNING: wandb.log(step={step}) after step {self._step}: steps only go up, these values are not logged (like wandb). How many calls were skipped is said at the end.")
          return
        if step > self._step:
          self._commit()
          self._step = step
      self._pending.update(flatten(data))
      if commit or (commit is None and step is None):
        self._commit()

  def define_metric(self, name: str, step_metric: Optional[str] = None, step_sync: Optional[bool] = None,
                    hidden: Optional[bool] = None, summary: Optional[str] = None, goal: Optional[str] = None,
                    overwrite: Optional[bool] = None, **kwargs: Any) -> None:
    """
    Says which value of a series is the run's metric, `summary="min"` (also max, mean, last, none, best with a goal),
    and which way is better, `goal="minimize"`. `name` can be a pattern, "val/*". Series are plotted by step and by
    time: `step_metric` is ignored, with a warning.
    """
    _ignore_arguments('wandb.define_metric', kwargs)
    if self._done('define_metric'):
      return
    if step_metric is not None:
      _warn_once('step-metric', "WARNING: wandb.define_metric(step_metric=...): QA-Board plots series by step and by time, not by another series: ignored.")
    if hidden:
      _warn_once('hidden', "WARNING: wandb.define_metric(hidden=True) isn't supported by QA-Board: the series is shown.")
    if goal is not None and goal not in ('minimize', 'maximize'):
      _warn_once(f"goal:{goal}", f"WARNING: wandb.define_metric(goal={goal!r}): expected 'minimize' or 'maximize': ignored.")
      goal = None
    with self._lock:
      if goal is not None:
        self._goals[name] = goal == 'minimize'
        if not PATTERN.search(name):
          self._define(name, goal == 'minimize')
      elif name in self._goals:  # given by an earlier define_metric()
        goal = 'minimize' if self._goals[name] else 'maximize'
      if summary is not None:
        how = self._summary_of(name, str(summary), goal)
        if how is not None:
          if self._active is not None:
            self._active.set_summary(name, how)
          else:
            self._summaries[name] = how

  def _summary_of(self, name: str, summary: str, goal: Optional[str]) -> Optional[str]:
    """wandb's summary (several, comma-separated) as one of QA-Board's: QA-Board keeps one per series"""
    hows = []
    for how in (s.strip() for s in summary.split(',') if s.strip()):
      if how == 'best':
        if goal is None:
          _warn_once('best', "WARNING: wandb.define_metric(summary='best') needs goal='minimize' or 'maximize': ignored.")
          continue
        how = 'min' if goal == 'minimize' else 'max'
      elif how == 'copy':
        how = 'last'
      if how not in SUMMARIES:
        _warn_once(f"summary:{how}", f"WARNING: wandb.define_metric(summary={how!r}) isn't supported by QA-Board: ignored. Use {', '.join(SUMMARIES)}.")
        continue
      hows.append(how)
    if len(hows) > 1:
      _warn_once(f"summaries:{name}", f"WARNING: wandb.define_metric({name!r}, summary={summary!r}): QA-Board keeps one summary per series: {hows[0]}.")
    return hows[0] if hows else None

  def _define(self, key: str, smaller_is_better: bool) -> None:
    from ..metrics import define_metric as define_qaboard_metric
    define_qaboard_metric(key, smaller_is_better=smaller_is_better)
    self._defined.add(key)

  def save(self, glob_str: Optional[Union[str, Path]] = None, base_path: Optional[Union[str, Path]] = None, policy: str = 'live') -> List[str]:
    """
    Copies files into the run's folder, now and again when the run finishes (unless `policy="now"`). Their name is
    their path relative to `base_path`; without it, relative to the current folder, or their name alone for an
    absolute pattern, like wandb. Files already in the run's folder (`wandb.run.dir`) are the run's: nothing to copy.
    """
    if glob_str is None or self._done('save'):
      return []
    with self._lock:
      if self._started() is None:
        return []
      pattern = (str(glob_str), None if base_path is None else str(base_path))
      saved = self._copy_files(*pattern)
      if policy != 'now' and pattern not in self._saved:
        self._saved.append(pattern)
      return saved

  def log_artifact(self, artifact_or_path: Union[Artifact, str, Path], name: Optional[str] = None, type: Optional[str] = None,
                   aliases: Optional[List[str]] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Artifact:
    """Saves an `Artifact`, or a file or folder, in the run's folder, under artifacts/<name>/"""
    _ignore_arguments('wandb.log_artifact', kwargs)
    if isinstance(artifact_or_path, Artifact):
      artifact = artifact_or_path
    else:
      path = Path(artifact_or_path)
      artifact = Artifact(name or (path.stem if path.is_file() else path.name) or 'artifact', type or 'unspecified')
      if path.is_dir():
        artifact.add_dir(path)
      else:
        artifact.add_file(path)
    if self._done('log_artifact'):
      return artifact
    with self._lock:
      active = self._started()
      if active is None:
        return artifact
      for entry, source in artifact._files:
        self._keep(active, f"artifacts/{artifact.name}/{entry}", source)
    return artifact

  def log_model(self, path: Union[str, Path], name: Optional[str] = None, aliases: Optional[List[str]] = None, **kwargs: Any) -> None:
    """Saves a model's file or folder in the run's folder, under artifacts/<name>/"""
    self.log_artifact(path, name=name, type='model', aliases=aliases, **kwargs)

  def use_artifact(self, *args: Any, **kwargs: Any) -> Any:
    return _unsupported('wandb.use_artifact()', ': read the files from where they are')

  def watch(self, *args: Any, **kwargs: Any) -> None:
    watch(*args, **kwargs)

  def unwatch(self, *args: Any, **kwargs: Any) -> None:
    pass

  def alert(self, title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
    alert(title, text, level=level, wait_duration=wait_duration)

  def mark_preempting(self) -> None:
    pass

  def log_code(self, *args: Any, **kwargs: Any) -> None:
    log_code(*args, **kwargs)

  # --- lifecycle ---------------------------------------------------------------------------------------------------
  def finish(self, exit_code: Optional[int] = None, quiet: Optional[bool] = None) -> None:
    """Ends the run: as failed with an exit code other than 0"""
    if self._forked('finish'):
      return
    with self._lock:
      if self._finished:
        return
      self._finished = True
      try:
        active = self._started()
        if active is not None:
          self._flush(active)
          if exit_code:
            active.fail(f"exit code {exit_code}")
          else:
            active.finish()
      finally:
        _forget(self)

  def __enter__(self) -> 'Run':
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    if self._forked('finish'):
      return
    with self._lock:
      if not self._finished:
        self._finished = True
        try:
          active = self._started()
          if active is not None:
            self._flush(active)
            active.__exit__(exc_type, exc, tb)  # an exception fails the run
        finally:
          _forget(self)

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
      raise AttributeError(name)
    return _unsupported(f"wandb.run.{name}")

  # --- internals ---------------------------------------------------------------------------------------------------
  def _done(self, call: str) -> bool:
    if self._finished:
      _warn_once(f"finished:{call}", f"WARNING: wandb.{call}() after the run finished: ignored.")
      return True
    return False

  def _config_values(self) -> Dict[str, Any]:
    return dict(object.__getattribute__(self.config, '_values'))

  def _forked(self, call: str) -> bool:
    """Whether we are in a process forked from the run's (a data loader's worker, a pool's): it isn't its run"""
    if os.getpid() == self._pid:
      return False
    _warn_once(f"fork:{os.getpid()}", f"WARNING: wandb.{call}() in a process forked from the run's: ignored. Log from the process that called wandb.init(), or call wandb.init() in the worker for a run of its own.")
    return True

  def _started(self) -> Optional[ActiveRun]:
    """
    The QA-Board run, started the first time it is needed, with the configuration and name set until then. None when
    disabled, in a forked process, or when it could not start: the training goes on.
    """
    if self.disabled or self._broken or self._forked('log'):
      return None
    with self._lock:
      if self._active is None:
        options, reported = self._start_options()
        _same_identity(options)
        try:
          self._active = start_run(**options)
        except Exception as e:  # init() checked the setup: what is left is said, and the training goes on
          self._broken = True
          _say(f"WARNING: could not start the run in QA-Board: {e}")
          _say("         The training goes on without it: wandb calls do nothing for this run.")
          return None
        if self._options.get('resume'):
          self._step = max(self._step, self._active.last_step + 1)
        if reported:
          self._active.log_params(reported)
        if self._recorded:
          self._active.log_params({'wandb': dict(self._recorded)})
        configs = json.dumps(_jsonable(list(self._active.configs)), sort_keys=True)
        configs = f"the configurations {configs[:300]}{'...' if len(configs) > 300 else ''}" if self._active.configs else 'no configuration'
        _say(f"the wandb run is the input {self._active.input!r} with {configs}, in the batch {self._active.batch!r}: the same on the next commit is compared with it.")
        _watch_exit(self)
      return self._active

  def _start_options(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """start_run()'s arguments, and the configuration's values that are only reported, as parameters"""
    configuration = self._config_values()
    overrides = dict(self._overrides)
    excluded = set(overrides.pop('params', None) or ())
    identity: Dict[str, Any] = {}
    reported: Dict[str, Any] = {}
    for key, value in configuration.items():
      if key in excluded:
        reported[key] = value
      elif not _identifies(value):
        _warn_once(f"object:{key}", f"wandb.config[{key!r}] is a {type(value).__name__}, not plain data: recorded as a parameter, not in the run's identity.")
        reported[key] = value
      else:
        identity[key] = value
        for name, item in flatten({key: value}).items():
          if _volatile(item):
            _warn_once(f"volatile:{name}", f"WARNING: wandb.config[{name!r}] = {str(item)[:80]!r} looks like it changes at every run (a date, a time, a job id): as part of the run's identity, the next commit's run won't be compared with this one. To record it as a parameter instead: wandb.init(qaboard={{'params': [{key!r}]}}).")
    if self._name and (_volatile(self._name) or self.id in self._name):
      _warn_once('volatile:name', f"WARNING: wandb.init(name={self._name!r}) looks like it changes at every run (a date, a job id, the run's id): it is the run's input in QA-Board, so the next commit's run won't be compared with this one. Give a stable name, or none.")
    for what, value in (('project', self.project), ('group', self.group)):
      if value and _volatile(value):
        _warn_once(f"volatile:{what}", f"WARNING: wandb.init({what}={value!r}) looks like it changes at every run: it names the batch, so runs won't be compared from commit to commit. Use a stable {what}.")
    options: Dict[str, Any] = {**self._options, 'input': _input_of(self._name, self.job_type), 'configs': [identity] if identity else None}
    if self._summaries:
      options['summary'] = dict(self._summaries)
    if 'configs' in overrides:
      # start_run() gets its own configurations: wandb's is what the run reports
      reported = configuration
    options.update(overrides)
    return options, reported

  def _record(self, values: Dict[str, Any]) -> None:
    """The wandb.* parameters"""
    with self._lock:
      self._recorded.update({k: v for k, v in values.items() if v is not None})
      if self._active is not None and not self._finished:
        self._active.log_params({'wandb': self._recorded})

  def _set_config(self, values: Dict[str, Any]) -> None:
    if not values:
      return
    with self._lock:
      object.__getattribute__(self.config, '_values').update(values)
      if self._active is not None and not self._finished:
        try:
          self._active.log_params(values)
        except Exception as e:  # logging never ends a training
          _warn_once('params', f"WARNING: wandb.config: could not record {sorted(values)}: {e}")

  def _summary_values(self) -> Dict[str, Any]:
    if self._active is not None:
      return dict(self._active.metrics)
    return dict(self._disabled_summary)

  def _set_summary(self, values: Dict[str, Any]) -> None:
    if self._done('summary'):
      return
    with self._lock:
      active = self._started()
      flat = flatten(values)
      if active is None:
        self._disabled_summary.update(flat)
        return
      for key, value in flat.items():
        if isinstance(value, (_Media, _Unsupported)):
          self._skip(key, value, 'wandb.summary')
          continue
        try:
          active.log_metrics({key: value})
        except (TypeError, ValueError) as e:
          _warn_once(f"summary:{key}", f"WARNING: wandb.summary[{key!r}]: not kept: {e}")

  def _skip(self, key: str, value: Any, call: str = 'wandb.log()') -> None:
    kind = value._name if isinstance(value, _Unsupported) else type(value).__name__
    _warn_once(f"skip:{call}:{kind}", f"WARNING: {call}: {kind} values aren't kept by QA-Board: skipped, like {key!r}. Numbers make series; save files with wandb.save() or in wandb.run.dir.")

  def _commit(self) -> None:
    """Logs the step's values, and goes to the next step"""
    if not self._pending:
      return
    active = self._active
    assert active is not None
    pending, self._pending = self._pending, {}
    step = self._step
    self._step += 1
    numbers: Dict[str, Any] = {}
    for key, value in pending.items():
      if value is None:
        continue
      if not key or key.startswith('_') or key in RESERVED_METRICS or len(key) > MAX_NAME_LENGTH:
        _warn_once(f"key:{key[:40]}", f"WARNING: wandb.log(): QA-Board doesn't take {key[:40]!r} as a series' name (1 to {MAX_NAME_LENGTH} characters, not starting with '_', not {' or '.join(RESERVED_METRICS)}): skipped.")
        continue
      if hasattr(value, 'savefig'):  # a matplotlib figure, or pyplot itself
        value = Image(value)
      if isinstance(value, _Media) or (isinstance(value, (list, tuple)) and value and all(isinstance(v, _Media) for v in value)):
        self._log_media(active, key, value, step)
        continue
      try:
        as_number(value)
      except TypeError:
        self._skip(key, value)
        continue
      numbers[key] = value  # as given: a Metric(...) keeps its definition
      if key not in self._defined:
        for pattern, smaller_is_better in self._goals.items():
          if fnmatchcase(key, pattern):
            self._define(key, smaller_is_better)
            break
    if numbers:
      try:
        active.log(numbers, step=step)
      except Exception as e:  # logging never ends a training
        _warn_once('log', f"WARNING: wandb.log(): not logged: {e}")

  def _log_media(self, active: ActiveRun, key: str, value: Any, step: int) -> None:
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    for index, item in enumerate(items):
      try:
        source = item._source()
      except _NotSaved:
        continue
      except Exception as e:
        _warn_once(f"media:{type(item).__name__}", f"WARNING: wandb.log(): could not save a {type(item).__name__}: {e!r}")
        continue
      if source is None:
        self._skip(key, item)
        continue
      content, extension = source
      folder = f"media/{key}/{index}" if len(items) > 1 else f"media/{key}"
      self._keep(active, f"{folder}/{step}{extension}", content)

  def _keep(self, active: ActiveRun, name: str, source: Union[Path, bytes]) -> None:
    try:
      active.log_artifact(name, source)
    except Exception as e:  # logging never ends a training
      _warn_once(f"file:{type(e).__name__}", f"WARNING: could not save {name} with the run: {e}")

  def _copy_files(self, pattern: str, base_path: Optional[str]) -> List[str]:
    active = self._active
    assert active is not None
    pattern = os.path.expanduser(pattern)
    if base_path is not None:
      base = Path(base_path).expanduser().resolve()
    else:
      base = Path(os.path.dirname(pattern)).resolve() if os.path.isabs(pattern) else Path.cwd().resolve()
    output_dir = active.output_dir.resolve()
    saved = []
    for found in sorted(glob(pattern, recursive=True)):
      path = Path(found).resolve()
      if not path.is_file() or path == output_dir or output_dir in path.parents:
        continue
      try:
        name = path.relative_to(base).as_posix()
      except ValueError:
        name = path.name
      stat = path.stat()
      if self._copied.get(path) != (stat.st_mtime_ns, stat.st_size):  # else copied already, unchanged since
        self._keep(active, name, path)
        self._copied[path] = (stat.st_mtime_ns, stat.st_size)
      saved.append(str(path))
    return saved

  def _flush(self, active: ActiveRun) -> None:
    """At the end: the step not committed yet, and the files save() asked to keep"""
    with self._lock:
      self._commit()
      for pattern, base_path in self._saved:
        self._copy_files(pattern, base_path)
      if self._steps_back:
        _say(f"WARNING: {self._steps_back} wandb.log() calls with a step lower than the previous were not logged.")
        self._steps_back = 0


# --- the module's state, like wandb's: wandb.run, wandb.config, wandb.summary -------------------------------------------
run: Optional[Run] = None
config: Config = Config()
summary: Summary = Summary()


def _set_current(current: Optional[Run]) -> None:
  global run, config, summary
  with _lock:
    run = current
    config = current.config if current is not None else Config()
    summary = current.summary if current is not None else Summary()


def _forget(finished: Run) -> None:
  with _lock:
    if finished in _unfinished:
      _unfinished.remove(finished)
    if run is finished:
      _set_current(None)


def _watch_exit(started: Run) -> None:
  """
  Runs nobody finished end with the process (qaboard/active_run.py): before, we log what they still buffer. Registered
  after start_run() registered its own, so it runs first.
  """
  global _exit_hook
  with _lock:
    if started not in _unfinished:
      _unfinished.append(started)
    if not _exit_hook:
      _exit_hook = True
      atexit.register(_at_exit)


_identities: Set[str] = set()   # of the runs started in this process


def _same_identity(options: Mapping[str, Any]) -> None:
  """Warns when a run starts with the same identity as one before in this process: it replaces its files"""
  if options.get('resume'):
    return
  key = json.dumps(_jsonable({k: options.get(k) for k in ('project', 'commit', 'batch', 'input', 'configs', 'share', 'output_dir')}), sort_keys=True)
  with _lock:
    seen = key in _identities
    _identities.add(key)
  if seen:
    _say(f"WARNING: this run has the same batch, input ({options.get('input')!r}) and configuration as a run started before by this process: it is a new attempt of that run, and replaces its files. Give each run its own name= or job_type=.")


def _check_setup(overrides: Mapping[str, Any]) -> None:
  """What start_run() would refuse about the project or the commit, said by init() rather than by the first log()"""
  from ..active_run import _Setup
  try:
    _Setup(overrides.get('project'), overrides.get('commit'))
  except ValueError as e:
    message = str(e)
    if 'needs to know the project' in message:
      message = ("QA-Board doesn't know your project (your repository): run the script from a folder with a qaboard.yaml "
                 "(see `qa init`), or pass wandb.init(qaboard={'project': 'group/repo'}). wandb's project= names the run's batch.")
    raise Error(f"wandb.init(): {message}") from None


def _at_exit() -> None:
  with _lock:
    runs = list(_unfinished)
  for unfinished in runs:
    if unfinished._active is not None and not unfinished._finished and unfinished._pid == os.getpid():
      try:
        unfinished._flush(unfinished._active)
      except Exception as e:
        _say(f"WARNING: could not log the last values: {e!r}")


def _current(call: str) -> Run:
  if run is None:
    raise Error(f"call wandb.init() before wandb.{call}()")
  return run


def _env(name: str) -> Optional[str]:
  return os.environ.get(name) or None


def init(entity: Optional[str] = None, project: Optional[str] = None, dir: Optional[str] = None, id: Optional[str] = None,
         name: Optional[str] = None, notes: Optional[str] = None, tags: Optional[List[str]] = None, config: Any = None,
         config_exclude_keys: Optional[List[str]] = None, config_include_keys: Optional[List[str]] = None,
         allow_val_change: Optional[bool] = None, group: Optional[str] = None, job_type: Optional[str] = None,
         mode: Optional[str] = None, force: Optional[bool] = None, anonymous: Optional[str] = None, reinit: Any = None,
         resume: Union[bool, str, None] = None, resume_from: Optional[str] = None, fork_from: Optional[str] = None,
         save_code: Optional[bool] = None, tensorboard: Optional[bool] = None, sync_tensorboard: Optional[bool] = None,
         monitor_gym: Optional[bool] = None, settings: Any = None, qaboard: Optional[Mapping[str, Any]] = None,
         **kwargs: Any) -> Run:
  """
  Starts a run, like `wandb.init()`, and returns it; also `wandb.run`, `wandb.config` and `wandb.summary`.
  - `project` and `group` make the run's batch, "project/group". QA-Board's project is the repository.
  - `name` is the run's input, `config` its configuration: they identify it, and compare it with the next commit's.
    Without a name, the input is the script's name, and the `job_type`. Use stable names: a date or a job id makes
    every run new.
  - The run starts in QA-Board at the first call that needs it, with the configuration set until then.
  - `mode="disabled"` logs nothing; `mode="offline"` keeps the run in the project's output/ folder.
  - `resume="allow"`, "must" or True continue the previous run of the same name and configuration.
  - `qaboard={...}`: arguments of `qaboard.start_run()`, which win: `qaboard={"input": "train/coco-mini"}`; and
    `"params": ["seed", ...]`, configuration keys recorded as parameters only, not part of the run's identity.
  Other arguments are ignored, with a warning for those that would change what you see. Raises `Error` when QA-Board
  can't make the run: outside a project, give `qaboard={"project": "group/repo"}`.
  """
  _ignore_arguments('wandb.init', kwargs)
  for argument, value in (('resume_from', resume_from), ('fork_from', fork_from), ('tensorboard', tensorboard),
                          ('sync_tensorboard', sync_tensorboard), ('monitor_gym', monitor_gym)):
    if value:
      _warn_once(f"argument:wandb.init:{argument}", f"WARNING: wandb.init({argument}=...) isn't supported by QA-Board: ignored.")
  if qaboard is not None and not isinstance(qaboard, Mapping):
    raise TypeError(f"wandb.init(qaboard=...) takes start_run()'s arguments as a dict, like {{'input': 'train/coco-mini'}}, not {qaboard!r}")
  overrides = dict(qaboard or {})
  allowed = set(inspect.signature(start_run).parameters) | {'params'}
  unknown = sorted(set(overrides) - allowed)
  if unknown:
    raise TypeError(f"wandb.init(qaboard=...): {', '.join(unknown)}: expected arguments of qaboard.start_run(), or params: {', '.join(sorted(allowed))}")
  if not isinstance(overrides.get('params') or [], (list, tuple, set)):
    raise TypeError("wandb.init(qaboard={'params': [...]}) lists the configuration's keys that are only parameters, not part of the run's identity")

  # what wandb reads from the environment
  project = project or _env('WANDB_PROJECT')
  entity = entity or _env('WANDB_ENTITY')
  group = group or _env('WANDB_RUN_GROUP')
  name = name or _env('WANDB_NAME')
  notes = notes or _env('WANDB_NOTES')
  job_type = job_type or _env('WANDB_JOB_TYPE')
  if tags is None and _env('WANDB_TAGS'):
    tags = [t.strip() for t in (_env('WANDB_TAGS') or '').split(',') if t.strip()]
  if mode is None and _env('WANDB_MODE'):
    mode = _env('WANDB_MODE')
    if mode in OFFLINE_MODES + ('disabled',):
      _warn_once('env-mode', f"WANDB_MODE={mode}: {'nothing is logged' if mode == 'disabled' else 'runs stay in the output/ folder, and the server is not told'}.")
  if (_env('WANDB_DISABLED') or '').lower() in TRUE_VALUES:
    _warn_once('env-disabled', "WANDB_DISABLED: nothing is logged.")
    mode = 'disabled'
  if resume is None and _env('WANDB_RESUME'):
    resume = _env('WANDB_RESUME')
  id_given = bool(id or _env('WANDB_RUN_ID'))
  run_id = str(id or _env('WANDB_RUN_ID') or util.generate_id())

  if mode not in ONLINE_MODES + OFFLINE_MODES + ('disabled',):
    _warn_once(f"mode:{mode}", f"WARNING: wandb.init(mode={mode!r}): expected online, offline or disabled: online.")
    mode = None
  if resume not in RESUME_VALUES and resume not in (None, False, 'never'):
    _warn_once(f"resume:{resume}", f"WARNING: wandb.init(resume={resume!r}): expected 'allow', 'must', 'never', True or False: not resuming.")

  # a run is active: what wandb's reinit says
  with _lock:
    previous = run
  # a run of the process we were forked from isn't ours: neither returned, nor finished
  if previous is not None and not previous._finished and previous._pid == os.getpid():
    if reinit in (True, 'finish_previous') or (reinit in (None, 'default') and 'ipykernel' in sys.modules):
      previous.finish()
    elif reinit != 'create_new':
      _warn_once('reinit', "WARNING: wandb.init() while a run is active returns it, like wandb. Call wandb.finish() first, or pass reinit='finish_previous'.")
      return previous

  # wandb reads config-defaults.yaml in the current folder
  values = _as_mapping('config-defaults.yaml', 'wandb.init()') if os.path.isfile('config-defaults.yaml') else {}
  values.update(_as_mapping(config, 'wandb.init(config=...)'))
  if config_include_keys:
    values = {k: v for k, v in values.items() if k in config_include_keys}
  if config_exclude_keys:
    values = {k: v for k, v in values.items() if k not in config_exclude_keys}

  batch = '/'.join(str(p).strip('/') for p in (project, group) if p and str(p).strip('/')) or None
  options: Dict[str, Any] = {'batch': batch, 'resume': resume in RESUME_VALUES}
  if mode in OFFLINE_MODES:
    options['share'] = False
  if mode != 'disabled':
    _check_setup(overrides)
  started = Run(
    project=project, entity=entity, group=group, name=name, notes=notes, tags=list(tags) if tags else None,
    job_type=job_type, config=values, id=run_id, id_given=id_given, resume=resume in RESUME_VALUES,
    offline=mode in OFFLINE_MODES, disabled=mode == 'disabled', options=options, overrides=overrides,
  )
  _set_current(started)
  return started


def log(data: Mapping[str, Any], step: Optional[int] = None, commit: Optional[bool] = None, sync: Optional[bool] = None) -> None:
  """`wandb.log()`: see `Run.log()`"""
  _current('log').log(data, step=step, commit=commit)


def finish(exit_code: Optional[int] = None, quiet: Optional[bool] = None) -> None:
  """Ends `wandb.run`, if any: as failed with an exit code other than 0"""
  if run is not None:
    run.finish(exit_code=exit_code)


def define_metric(name: str, step_metric: Optional[str] = None, step_sync: Optional[bool] = None, hidden: Optional[bool] = None,
                  summary: Optional[str] = None, goal: Optional[str] = None, overwrite: Optional[bool] = None, **kwargs: Any) -> None:
  """`wandb.define_metric()`: see `Run.define_metric()`"""
  _current('define_metric').define_metric(name, step_metric=step_metric, step_sync=step_sync, hidden=hidden,
                                          summary=summary, goal=goal, overwrite=overwrite, **kwargs)


def save(glob_str: Optional[Union[str, Path]] = None, base_path: Optional[Union[str, Path]] = None, policy: str = 'live') -> List[str]:
  """`wandb.save()`: see `Run.save()`"""
  return _current('save').save(glob_str, base_path=base_path, policy=policy)


def log_artifact(artifact_or_path: Union[Artifact, str, Path], name: Optional[str] = None, type: Optional[str] = None,
                 aliases: Optional[List[str]] = None, **kwargs: Any) -> Artifact:
  """`wandb.log_artifact()`: see `Run.log_artifact()`"""
  return _current('log_artifact').log_artifact(artifact_or_path, name=name, type=type, aliases=aliases, **kwargs)


def log_model(path: Union[str, Path], name: Optional[str] = None, aliases: Optional[List[str]] = None, **kwargs: Any) -> None:
  """`wandb.log_model()`: see `Run.log_model()`"""
  _current('log_model').log_model(path, name=name, aliases=aliases, **kwargs)


def use_artifact(*args: Any, **kwargs: Any) -> Any:
  return _unsupported('wandb.use_artifact()', ': read the files from where they are')


def watch(*args: Any, **kwargs: Any) -> None:
  """Not supported: gradients and parameters aren't logged"""
  _unsupported('wandb.watch()', ': gradients and parameters are not logged')


def unwatch(*args: Any, **kwargs: Any) -> None:
  pass


def alert(title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
  """Printed, and saved in the run's log: QA-Board doesn't send alerts"""
  _warn_once('alert', "wandb.alert(): QA-Board doesn't send alerts: they are printed in the run's log.")
  level_name = getattr(level, 'value', level) or 'INFO'
  _say(f"wandb.alert(): [{level_name}] {title}: {text}")


def login(*args: Any, **kwargs: Any) -> bool:
  """Does nothing: QA-Board uses the token of `qa auth login`, or QABOARD_TOKEN"""
  return True


def mark_preempting() -> None:
  pass


def log_code(*args: Any, **kwargs: Any) -> None:
  """Does nothing: runs record their commit, and whether the code differs from it"""
  _warn_once('log_code', "wandb.log_code(): runs record their git commit, and whether the code differs from it: nothing to save.")


def setup(*args: Any, **kwargs: Any) -> None:
  pass


def teardown(*args: Any, **kwargs: Any) -> None:
  finish()


def __getattr__(name: str) -> Any:
  """The rest of wandb's API (Api, sweep, agent, plot, Settings...): warns once, then does nothing"""
  if name.startswith('_'):
    raise AttributeError(name)
  return _unsupported(f"wandb.{name}")

