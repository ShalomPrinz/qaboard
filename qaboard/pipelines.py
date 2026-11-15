"""
Pipelines: batches whose run does steps one after the other, batches (`qa batch`) and commands.

```yaml
my-pipeline:
  type: pipeline
  steps:
  - batch: calibration-images
  - name: calibrate
    run: python calibrate.py {steps.calibration-images.output_dirs} --output {output_dir}/calibration.yaml
  - batch: evaluation
    configs: ["{output_dir}/calibration.yaml"]
  - run: python report.py {steps.evaluation.runs_file}
```

`qa batch my-pipeline` starts one run, the pipeline. Its output directory has the files its commands write, and:
- pipeline.json: the steps and their status, updated as they run (the web app shows it)
- steps/<name>/runs.json: the runs of a batch step, with their output directories, status and metrics
- batches.json: the runs of the batches, by batch (the first format, kept for the scripts that read it)
- tuning.json: the tuning parameters the pipeline got, if any

Steps can also be listed among the `configurations` (the first syntax): `steps:` is the same thing, easier to read.
User docs: website/docs/pipelines.mdx.
"""
import os
import re
import sys
import json
import time
import shlex
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click

# The input of the pipelines that don't give one. It doesn't have to exist: a pipeline works on what its steps say.
PIPELINE_INPUT = 'PIPELINE'
PIPELINE_TYPE = 'pipeline'

STEP_OPTIONS = {
  # option: other spellings
  'name': (),
  'allow_failure': ('allow-failure',),
  'runner': (),
  # batch steps only
  'label': (),
  'tuning': (),
  'configs': ('configurations', 'configuration'),
  'database': (),
  'batches_file': ('batches-file', 'batches-files', 'batches_files'),
  'use_cache': ('use-cache',),
  'commit': (),
}
BATCH_ONLY_OPTIONS = ('label', 'tuning', 'configs', 'database', 'batches_file', 'use_cache', 'commit')
_OPTION_NAMES = {spelling: option for option, others in STEP_OPTIONS.items() for spelling in (option, *others)}
_VALID_NAME = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_+-]*$')
# `python my_script.py` is named after my_script: we look for a script among the arguments of these
_INTERPRETERS = {
  'python', 'python2', 'python3', 'py', 'uv', 'uvx', 'pipx', 'poetry', 'conda', 'bash', 'sh', 'zsh', 'csh', 'tcsh', 'source',
  'env', 'sudo', 'time', 'nice', 'matlab', 'octave', 'rscript', 'node', 'npx', 'perl', 'ruby', 'run', 'exec', 'call', 'cmd', 'powershell',
}
# ...and these say nothing about what the step does
_SHELL_COMMANDS = {'cd', 'echo', 'export', 'set', 'mkdir', 'cp', 'mv', 'rm', 'ls', 'touch', 'cat', 'true', 'false', 'pwd', 'sleep', 'chmod'}
_SCRIPT = re.compile(r'^(.+)\.(py|sh|bash|csh|m|r|pl|rb|js|ts|exe|bat|cmd|ps1|ipynb)$', re.IGNORECASE)


class PipelineError(Exception):
  """A mistake in the definition of a pipeline, with a message for the people who wrote it."""


def engine(config: Optional[Dict], environ=os.environ) -> str:
  """
  Who runs the runs of type `pipeline`: 'builtin' (default, this module) or 'entrypoint' (the project's run()
  function, for the projects that had their own pipelines before they were part of QA-Board).
  $QABOARD_PIPELINES_ENGINE, else `pipelines.engine` in qaboard.yaml.
  """
  value = environ.get('QABOARD_PIPELINES_ENGINE') or ((config or {}).get('pipelines') or {}).get('engine') or 'builtin'
  return 'entrypoint' if str(value).lower() in ('entrypoint', 'custom', 'legacy') else 'builtin'


def is_builtin(input_type: Optional[str], config: Optional[Dict]) -> bool:
  """Whether `qa run` does the runs of this type itself, as pipelines"""
  return input_type == PIPELINE_TYPE and engine(config) == 'builtin'


def has_own_inputs(config: Optional[Dict]) -> bool:
  """Projects can define the `pipeline` input type in qaboard.yaml: then its database and globs say what the pipelines' inputs are."""
  return PIPELINE_TYPE in (((config or {}).get('inputs') or {}).get('types') or {})


def default_settings(config_inputs: Dict) -> Dict:
  """The settings of the `pipeline` input type, for the projects that don't define it"""
  settings = {k: v for k, v in config_inputs.items() if k not in ('globs', 'glob', 'use_parent_folder', 'types')}
  # relative to the project: the same on every machine and platform, and it doesn't have to exist
  return {**settings, "database": ".", "type": PIPELINE_TYPE, "globs": None}


def is_default_input(run_context) -> bool:
  return run_context.type == PIPELINE_TYPE and Path(run_context.input_path).name == PIPELINE_INPUT


def is_step(configuration: Any) -> bool:
  return isinstance(configuration, dict) and ('run' in configuration or 'batch' in configuration)


def steps_of(batch: Dict, batch_name: Optional[str] = None) -> List[Dict]:
  """The `steps:` of a batch, checked. They are added to its configurations: it's how runs carry them."""
  steps = batch.get('steps')
  if steps is None:
    return []
  if isinstance(steps, dict):
    # We could support it, but the order of steps matters: tools that write YAML (yaml.dump...) sort mappings
    raise PipelineError(
      f'The steps of "{batch_name}" must be a list, each item starting with a dash:\n'
      '  steps:\n'
      '  - batch: my-batch\n'
      '  - name: my-script\n'
      '    run: python my_script.py'
    )
  if not isinstance(steps, list):
    raise PipelineError(f'The steps of "{batch_name}" must be a list of steps like "- batch: my-batch" or "- run: python my_script.py", not: {steps}')
  for index, step in enumerate(steps):
    if not is_step(step):
      raise PipelineError(f'Step {index + 1} of "{batch_name}" must have "run:" (a command) or "batch:" (a batch to run), got: {step}')
  return steps


@dataclass
class Step:
  """One step of a pipeline, as its definition says"""
  index: int                 # among the pipeline's steps, from 1
  kind: str                  # 'run' or 'batch'
  raw: Dict                  # as written
  name: str = ''
  commands: List[str] = field(default_factory=list)   # run steps: done one after the other, while they succeed
  batches: List[str] = field(default_factory=list)    # batch steps
  options: Dict[str, Any] = field(default_factory=dict)

  @property
  def allow_failure(self) -> bool:
    return bool(self.options.get('allow_failure'))

  def describe(self) -> str:
    return f"step {self.index} ({self.name})"


def _set_option(options: Dict, key: str, value: Any, step_index: int, kind: str) -> None:
  option = _OPTION_NAMES.get(key)
  if option is None:
    click.secho(f'WARNING: step {step_index}: unknown option "{key}", ignored. Options: {", ".join(STEP_OPTIONS)}', fg='yellow', err=True)
    return
  if kind == 'run' and option in BATCH_ONLY_OPTIONS:
    click.secho(f'WARNING: step {step_index}: "{key}" is an option of batch steps, ignored for a run step.', fg='yellow', err=True)
    return
  if option == 'batches_file':
    options.setdefault(option, []).extend(value if isinstance(value, list) else [value])
  else:
    options[option] = value


def parse_step(configuration: Dict, index: int) -> Step:
  """
  Reads a step. Options are keys next to run/batch:
      - batch: my-batch
        label: my-label
  or, in the first syntax, mappings in the list:
      - batch:
        - my-batch
        - label: my-label
  """
  kind = 'run' if 'run' in configuration else 'batch'
  step = Step(index=index, kind=kind, raw=configuration)
  data = configuration[kind]
  if isinstance(data, dict):
    example = '- run: python my_script.py' if kind == 'run' else '- batch: my-batch\n  label: my-label'
    raise PipelineError(f'Step {index}: "{kind}:" must be followed by a text or a list, not a mapping ({data}). Like:\n{example}')
  items = data if isinstance(data, list) else [data]
  values = [str(i) for i in items if not isinstance(i, dict) and i is not None]
  for item in items:
    if isinstance(item, dict):
      for key, value in item.items():
        _set_option(step.options, key, value, index, kind)
  for key, value in configuration.items():
    if key != kind:
      _set_option(step.options, key, value, index, kind)
  if not values:
    what = 'a command' if kind == 'run' else 'the name of a batch'
    raise PipelineError(f'Step {index}: "{kind}:" needs {what}, got: {data}')
  if kind == 'run':
    step.commands = values
  else:
    step.batches = values
  return step


def _slug(text: str) -> str:
  return re.sub(r'[^A-Za-z0-9_+-]+', '-', text).strip('-')


def _command_name(commands: List[str]) -> str:
  """`python $REPO/scripts/calibrate.py --fast` -> calibrate, `my-tool --fast` -> my-tool"""
  for command in commands:
    try:
      tokens = shlex.split(command)
    except ValueError:
      tokens = command.split()
    # VAR=value prefixes
    while tokens and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', tokens[0]):
      tokens = tokens[1:]
    if not tokens:
      continue
    program = re.split(r'[\\/]', tokens[0])[-1]
    if program.lower() in _SHELL_COMMANDS:
      continue
    script = _SCRIPT.match(program)
    if script:
      return _slug(script.group(1))
    if program.lower() not in _INTERPRETERS and not re.match(r'^python[\d.]*$', program.lower()):
      return _slug(program)
    for previous, token in zip(tokens, tokens[1:]):
      base = re.split(r'[\\/]', token)[-1]
      script = _SCRIPT.match(base)
      if script:
        return _slug(script.group(1))
      if previous == '-m': # python -m package.module
        return _slug(token.split('.')[-1])
  return ''


def parse_steps(configurations: List[Any]) -> List[Step]:
  """The steps among a pipeline's configurations, named: as asked, or after their batch or script (`step-<N>` otherwise)."""
  steps = [parse_step(c, index + 1) for index, c in enumerate(c for c in configurations if is_step(c))]
  names: Dict[str, Step] = {}
  for step in steps:
    if 'name' not in step.options:
      continue
    name = str(step.options['name'])
    if not _VALID_NAME.match(name):
      raise PipelineError(f'Step {step.index}: the name "{name}" can only have letters, digits, "-" and "_" (no spaces or dots).')
    if name in names:
      raise PipelineError(f'Steps {names[name].index} and {step.index} have the same name, "{name}": names are how later steps refer to a step.')
    step.name = name
    names[name] = step
  for step in steps:
    if step.name:
      continue
    base = _slug('+'.join(step.batches)) if step.kind == 'batch' else _command_name(step.commands)
    if not base or not _VALID_NAME.match(base):
      base = f"step-{step.index}"
    name, n = base, 1
    while name in names:
      n += 1
      name = f"{base}-{n}"
    step.name = name
    names[name] = step
  return steps


_STEP_REFERENCE = re.compile(r'\{steps(?:\.([^.\[\]{}:!]+)|\[([^\]{}]+)\])')

def _strings(value: Any):
  if isinstance(value, str):
    yield value
  elif isinstance(value, dict):
    for v in value.values():
      yield from _strings(v)
  elif isinstance(value, (list, tuple)):
    for v in value:
      yield from _strings(v)


def check_references(steps: List[Step]) -> None:
  """`{steps.<name>...}` must name an earlier step: said before anything runs, not after an hour of batches."""
  seen: List[str] = []
  all_names = [s.name for s in steps]
  for step in steps:
    for text in _strings([step.commands, step.batches, step.options]):
      for match in _STEP_REFERENCE.finditer(text):
        name = match.group(1) or match.group(2)
        if name in seen:
          continue
        if name == step.name:
          raise PipelineError(f'Step {step.index} ({step.name}) uses its own results, {{steps.{name}...}}: only earlier steps have results.')
        if name in all_names:
          raise PipelineError(f'Step {step.index} ({step.name}) uses {{steps.{name}...}}, but the step "{name}" runs after it. Move it before.')
        raise PipelineError(
          f'Step {step.index} ({step.name}) uses {{steps.{name}...}}, but no step is named "{name}". '
          f'The steps before it: {", ".join(seen) or "none"}. Give a step a name with "name: {name}".'
        )
    seen.append(step.name)


def validate(configurations: List[Any]) -> List[Step]:
  steps = parse_steps(configurations)
  check_references(steps)
  return steps


####################################################################################################
# What commands can use: {steps.<name>.output_dirs}...

def shell_quote(value: Any) -> str:
  text = str(value)
  if os.name == 'nt':
    return f'"{text}"' if re.search(r'\s', text) else text
  return shlex.quote(text)


class ShellList(list):
  """
  A list of paths that commands can use as is: `{steps.calib.output_dirs}` is the paths separated by spaces, each
  quoted if it needs to be. `{steps.calib.output_dirs:,}` separates them with commas instead.
  """
  def __format__(self, separator: str) -> str:
    return (separator or ' ').join(shell_quote(item) for item in self)

  def __str__(self) -> str:
    return format(self, '')


class _Files:
  """`{steps.calib.files[output.png]}`: the file in each output directory"""
  def __init__(self, output_dirs: List[str]):
    self.output_dirs = output_dirs

  def __getitem__(self, path) -> ShellList:
    return ShellList(str(Path(d) / str(path)) for d in self.output_dirs)


class StepResults:
  """What later steps can use from a step: `{steps.<name>.<attribute>}`"""
  def __init__(self, name: str, kind: str, runs: Optional[List[Dict]] = None, runs_file: Optional[Path] = None, status: str = 'success', label: str = ''):
    self.name = name
    self.kind = kind
    self.runs = runs or []
    self._runs_file = runs_file
    self.status = status
    self.label = label

  def _batch_only(self, attribute: str) -> None:
    if self.kind != 'batch':
      raise PipelineError(f'{{steps.{self.name}.{attribute}}}: "{self.name}" is a run step, only batch steps have runs. Its files are where your command wrote them, e.g. under {{output_dir}}.')

  @property
  def output_dirs(self) -> ShellList:
    self._batch_only('output_dirs')
    return ShellList(run['output_dir'] for run in self.runs)

  @property
  def output_dir(self) -> str:
    self._batch_only('output_dir')
    if len(self.runs) != 1:
      raise PipelineError(f'{{steps.{self.name}.output_dir}}: the step has {len(self.runs)} runs, not one. Use {{steps.{self.name}.output_dirs}} for all of them.')
    return str(self.runs[0]['output_dir'])

  @property
  def files(self) -> _Files:
    self._batch_only('files')
    return _Files([run['output_dir'] for run in self.runs])

  @property
  def input_paths(self) -> ShellList:
    self._batch_only('input_paths')
    return ShellList(run['input_path'] for run in self.runs)

  @property
  def runs_file(self) -> str:
    self._batch_only('runs_file')
    return str(self._runs_file)

  def _only(self, failed: bool) -> 'StepResults':
    self._batch_only('failed' if failed else 'ok')
    return StepResults(self.name, self.kind, [r for r in self.runs if bool(r.get('is_failed')) == failed], self._runs_file, self.status, self.label)

  @property
  def ok(self) -> 'StepResults':
    """The runs that succeeded: {steps.calib.ok.output_dirs}"""
    return self._only(failed=False)

  @property
  def failed(self) -> 'StepResults':
    return self._only(failed=True)

  def __getattr__(self, attribute: str):
    if attribute.startswith('_'):
      raise AttributeError(attribute)
    raise PipelineError(
      f'{{steps.{self.name}.{attribute}}}: steps have no "{attribute}". '
      'Use output_dirs, files[<name>], output_dir (one run), input_paths, runs_file, ok.output_dirs, label or status.'
    )

  def __str__(self) -> str:
    return self.name


class _Steps:
  """`steps` in commands: {steps.<name>...} or {steps[<name>]...}"""
  def __init__(self, later: List[str]):
    self._results: Dict[str, StepResults] = {}
    self._later = later

  def add(self, results: StepResults) -> None:
    self._results[results.name] = results

  def __getitem__(self, name) -> StepResults:
    name = str(name)
    if name in self._results:
      return self._results[name]
    if name in self._later:
      raise PipelineError(f'{{steps.{name}...}}: the step "{name}" did not run yet, or it failed before it had results.')
    raise PipelineError(f'{{steps.{name}...}}: no step is named "{name}". Steps: {", ".join(self._results) or "none yet"}.')

  def __getattr__(self, name: str) -> StepResults:
    if name.startswith('_'):
      raise AttributeError(name)
    return self[name]


_FIELD = re.compile(r'\{[^{}]+\}')
_MAX_PASSES = 20

def interpolate(text: str, params: Dict[str, Any]) -> str:
  """
  Replaces $ENVIRONMENT_VARIABLES and {fields} in a command, again and again while it changes: values can use fields too.

  Like the first implementation of pipelines for the commands it handled: the text is formatted at once (str.format).
  When that fails, because of a brace that isn't a field (JSON, awk, a typo...), it used to leave *every* field
  as is; now each `{field}` we know is replaced, and the others are left alone.
  """
  for _ in range(_MAX_PASSES):
    new_text = os.path.expandvars(text)
    try:
      new_text = new_text.format(**params)
    except PipelineError:
      raise
    except Exception:
      new_text = _FIELD.sub(lambda match: _format_field(match.group(0), params), new_text)
    if new_text == text:
      break
    text = new_text
  return text


def _format_field(field_text: str, params: Dict[str, Any]) -> str:
  try:
    return field_text.format(**params)
  except PipelineError:
    raise
  except Exception:
    return field_text


def deep_interpolate(value: Any, params: Dict[str, Any]) -> Any:
  if isinstance(value, str):
    return interpolate(value, params)
  if isinstance(value, dict):
    return {k: deep_interpolate(v, params) for k, v in value.items()}
  if isinstance(value, list):
    return [deep_interpolate(v, params) for v in value]
  return value


def context_fields(context) -> Dict[str, Any]:
  """Like context.asdict(), without copying the click context"""
  skipped = ('input_metadata', 'click_context')
  values = {f.name: getattr(context, f.name) for f in fields(context) if f.name not in skipped}
  values['rel_input_path'] = context.rel_input_path
  return values


def step_configurations(configurations: List[Any]) -> List[Any]:
  return [c for c in configurations if not is_step(c)]


####################################################################################################

def _now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path: Path, data: Any) -> None:
  """Atomic: the web app reads pipeline.json while the pipeline runs"""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  with temporary.open('w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
  os.replace(str(temporary), str(path))


def run_live(command: str, cwd=None, env=None) -> int:
  """Runs a shell command; its output shows as it comes, in the terminal and in the run's log."""
  process = subprocess.Popen(
    command,
    shell=True,
    cwd=cwd,
    env=env,
    encoding='utf-8',
    errors='replace',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
  )
  assert process.stdout
  for line in iter(process.stdout.readline, ''):
    print(line.rstrip('\r\n'), flush=True)
  return process.wait()


def _qa() -> str:
  # our own tests must run the code they test
  return f'{shell_quote(sys.executable)} -m qaboard' if 'QA_TESTING' in os.environ else 'qa'


def _run_status(output_dir: Path) -> Tuple[Optional[bool], Dict]:
  """(is_failed, metrics) of the run in the directory; (None, {}) when it didn't finish"""
  try:
    with (output_dir / 'metrics.json').open(encoding='utf-8') as f:
      metrics = json.load(f)
    return bool(metrics.get('is_failed', True)), metrics
  except (OSError, ValueError):
    return None, {}


def _latest_dir(output_dir: str) -> str:
  """Runs done again are saved in a new directory (qaboard/attempts.py): where is it now?"""
  try:
    from . import attempts
    first_dir, _ = attempts.split(Path(output_dir))
    return str(attempts.latest_dir(first_dir))
  except Exception:
    return output_dir


class Pipeline:
  def __init__(self, context, run_command: Callable = run_live):
    self.context = context
    self.run_command = run_command
    self.output_dir = Path(context.output_dir)
    self.obj = context.obj
    self.cwd = self.obj.get('previous_cwd') or os.getcwd()
    self.steps = validate(list(context.configurations))
    self.results = _Steps(later=[s.name for s in self.steps])
    self.batches_info: Dict[str, List[Dict]] = {}
    try:
      self.batches_files = [str(b) for b in json.loads(os.environ.get('QA_BATCHES_FILES') or '[]')]
    except ValueError:
      self.batches_files = []
    self.tuning_path: Optional[Path] = None
    self.metrics: Dict[str, Any] = {"is_failed": False}
    self.state: Dict[str, Any] = {
      "version": 1,
      "status": "running",
      "started_at": _now(),
      "steps": [self._initial_state(step) for step in self.steps],
    }

  def _initial_state(self, step: Step) -> Dict[str, Any]:
    state: Dict[str, Any] = {"name": step.name, "type": step.kind, "status": "pending"}
    if step.kind == 'batch':
      state["batches"] = step.batches
    else:
      state["commands"] = step.commands
    if step.allow_failure:
      state["allow_failure"] = True
    return state

  def save_state(self) -> None:
    try:
      write_json(self.output_dir / 'pipeline.json', self.state)
    except OSError as e:
      click.secho(f"WARNING: could not save pipeline.json: {e}", fg='yellow', err=True)

  def params(self, step: Optional[Step] = None) -> Dict[str, Any]:
    """
    What commands can use as {field}. Later sources win: the run's context (output_dir, batch_label...), the parameters
    from the pipeline's configurations and tuning, then the environment variables.
    """
    context = self.context
    values: Dict[str, Any] = {
      "os": os, "json": json, "re": re,
      "context": context,
      "batches_info": self.batches_info,
      "batches_files": self.batches_files,
      "tuning_path": self.tuning_path,
      "runtime_metrics": self.metrics,
    }
    if step:
      values.update(idx=step.index - 1, step=step.raw, step_name=step.name)
    values.update({k: v for k, v in self.obj.items() if isinstance(k, str)})
    values.update(context_fields(context))
    values.update({k: v for k, v in context.params.items() if isinstance(k, str)})
    values.update(os.environ)
    if 'steps' in values:
      if not getattr(self, '_warned_steps', False):
        self._warned_steps = True
        click.secho('WARNING: you have a parameter or environment variable named "steps": {steps...} is its value, not the pipeline\'s steps. Rename it to use the results of steps.', fg='yellow', err=True)
    else:
      values['steps'] = self.results
    return values

  ###### batch steps

  def child_label(self, step: Step, params: Dict) -> Tuple[str, str]:
    """(--label for qa, the label as the web app shows it)"""
    raw_label = self.obj.get('raw_batch_label') or 'default'
    shown_label = str(self.obj.get('batch_label') or raw_label)
    prefix = shown_label[:-len(raw_label)] if shown_label.endswith(raw_label) else '' # "@user| " for shared local runs
    if 'label' in step.options and step.options['label'] is not None:
      raw_label = interpolate(str(step.options['label']), params)
      if raw_label == 'CI':
        raw_label = 'default'
    return raw_label, f"{prefix}{raw_label}"

  def batch_command(self, step: Step, params: Dict) -> Tuple[str, str]:
    """(the `qa batch` command, the label of its runs in the web app)"""
    options = step.options
    raw_label, shown_label = self.child_label(step, params)
    args = [_qa()]
    if raw_label != 'default':
      args.append(f'--label "{raw_label}"')
    if 'database' in options and options['database'] is not None:
      args.append(f'--database "{interpolate(str(options["database"]), params)}"')
    if self.obj.get('share'):
      args.append('--share')
    if self.obj.get('offline'):
      args.append('--offline')
    tuning = options.get('tuning', True)
    if 'tuning' in options and tuning is None:
      click.secho(f"WARNING: {step.describe()}: empty tuning, ignored", fg='yellow', err=True)
    if tuning is not False:
      if self.tuning_path:
        args.append(f'--tuning-filepath {shell_quote(self.tuning_path)}')
      if tuning is not True and tuning is not None:
        args.append(f'--tuning {shell_quote(json.dumps(deep_interpolate(tuning, params)))}')
    configs = options.get('configs')
    if configs is not None:
      # added after the batch's configurations, like the parameters of tuning
      configs = configs if isinstance(configs, list) else [configs]
      args.append(f'--tuning {shell_quote(json.dumps(deep_interpolate(configs, params)))}')
    args.append('batch')
    if options.get('runner'):
      args.append(f'--runner="{interpolate(str(options["runner"]), params)}"')
    batches = [interpolate(b, params) for b in step.batches]
    args.extend([f'"{b}"' for b in batches] if len(batches) == 1 else [f'--batch "{b}"' for b in batches])
    # like the first implementation: a step's batches files are also known to the next steps
    self.batches_files.extend(interpolate(str(b), params) for b in options.get('batches_file', []))
    args.extend(f'--batches-file "{b}"' for b in self.batches_files)
    return ' '.join(args), shown_label

  def list_runs(self, command: str, step: Step) -> List[Dict]:
    """What the batch runs: `qa batch --list`"""
    env = dict(os.environ)
    flags = ''
    commit = step.options.get('commit')
    if commit:
      click.secho(f"Using the results of commit {commit}", fg='blue')
      # we list the results of another commit: the runs must exist
      env.update(CI='true', QA_BATCH_COMPLEX_MATCHING='true', CI_COMMIT_SHA=str(commit))
      flags = ' --action-on-existing=assert-exists'
    process = subprocess.run(f"{command}{flags} --list", shell=True, cwd=self.cwd, env=env, encoding='utf-8', errors='replace',
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
      click.secho(process.stderr, fg='red')
      raise PipelineError(f"Could not list the runs of the batch (exit code {process.returncode}): {command}{flags} --list")
    try:
      runs = json.loads(process.stdout)
    except ValueError:
      click.secho(process.stdout, fg='red')
      raise PipelineError(f"Could not read the runs of the batch: {command}{flags} --list")
    for run in runs:
      run.pop("job_options", None)
      run.pop("command", None)
    return runs

  def save_batches(self, step: Step, runs: List[Dict]) -> None:
    key = step.batches[0] if len(step.batches) == 1 else ' '.join(f'--batch "{b}"' for b in step.batches)
    self.batches_info[key] = runs
    if len(step.batches) > 1:
      for batch in step.batches:
        self.batches_info[batch] = [r for r in runs if r.get('batch') == batch]
    write_json(self.output_dir / 'batches.json', self.batches_info)

  def save_runs(self, step: Step, runs: List[Dict], cached: Optional[List[bool]] = None) -> Tuple[List[Dict], Path]:
    """steps/<name>/runs.json: the step's runs with their status and metrics"""
    detailed = []
    for index, run in enumerate(runs):
      is_failed, metrics = _run_status(Path(run['output_dir']))
      rel_input_path = run.get('rel_input_path')
      detailed.append({
        "batch": run.get('batch'),
        "input": Path(rel_input_path).as_posix() if rel_input_path else None,
        # absolute, also when the database is relative to the project
        "input_path": os.path.abspath(str(run['input_path'])) if run.get('input_path') else None,
        "database": run.get('database'),
        "type": run.get('type'),
        "platform": run.get('platform'),
        "configurations": run.get('configurations'),
        "extra_parameters": run.get('extra_parameters'),
        "output_dir": run['output_dir'],
        "is_failed": True if is_failed is None else is_failed,
        "is_cached": bool(cached[index]) if cached else False,
        "metrics": metrics,
      })
    runs_file = self.output_dir / 'steps' / step.name / 'runs.json'
    write_json(runs_file, detailed)
    return detailed, runs_file

  def do_batch(self, step: Step, state: Dict) -> bool:
    params = self.params(step)
    command, label = self.batch_command(step, params)
    state.update(label=label, command=command)
    self.save_state()
    runs = self.list_runs(command, step)
    self.save_batches(step, runs)
    returncode = 0
    reuses = bool(step.options.get('commit'))
    use_cache = bool(step.options.get('use_cache'))
    cached = [reuses or (use_cache and _run_status(Path(run['output_dir']))[0] is False) for run in runs]
    if reuses:
      state["commit"] = str(step.options['commit'])
    else:
      # We list all the runs, also those we don't run again: --action-on-existing is only for the command that runs
      command = f"{command} --action-on-existing={'skip' if use_cache else 'run'} --action-on-pending=wait"
      state["command"] = command
      click.secho(f"$ {command}", fg='cyan')
      returncode = self.run_command(command, cwd=self.cwd)
      for run in runs:
        run['output_dir'] = _latest_dir(run['output_dir'])
      self.save_batches(step, runs)
    detailed, runs_file = self.save_runs(step, runs, cached)
    self.results.add(StepResults(step.name, 'batch', detailed, runs_file, label=label))
    state.update(
      runs=len(detailed),
      failed_runs=len([r for r in detailed if r['is_failed']]),
      cached_runs=len([c for c in cached if c]),
      runs_file=runs_file.relative_to(self.output_dir).as_posix(),
    )
    if returncode != 0:
      state["returncode"] = returncode
    return returncode == 0

  ###### run steps

  def do_run(self, step: Step, state: Dict) -> bool:
    params = self.params(step)
    command = interpolate(step.commands[0] if len(step.commands) == 1 else " && ".join(step.commands), params)
    state["command"] = command
    self.save_state()
    click.secho(f"$ {command}", fg='cyan')
    if step.options.get('runner') == 'windows':
      success = self.run_on_windows(step, command, state)
    else:
      returncode = self.run_command(command, cwd=self.cwd)
      state["returncode"] = returncode
      success = returncode == 0
    self.results.add(StepResults(step.name, 'run', status='success' if success else 'failed'))
    return success

  def run_on_windows(self, step: Step, command: str, state: Dict) -> bool:
    """From linux, with the `windows` runner"""
    from .runners import JenkinsWindowsRunner
    from .compat import linux_to_windows
    run_context = deepcopy(self.context)
    run_context.command = " ".join(linux_to_windows(re.sub('["\']', '', element)) for element in command.split())
    click.secho(f"Windows command: {run_context.command}", fg='cyan')
    log_path = self.output_dir / f'log_windows_{step.index}.txt'
    state["log"] = log_path.name
    try:
      JenkinsWindowsRunner(run_context).start(log_path=log_path)
      return True
    except Exception as e:
      click.secho(f"ERROR: {e}", fg='red')
      state["error"] = str(e)
      return False

  ######

  def run(self) -> Dict[str, Any]:
    os.environ['PIPELINE_OUTPUT_DIR'] = str(self.output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    if self.context.extra_parameters:
      self.tuning_path = self.output_dir / "tuning.json"
      write_json(self.tuning_path, self.context.extra_parameters)
      os.environ['PIPELINE_TUNING_FILE'] = str(self.tuning_path)
    self.save_state()

    stopped = False
    for step, state in zip(self.steps, self.state['steps']):
      if stopped:
        state["status"] = "skipped"
        continue
      click.secho(f"\n[Step {step.index}/{len(self.steps)}] {step.name}: {step.raw}", fg='cyan', bold=True)
      os.environ['PIPELINE_STEP'] = step.name
      start = time.time()
      state.update(status="running", started_at=_now())
      try:
        success = self.do_batch(step, state) if step.kind == 'batch' else self.do_run(step, state)
      except PipelineError as e:
        click.secho(f"ERROR: {e}", fg='red', bold=True)
        state["error"] = str(e)
        success = False
      state.update(status="success" if success else "failed", duration=round(time.time() - start, 3))
      # kept from the first implementation
      self.metrics[f"step {step.index}: {step.raw}"] = "SUCCESS" if success else "FAILED"
      if not success:
        self.metrics["is_failed"] = True
        click.secho(f"ERROR: Failed step {step.index} ({step.name}): {step.raw}", fg='red', bold=True)
        if step.allow_failure:
          click.secho("       allow_failure: we continue with the next step", fg='yellow')
        else:
          stopped = True
      self.save_state()

    self.state.update(status="failed" if self.metrics["is_failed"] else "success", ended_at=_now())
    self.save_state()
    return self.metrics


def print_steps(steps: List[Step]) -> None:
  for step in steps:
    what = ', '.join(step.batches) if step.kind == 'batch' else ' && '.join(step.commands)
    click.secho(f"  {step.index}. {step.name}: {step.kind} {what}", fg='cyan')


def run_pipeline(context, run_command: Callable = run_live) -> Dict[str, Any]:
  """
  Does the steps of a pipeline, one after the other, and returns its metrics.
  `qa run` calls it for runs of type `pipeline`; projects that do it themselves (`pipelines.engine: entrypoint`)
  can call it from their run() function.
  """
  try:
    pipeline = Pipeline(context, run_command=run_command)
    if not pipeline.steps:
      raise PipelineError(
        'This pipeline has no steps. Define them like:\n'
        '  my-pipeline:\n'
        '    type: pipeline\n'
        '    steps:\n'
        '    - batch: my-batch\n'
        '    - run: python my_script.py'
      )
  except PipelineError as e:
    click.secho(f"ERROR: {e}", fg='red', bold=True)
    return {"is_failed": True}
  if context.obj.get('dryrun'):
    print_steps(pipeline.steps)
    return {"is_failed": False}
  return pipeline.run()
