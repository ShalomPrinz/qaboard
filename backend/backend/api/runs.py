"""
A project's runs across commits, by experiment: the Runs page, `qaboard.runs(project=, experiment=)`.
The queries are in backend/runs_list.py; docs/rfcs/python-sdk-runs-follow-ups.md (part A) says why.
"""
import re
import datetime

import ujson
from flask import request, jsonify, make_response, g
from sqlalchemy.orm import selectinload

from qaboard.filters import match_query, searched_text_of

from backend import app, db_session
from ..models import Output, Batch
from ..mine import identity_of, parse_origins, BATCH_SCOPES
from ..runs_list import RunsFilter, STATUSES, experiments, run_ids, experiment_of
from .auth import is_authorized_user, forbidden_project


DEFAULT_DAYS = 30
MAX_DAYS = 3650
# With a text filter, we read runs until a page is full, up to this many: the filter is the web app's, in Python
MAX_SCANNED = 5000
SCAN_CHUNK = 500


def _int(name, default, low, high):
  value = request.args.get(name)
  if value is None or value == '':
    return default
  value = int(value)
  if value < low:
    raise ValueError(f"{name} must be at least {low}")
  return min(value, high)


def runs_filter():
  """The selection from the query string, shared by /runs and /experiments. Raises ValueError, PermissionError"""
  project = request.args.get('project')
  if not project:
    raise ValueError("Which project? Add ?project=group/repo")
  scope = request.args.get('scope') or 'all'
  if scope not in BATCH_SCOPES:
    raise ValueError(f"Unknown scope {scope!r}: use {', '.join(BATCH_SCOPES)}")
  identity = None
  if scope != 'all':
    identity = identity_of(g.actor.user)
    if identity is None:
      raise PermissionError("Sign in to see your runs.")
  days = _int('since_days', DEFAULT_DAYS, 1, MAX_DAYS)
  statuses = [s.strip() for s in (request.args.get('status') or '').split(',') if s.strip()]
  unknown = [s for s in statuses if s not in STATUSES]
  if unknown:
    raise ValueError(f"Unknown status {', '.join(unknown)}: use {', '.join(STATUSES)}")
  return RunsFilter(
    project=project,
    # naive UTC, like the database's dates
    since=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(days=days),
    experiment=(request.args.get('experiment') or '').strip('/') or None,
    identity=identity,
    scope=scope,
    user=request.args.get('user') or None,
    origins=parse_origins(request.args.get('origin')),
    statuses=statuses,
    from_code=request.args.get('from_code') in ('1', 'true'),
  )


def _refused(error):
  if isinstance(error, PermissionError):
    return jsonify({"error": f"{error}", "reason": "sign_in_required"}), 401
  return jsonify({"error": f"{error}"}), 400


def _json(payload, has_more=None):
  response = make_response(ujson.dumps(payload))
  response.headers['Content-Type'] = 'application/json'
  if has_more is not None:
    response.headers['X-Has-More'] = 'true' if has_more else 'false'
    response.headers['Access-Control-Expose-Headers'] = 'X-Has-More'
  return response


@app.route("/api/v1/experiments")
@app.route("/api/v1/experiments/")
def get_experiments():
  """
  A project's experiments: its batches' labels without the "@user| " prefix of runs made outside the CI, with
  how many runs each has, how many run or failed, and when it last got a run. The web app folds them by their slashes.
  ?project= (required), ?since_days= (default 30), ?scope=mine|others|all, ?origin=webapp,local,ci, ?user=,
  ?from_code=1 (only runs started with qaboard.start_run()). X-Has-More: more experiments than we list.
  """
  try:
    selection = runs_filter()
  except (ValueError, PermissionError) as e:
    return _refused(e)
  if not is_authorized_user(None, selection.project):
    return forbidden_project(selection.project)
  rows, has_more = experiments(db_session, selection)
  return _json(rows, has_more)


def serialize_run(output):
  batch = output.batch
  commit = batch.ci_commit
  return {
    **output.to_dict(),
    'batch': {
      'id': batch.id,
      'label': batch.label,
      'experiment': experiment_of(batch.label),
      'origin': batch.origin,
      'user_name': batch.user_name,
    },
    'commit': {
      'id': commit.hexsha,
      'branch': re.sub('^origin/', '', commit.branch) if commit.branch else '',
      'message': commit.message,
      'committer_name': commit.committer_name,
      'authored_datetime': commit.authored_datetime.isoformat() if commit.authored_datetime else None,
    },
  }


def _load(ids):
  """Runs by id, in the order of the ids, with their batch and commit"""
  if not ids:
    return []
  outputs = (db_session.query(Output)
             .filter(Output.id.in_(ids))
             .options(selectinload(Output.batch).selectinload(Batch.ci_commit))
             .all())
  by_id = {o.id: o for o in outputs}
  return [serialize_run(by_id[i]) for i in ids if i in by_id]


def searched_text(run):
  """The web app's filter text of a run, with its experiment, who ran it and its commit"""
  return f"{searched_text_of(run)} {run['batch']['experiment']} {run['batch']['user_name'] or ''} {run['commit']['id']} {run['commit']['branch']}"


@app.route("/api/v1/runs")
@app.route("/api/v1/runs/")
def get_runs():
  """
  A project's runs across its commits, the running first, then the latest: with their batch (and experiment) and commit.
  Same selection as /api/v1/experiments, and:
  - ?experiment=a/b: the runs of the experiment "a/b" and of those under it ("a/b/c")
  - ?status=running,pending,failed,finished
  - ?q=: the web app's filter (qaboard/filters.py), also matching the experiment, who ran it and the commit.
  - Pages: ?limit= (default 100, at most 500) &offset=. X-Has-More says if there are more.
  """
  try:
    selection = runs_filter()
    limit = _int('limit', 100, 1, 500)
    offset = _int('offset', 0, 0, 100000)
  except (ValueError, PermissionError) as e:
    return _refused(e)
  if not is_authorized_user(None, selection.project):
    return forbidden_project(selection.project)

  q = (request.args.get('q') or '').strip()
  if not q:
    ids = run_ids(db_session, selection, limit + 1, offset)
    return _json(_load(ids[:limit]), len(ids) > limit)

  # The filter is matched in Python: we read the runs in order until we have the page, or read enough.
  # offset counts matching runs.
  matches = match_query(q)
  found, scanned, exhausted = [], 0, False
  while len(found) < offset + limit + 1 and scanned < MAX_SCANNED:
    ids = run_ids(db_session, selection, SCAN_CHUNK, scanned)
    scanned += len(ids)
    found.extend(run for run in _load(ids) if matches(searched_text(run)))
    if len(ids) < SCAN_CHUNK:
      exhausted = True
      break
  page = found[offset:offset + limit]
  response = _json(page, len(found) > offset + limit or not exhausted)
  if not exhausted and len(found) <= offset + limit:
    # we stopped reading before the page was full: the web app says the filter only looked at the latest runs
    response.headers['X-Scanned'] = str(scanned)
    response.headers['Access-Control-Expose-Headers'] = 'X-Has-More, X-Scanned'
  return response
