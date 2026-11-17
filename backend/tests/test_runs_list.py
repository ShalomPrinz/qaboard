"""
The Runs page's queries (backend/runs_list.py): experiments from batch labels, filters, order and paging.

The SQL runs only against a real PostgreSQL, in temporary tables that hide the real ones, in a transaction that is
rolled back: set QABOARD_TEST_DATABASE_URL (postgresql://user:password@host/db).
"""
import os
import re
import datetime

import pytest

from backend.mine import Identity
from backend.runs_list import RunsFilter, experiment_of, experiments_sql, experiment_rows, run_ids_sql, escape_like


NOW = datetime.datetime(2026, 11, 17, 12, 0)
WEEK_AGO = NOW - datetime.timedelta(days=7)


def selection(**kwargs):
  return RunsFilter(**{'project': 'kitefin/detector', 'since': WEEK_AGO, **kwargs})


def test_experiments_are_labels_without_the_user():
  assert experiment_of('@alice| backbone/lr-sweep') == 'backbone/lr-sweep'
  assert experiment_of('backbone/lr-sweep') == 'backbone/lr-sweep'
  assert experiment_of('@alice| default') == 'default'
  assert experiment_of(None) == 'default'
  # only the prefix qa and the SDK add
  assert experiment_of('a | b') == 'a | b'


def test_experiment_prefixes_are_escaped():
  assert escape_like('a_b%c\\') == 'a\\_b\\%c\\\\'
  where, params = selection(experiment='/backbone/').batches_where()
  assert params['experiment'] == 'backbone'
  assert params['experiment_prefix'] == 'backbone/%'


def test_scope_needs_an_identity():
  where, _ = selection(scope='mine').batches_where()
  # without identity, nothing is filtered: the endpoint refuses before
  assert not any('user_id' in w for w in where)
  where, params = selection(scope='mine', identity=Identity(user_id=3, user_name='alice')).batches_where()
  assert params['user_id'] == 3 and params['user_name'] == 'alice'
  where, _ = selection(scope='others', identity=Identity(user_id=3)).batches_where()
  assert where[-1].startswith('NOT coalesce(')


@pytest.fixture
def postgres():
  url = os.environ.get('QABOARD_TEST_DATABASE_URL')
  if not url:
    pytest.skip("QABOARD_TEST_DATABASE_URL is not set")
  psycopg2 = pytest.importorskip('psycopg2')
  connection = psycopg2.connect(url)
  try:
    cursor = connection.cursor()
    cursor.execute("""
      CREATE TEMPORARY TABLE ci_commits (id serial PRIMARY KEY, project_id text, hexsha text) ON COMMIT DROP;
      CREATE TEMPORARY TABLE batches (
        id serial PRIMARY KEY, ci_commit_id integer, label text, origin text, user_name text, user_id integer,
        latest_output_datetime timestamp, data jsonb NOT NULL DEFAULT '{}'
      ) ON COMMIT DROP;
      CREATE TEMPORARY TABLE outputs (
        id serial PRIMARY KEY, batch_id integer, output_type text, created_date timestamp, started_at timestamp,
        is_pending boolean, is_running boolean, is_failed boolean, deleted boolean
      ) ON COMMIT DROP;
    """)
    yield cursor
  finally:
    connection.rollback()
    connection.close()


def execute(cursor, sql, params):
  """Runs SQL written for sqlalchemy.text (:name) with psycopg2 (%(name)s)"""
  cursor.execute(re.sub(r'(?<![:\w]):(\w+)', r'%(\1)s', sql.replace('%', '%%')), params)
  return cursor.fetchall()


class Data:
  def __init__(self, cursor):
    self.cursor = cursor
    self.commits = {}

  def commit(self, hexsha, project='kitefin/detector'):
    if hexsha not in self.commits:
      self.cursor.execute("INSERT INTO ci_commits (project_id, hexsha) VALUES (%s, %s) RETURNING id", (project, hexsha))
      self.commits[hexsha] = self.cursor.fetchone()[0]
    return self.commits[hexsha]

  def batch(self, label, commit='a', origin='local', user_name='alice', user_id=None, latest=NOW, sdk=True, project='kitefin/detector'):
    self.cursor.execute(
      "INSERT INTO batches (ci_commit_id, label, origin, user_name, user_id, latest_output_datetime, data) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
      (self.commit(commit, project), label, origin, user_name, user_id, latest, '{"sdk": true}' if sdk else '{}'),
    )
    return self.cursor.fetchone()[0]

  def run(self, batch_id, started=NOW, status='finished', output_type=''):
    self.cursor.execute(
      "INSERT INTO outputs (batch_id, output_type, created_date, started_at, is_pending, is_running, is_failed, deleted) VALUES (%s, %s, %s, %s, %s, %s, %s, false) RETURNING id",
      (batch_id, output_type, started, started, status in ('running', 'pending'), status == 'running', status == 'failed'),
    )
    return self.cursor.fetchone()[0]


def tree(cursor, **kwargs):
  rows, has_more = experiment_rows(execute(cursor, *experiments_sql(selection(**kwargs))))
  assert not has_more
  return {row['experiment']: row for row in rows}


def ids(cursor, limit=100, offset=0, **kwargs):
  return [row[0] for row in execute(cursor, *run_ids_sql(selection(**kwargs), limit, offset))]


def test_experiments_and_runs(postgres):
  data = Data(postgres)
  hour = datetime.timedelta(hours=1)
  # Alice and Bob run the same experiment on two commits: one experiment
  alice_a = data.batch('@alice| backbone/lr-sweep', commit='a', user_id=1, latest=NOW - hour)
  bob_b = data.batch('@bob| backbone/lr-sweep', commit='b', user_name='bob', user_id=2)
  r1 = data.run(alice_a, started=NOW - 3 * hour)
  r2 = data.run(alice_a, started=NOW - 2 * hour, status='failed')
  r3 = data.run(bob_b, started=NOW - 1 * hour, status='running')
  r4 = data.run(bob_b, started=NOW - 4 * hour, status='pending')
  # qa optimize's iterations are not listed
  data.run(bob_b, output_type='optim_iteration')
  # under it, and next to it
  deeper = data.batch('@alice| backbone/lr-sweep/warmup', user_id=1)
  r5 = data.run(deeper, started=NOW - 5 * hour)
  sibling = data.batch('@alice| backbone-v2', user_id=1)
  r6 = data.run(sibling, started=NOW - 6 * hour)
  # LIKE's wildcards in names are not wildcards
  underscore = data.batch('backbone_x', origin='ci', user_name=None, sdk=False)
  r7 = data.run(underscore, started=NOW - 7 * hour)
  # too old, and another project
  old = data.batch('@alice| backbone/lr-sweep', commit='c', latest=NOW - datetime.timedelta(days=30))
  data.run(old, started=NOW - datetime.timedelta(days=30))
  other = data.batch('@alice| backbone/lr-sweep', commit='z', project='other/project')
  data.run(other)

  experiments = tree(postgres)
  assert set(experiments) == {'backbone/lr-sweep', 'backbone/lr-sweep/warmup', 'backbone-v2', 'backbone_x'}
  sweep = experiments['backbone/lr-sweep']
  assert (sweep['batches'], sweep['runs'], sweep['running'], sweep['pending'], sweep['failed']) == (2, 4, 1, 1, 1)
  assert sweep['users'] == ['alice', 'bob'] and sweep['from_code'] is True
  assert sweep['latest'] == f"{NOW.isoformat()}Z"
  assert experiments['backbone_x']['from_code'] is False

  # running first, then the latest
  assert ids(postgres) == [r3, r2, r1, r4, r5, r6, r7]
  assert ids(postgres, experiment='backbone/lr-sweep') == [r3, r2, r1, r4, r5]
  # a node has what is under it, not its siblings with the same start
  assert ids(postgres, experiment='backbone') == [r3, r2, r1, r4, r5]
  assert ids(postgres, experiment='backbone_x') == [r7]
  assert ids(postgres, limit=2, offset=1) == [r2, r1]

  # whose
  alice = Identity(user_id=1, user_name='alice')
  assert ids(postgres, scope='mine', identity=alice) == [r2, r1, r5, r6]
  assert ids(postgres, scope='others', identity=alice) == [r3, r4, r7]
  assert ids(postgres, user='bob') == [r3, r4]
  assert set(tree(postgres, scope='mine', identity=alice)) == {'backbone/lr-sweep', 'backbone/lr-sweep/warmup', 'backbone-v2'}
  # unverified runs are mine by user name
  assert ids(postgres, scope='mine', identity=Identity(user_name='bob')) == []
  # where from
  assert ids(postgres, origins=['webapp', 'local']) == [r3, r2, r1, r4, r5, r6]
  assert ids(postgres, from_code=True) == [r3, r2, r1, r4, r5, r6]
  # statuses
  assert ids(postgres, statuses=['running']) == [r3]
  assert ids(postgres, statuses=['running', 'failed']) == [r3, r2]
  assert ids(postgres, statuses=['finished']) == [r1, r5, r6, r7]
  assert ids(postgres, statuses=['pending']) == [r4]
  # the window
  assert ids(postgres, since=NOW - datetime.timedelta(hours=2, minutes=30)) == [r3, r2]


# The endpoints, without a database: what they accept, and the filter's paging
@pytest.fixture
def runs_api(dummy_app, monkeypatch):
  import backend.api.runs as api
  monkeypatch.setattr(api, 'is_authorized_user', lambda user, project: True)
  return api


def call(dummy_app, handler, url, user=None):
  import json
  from types import SimpleNamespace
  from flask import g
  with dummy_app.test_request_context(url):
    g.actor = SimpleNamespace(user=user)
    response = handler()
    if isinstance(response, tuple):
      response, status = response
    else:
      status = response.status_code
    return status, json.loads(response.get_data(as_text=True)), response.headers


@pytest.mark.parametrize('query, error', [
  ('', 'Which project?'),
  ('project=p&scope=team', 'Unknown scope'),
  ('project=p&status=done', 'Unknown status'),
  ('project=p&origin=laptop', 'Unknown origin'),
  ('project=p&since_days=0', 'since_days'),
])
def test_bad_selections_are_refused(runs_api, dummy_app, query, error):
  status, body, _ = call(dummy_app, runs_api.get_runs, f'/api/v1/runs?{query}')
  assert status == 400 and error in body['error']


def test_your_runs_need_to_sign_in(runs_api, dummy_app):
  status, body, _ = call(dummy_app, runs_api.get_experiments, '/api/v1/experiments?project=p&scope=mine')
  assert status == 401 and body['reason'] == 'sign_in_required'


def fake_runs(n):
  return [{
    'id': i, 'test_input_path': f'train/{i}', 'platform': 'linux', 'configurations': [{'lr': i}], 'extra_parameters': {},
    'data': {}, 'batch': {'experiment': 'sweep', 'user_name': 'alice'}, 'commit': {'id': 'abc', 'branch': 'main'},
  } for i in range(n)]


def test_text_filter_reads_until_the_page_is_full(runs_api, dummy_app, monkeypatch):
  everything = fake_runs(1200)
  reads = []
  def run_ids(session, selection, limit, offset):
    reads.append((limit, offset))
    return [r['id'] for r in everything[offset:offset + limit]]
  monkeypatch.setattr(runs_api, 'run_ids', run_ids)
  monkeypatch.setattr(runs_api, '_load', lambda ids: [everything[i] for i in ids])
  # lr=1x: 1, 10-19, 100-199, 1000-1199: one in a hundred in the first chunk
  status, body, headers = call(dummy_app, runs_api.get_runs, '/api/v1/runs?project=p&q=train/1&limit=5')
  assert status == 200
  assert [r['id'] for r in body] == [1, 10, 11, 12, 13]
  assert headers['X-Has-More'] == 'true' and 'X-Scanned' not in headers
  assert reads == [(runs_api.SCAN_CHUNK, 0)]
  # the experiment and who ran it match too
  status, body, headers = call(dummy_app, runs_api.get_runs, '/api/v1/runs?project=p&q=alice&limit=2&offset=1199')
  assert [r['id'] for r in body] == [1199] and headers['X-Has-More'] == 'false'


def test_text_filter_says_when_it_stopped_reading(runs_api, dummy_app, monkeypatch):
  monkeypatch.setattr(runs_api, 'MAX_SCANNED', 1000)
  everything = fake_runs(3000)
  monkeypatch.setattr(runs_api, 'run_ids', lambda session, selection, limit, offset: [r['id'] for r in everything[offset:offset + limit]])
  monkeypatch.setattr(runs_api, '_load', lambda ids: [everything[i] for i in ids])
  status, body, headers = call(dummy_app, runs_api.get_runs, '/api/v1/runs?project=p&q=nothing-matches')
  assert body == [] and headers['X-Scanned'] == '1000' and headers['X-Has-More'] == 'true'
