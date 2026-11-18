// What the run panel shows besides the run's card (RunDetails.jsx): its files with its checkpoints first, its
// configuration as rows, and links: the run's own, its commit's artifacts, the project's integrations.
import { make_eval_templates_recursively } from "../utils";
import { is_hidden } from "../configurations/tokens";

// Saved models, whatever the framework: PyTorch, Lightning, safetensors, ONNX, Keras/TensorFlow, TensorRT, JAX, GGUF...
const MODEL_EXTENSIONS = new Set([
  'pt', 'pth', 'ckpt', 'safetensors', 'onnx', 'h5', 'hdf5', 'keras', 'pb', 'tflite', 'engine', 'plan', 'gguf',
  'joblib', 'msgpack', 'pdparams', 'mlmodel', 'mar', 'torchscript', 'trt',
]);
// ...and files in folders named like checkpoints: checkpoints/epoch-3/config.json is part of a checkpoint
const MODEL_FOLDERS = /(^|\/)(checkpoints?|ckpts?|models?|weights)(\/|$)/i;
// names that say what generic files hold: model.bin, best_model.pkl
const MODEL_NAMES = /(^|[/_.-])(model|checkpoint|ckpt|weights)[^/]*\.(bin|pkl|pickle|npz)$/i;

// The files QA-Board writes for itself: the list of files itself
const LISTINGS = new Set(['manifest.outputs.json']);

export const is_checkpoint = path => {
  const extension = path.split('.').pop()?.toLowerCase();
  return MODEL_EXTENSIONS.has(extension) || MODEL_FOLDERS.test(path.split('/').slice(0, -1).join('/')) || MODEL_NAMES.test(path);
};

// The checkpoints people want first: the best, the last, the exported models, then the latest epochs or steps
const rank = path => {
  const name = path.split('/').pop().toLowerCase();
  if (/best/.test(name)) return 0;
  if (/last|latest|final/.test(name)) return 1;
  if (!MODEL_FOLDERS.test(path.split('/').slice(0, -1).join('/'))) return 2;
  return 3;
};
const by_usefulness = (a, b) => (rank(a.path) - rank(b.path)) || b.path.localeCompare(a.path, undefined, { numeric: true });

// The manifest's files ({path: {st_size, md5}}) -> the checkpoints, the other files, the total size
export const run_files = manifest => {
  const files = Object.entries(manifest ?? {})
    .filter(([path]) => !LISTINGS.has(path))
    .map(([path, info]) => ({ path, size: info?.st_size ?? null }));
  const checkpoints = files.filter(file => is_checkpoint(file.path)).sort(by_usefulness);
  const others = files.filter(file => !is_checkpoint(file.path))
    .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
  const total_size = files.reduce((total, file) => total + (file.size ?? 0), 0);
  return { checkpoints, others, total_size, count: files.length };
};

// The URL of a file of the run, from its folder's: each part of the path is encoded, not the slashes
export const file_url = (output_dir_url, path) => `${output_dir_url}/${path.split('/').map(encodeURIComponent).join('/')}`;

// How to get a file from Python, from the address of this page, which opens the run
export const load_snippet = (page_url, path) => {
  const py_string = text => JSON.stringify(text);
  const lines = [
    'import qaboard as qb',
    '',
    `run = qb.runs(${py_string(page_url)})[0]`,
    `path = run.output_dir / ${py_string(path)}`,
  ];
  if (/\.(pt|pth|ckpt)$/i.test(path)) lines.push('state = torch.load(path, map_location="cpu")');
  else if (/\.safetensors$/i.test(path)) lines.push('state = safetensors.torch.load_file(path)');
  return lines.join('\n');
};

// The run's configuration, as rows. The configurations' texts (files like configs/detector.yaml) come first, then
// the parameters, nested ones as "a.b": what configs= chose, then what run.log_params() reported.
const flatten = (value, prefix, rows) => {
  if (value !== null && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length > 0) {
    for (const [key, child] of Object.entries(value)) flatten(child, prefix ? `${prefix}.${key}` : key, rows);
  } else {
    rows.push([prefix, value]);
  }
  return rows;
};
// Each row says whether the run logged it (run.log_params()): those don't tell runs apart, configs= does.
export const configuration_rows = run => {
  const texts = (run?.configurations ?? []).filter(c => typeof c === 'string');
  const params = Object.fromEntries(Object.entries(run?.params ?? {}).filter(([key, value]) => !is_hidden(key, value)));
  const chosen = new Set();
  for (const c of run?.configurations ?? [])
    if (c && typeof c === 'object') for (const key of Object.keys(c)) chosen.add(key);
  const logged = new Set(Object.keys(run?.data?.params ?? {}).filter(key => !chosen.has(key)));
  const rows = flatten(params, '', []).filter(([key]) => key !== '')
    .map(([key, value]) => [key, value, logged.has(key.split('.')[0]) ? 'logged' : 'configs']);
  return { texts, params: rows };
};

export const param_text = value => typeof value === 'string' ? value : JSON.stringify(value);

// What the run's configuration is as JSON: what to paste in a config file, or in configs= to run it again
export const configuration_json = run => {
  const params = Object.fromEntries(Object.entries(run?.params ?? {}).filter(([key, value]) => !is_hidden(key, value)));
  return JSON.stringify(params, null, 2);
};

// The run's own links: what it logged with run.log_link(), or what run() returned as params.badges
export const run_links = run => (run?.params?.badges ?? [])
  .filter(badge => badge && typeof badge === 'object' && badge.href)
  .map(badge => ({ text: badge.text ?? badge.href, href: badge.href, icon: badge.icon }));

// The links of the project's Integrations menu (`integrations` in qaboard.yaml), for this run's commit and batch.
// Only links: webhooks and CI jobs act on commits, they stay in the commit page's menu. Like the menu, an item whose
// templates need something the run doesn't have (`${branch}`, `only: ...`) is left out.
const ACTIONS = ['webhook', 'gitlabCI', 'githubActions', 'jenkins'];
export const integration_links = ({ integrations, project, project_data, commit, batch, user }) => {
  if (!Array.isArray(integrations) || integrations.length === 0 || !commit) return [];
  const evaluate = make_eval_templates_recursively({ project, project_data, commit, batch, user });
  const links = [];
  const visit = items => {
    for (const item of items ?? []) {
      if (!item || typeof item !== 'object' || item.divider) continue;
      // a sub-menu's items are filled one by one: one that can't be doesn't hide the others
      const { sub, ...own } = item;
      let filled;
      try {
        filled = evaluate(own);
      } catch {
        continue;
      }
      if (!filled) continue;
      if (Array.isArray(sub)) visit(sub);
      if (!filled.href || ACTIONS.some(action => filled[action])) continue;
      links.push({ text: filled.text ?? filled.alt ?? filled.href, href: filled.href, icon: filled.icon });
    }
  };
  visit(integrations);
  return links;
};

// The integrations of the run's batch's qaboard.yaml, else its commit's, else the project's (like the commit page)
export const integrations_of = ({ batch, commit, project_data }) =>
  batch?.data?.qatools_config?.integrations ?? commit?.data?.qatools_config?.integrations ?? project_data?.data?.qatools_config?.integrations ?? [];
