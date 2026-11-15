// Pipelines run batches and commands one after the other (qaboard/pipelines.py). Most have no input of their own.
export const is_pipeline = output => !!output && (output.output_type === "pipeline" || output.test_input_path === "PIPELINE");

// Among a pipeline's configurations, its steps: {batch: ...} or {run: ...}. Cards show them as steps, not as settings.
const is_step = configuration => !!configuration && typeof configuration === 'object' && !Array.isArray(configuration) && ('run' in configuration || 'batch' in configuration);
export const without_steps = (output, configurations = output.configurations) => {
  if (!is_pipeline(output) || !Array.isArray(configurations) || !configurations.some(is_step)) return configurations;
  return configurations.filter(c => !is_step(c));
};

// A run's name: the label of its input's metadata, else its input (absolute when its database is /)
export const runName = output => {
  if (output.test_input_metadata?.label) return output.test_input_metadata.label
  if (is_pipeline(output)) return `${output.data?.batch} (pipeline)`
  return `${output.test_input_database === '/' ? '/' : ''}${output.test_input_path}`
}
