// The steps of a pipeline, on its run's card: what ran, in which order, and how it went.
// Pipelines are runs that do batches and commands one after the other (qaboard/pipelines.py): they save their
// progress in pipeline.json, in their output directory.
import { useQuery } from "@tanstack/react-query";
import styled from "styled-components";
import { Button, Classes, Icon, Intent, Spinner, Tag } from "@blueprintjs/core";

import { fileQuery } from "../api/queries";
import { updateSelected } from "../hooks";
import { LazyPopover } from "./LazyPopover";


const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const statuses = {
  success: { icon: 'tick-circle', intent: Intent.SUCCESS, label: 'Succeeded' },
  failed: { icon: 'error', intent: Intent.DANGER, label: 'Failed' },
  running: { label: 'Running' },
  pending: { icon: 'circle', label: 'Not started' },
  // pipelines done by the project's own code, or from before pipeline.json: we only know their definition
  unknown: { icon: 'circle', label: 'Status unknown: this pipeline doesn\'t report its steps' },
  skipped: { icon: 'disable', label: 'Skipped: an earlier step failed' },
};

// 75 -> "1m 15s"
export const duration_label = seconds => {
  if (seconds === undefined || seconds === null) return null;
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
};

// "12 runs · 2 failed · 3 reused"
export const runs_summary = step => {
  if (step.type !== 'batch' || step.runs === undefined) return null;
  return [
    plural(step.runs, 'run'),
    step.failed_runs > 0 && `${step.failed_runs} failed`,
    step.cached_runs > 0 && (step.cached_runs === step.runs ? 'all reused' : `${step.cached_runs} reused`),
  ].filter(Boolean).join(' · ');
};

// A pipeline whose run is over can't have a step still going: it was killed
const step_status = (step, is_over) => is_over && ['running', 'pending'].includes(step.status)
  ? (step.status === 'running' ? 'failed' : 'skipped')
  : step.status;


const Steps = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  margin: 6px 5px;
`;
const Arrow = styled(Icon)`
  color: var(--qa-text-muted);
  opacity: 0.6;
`;
const StepButton = styled.button`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid var(--qa-border);
  border-radius: 4px;
  background: var(--qa-surface-alt);
  color: var(--qa-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
  opacity: ${props => props.$dimmed ? 0.6 : 1};
  &:hover { background: var(--qa-hover); }
  &[data-status="failed"] { border-color: var(--qa-text-danger); }
`;
const Details = styled.div`
  max-width: 520px;
  padding: 12px;
  & pre {
    margin: 6px 0;
    max-height: 160px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-all;
    font-size: 12px;
  }
`;


const StepDetails = ({ step, status }) => {
  const summary = runs_summary(step);
  // the runs of the step: in their batch (the label), from this batch of qaboard.yaml
  const show_runs = () => updateSelected({
    selected_batch_new: step.label,
    filter_batch_new: step.batches?.length === 1 ? step.batches[0] : '',
    selected_views: 'output-list',
  });
  return <Details>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <strong>{step.name}</strong>
      <Tag minimal intent={statuses[status]?.intent}>{statuses[status]?.label ?? status}</Tag>
      {step.allow_failure && <Tag minimal>Allowed to fail</Tag>}
      {step.duration !== undefined && <span className={Classes.TEXT_MUTED}>{duration_label(step.duration)}</span>}
    </div>
    {step.type === 'batch' && <p style={{ margin: '8px 0 0' }}>
      {step.batches?.length > 1 ? 'Runs the batches ' : 'Runs the batch '}<strong>{step.batches?.join(', ')}</strong>
      {step.commit && <> (the results of commit <code>{String(step.commit).slice(0, 8)}</code>)</>}
      {summary && <>: {summary}</>}.
    </p>}
    {step.error && <p style={{ margin: '8px 0 0', color: 'var(--qa-text-danger)' }}>{step.error}</p>}
    {step.returncode !== undefined && step.returncode !== 0 && <p style={{ margin: '8px 0 0' }}>The command exited with the code {step.returncode}.</p>}
    {(step.command || step.commands) && <pre className={Classes.CODE_BLOCK}>{step.command ?? step.commands.join('\n')}</pre>}
    {status === 'unknown' && <p className={Classes.TEXT_MUTED} style={{ margin: '8px 0 0' }}>This pipeline doesn't report how its steps went.</p>}
    {step.type === 'batch' && step.label !== undefined && step.runs > 0 && <Button
      size="small"
      icon="arrow-right"
      text={`Show its ${plural(step.runs, 'run')}`}
      onClick={show_runs}
    />}
    {(status === 'failed' || status === 'unknown') && <Button
      size="small"
      variant="minimal"
      icon="console"
      text="Open the pipeline's log"
      style={{ marginLeft: 4 }}
      onClick={() => updateSelected({ selected_views: 'logs' })}
    />}
  </Details>;
};


const Step = ({ step, is_over }) => {
  const status = step_status(step, is_over);
  const { icon, intent } = statuses[status] ?? {};
  const summary = runs_summary(step);
  return <LazyPopover content={<StepDetails step={step} status={status}/>} placement="bottom-start" popupKind="dialog">
    <StepButton
      type="button"
      data-status={status}
      data-testid="pipeline-step"
      $dimmed={status === 'pending' || status === 'skipped'}
      title={statuses[status]?.label}
      aria-label={`${step.name}: ${statuses[status]?.label ?? status}${summary ? `, ${summary}` : ''}`}
    >
      {status === 'running' ? <Spinner size={14}/> : <Icon icon={icon ?? 'circle'} intent={intent} size={14}/>}
      <Icon icon={step.type === 'batch' ? 'layers' : 'console'} size={12} style={{ color: 'var(--qa-text-muted)' }} title={step.type === 'batch' ? 'Batch' : 'Command'}/>
      <span>{step.name}</span>
      {summary && <span className={Classes.TEXT_MUTED}>{summary}</span>}
      {step.duration >= 1 && <span className={Classes.TEXT_MUTED}>{duration_label(step.duration)}</span>}
    </StepButton>
  </LazyPopover>;
};


// A run step's name, from its script: `python x/calibrate.py --fast` is "calibrate" (simpler than qaboard/pipelines.py)
const script_name = commands => {
  for (const command of commands)
    for (const token of command.split(/\s+/)) {
      const match = token.split(/[\\/]/).pop().match(/^(.+)\.(py|sh|bash|m|r|pl|rb|js|exe|bat|ps1)$/i);
      if (match) return match[1].replace(/[^A-Za-z0-9_+-]+/g, '-');
    }
};

// The steps a pipeline's configurations define: `{batch: ...}` or `{run: ...}`, with their options next to them
// or, in the first syntax, as mappings in their list. `status`: what we can say without pipeline.json.
export const defined_steps = (configurations, status = 'unknown') => {
  const seen = new Set();
  return (Array.isArray(configurations) ? configurations : [])
    .filter(c => c && typeof c === 'object' && !Array.isArray(c) && ('run' in c || 'batch' in c))
    .map((c, index) => {
      const type = 'run' in c ? 'run' : 'batch';
      const items = Array.isArray(c[type]) ? c[type] : [c[type]];
      const values = items.filter(i => i !== null && typeof i !== 'object').map(String);
      const options = Object.assign({}, ...items.filter(i => i && typeof i === 'object'), c);
      const base = typeof options.name === 'string' ? options.name
        : type === 'batch' && values.length ? values.join('+').replace(/[^A-Za-z0-9_+-]+/g, '-')
        : script_name(values) ?? `step-${index + 1}`;
      let name = base;
      for (let n = 2; seen.has(name); n++) name = `${base}-${n}`;
      seen.add(name);
      return {
        name, type, status,
        ...(type === 'batch' ? { batches: values } : { commands: values }),
        ...(options.allow_failure || options['allow-failure'] ? { allow_failure: true } : {}),
      };
    });
};


// Without pipeline.json (pipelines done by the project's own code, or from before the file), the steps of the definition
export const PipelineSteps = ({ output, viewable = true }) => {
  const is_over = !output.is_pending && !output.is_running;
  // queued: no pipeline.json until its job starts
  const url = output.output_dir_url ? `${output.output_dir_url}/pipeline.json` : undefined;
  const { data } = useQuery({
    ...fileQuery(url, { is_running: !is_over }),
    enabled: viewable && !!url && !output.deleted,
    retry: false,
    // the pipeline updates the file after each step
    refetchInterval: output.is_running ? 5000 : false,
  });
  const reported = Array.isArray(data?.steps) ? data.steps : [];
  const steps = reported.length > 0 ? reported : defined_steps(output.configurations, is_over ? 'unknown' : 'pending');
  if (steps.length === 0) return null;
  return <Steps data-testid="pipeline-steps" role="group" aria-label="Pipeline steps">
    {steps.map((step, index) => <span key={index} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      {index > 0 && <Arrow icon="chevron-right" size={12}/>}
      <Step step={step} is_over={is_over}/>
    </span>)}
  </Steps>;
};
