// The run panel's tabs besides the run's card: its files, checkpoints first, its configuration, and its links.
// What people look for after the curves: "where is the model?", "what did it run with?", "where is the rest?"
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnchorButton, Button, Callout, Classes, HTMLTable, Icon, InputGroup, NonIdealState, Spinner, Tag, Tooltip } from "@blueprintjs/core";

import { manifestQuery } from "../api/queries";
import { errorMessage } from "../api/http";
import { copyWithToast } from "../utils/clipboard";
import { folder_url } from "../utils/paths";
import { copyDirectory } from "../components/logs/DirectoryActions";
import { formatBytes } from "../storage/format";
import { useSiteConfig } from "../hooks";
import {
  configuration_json, configuration_rows, file_url, integration_links, integrations_of, load_snippet, param_text, run_files, run_links,
} from "./files";

const docs = (docs_root, page) => `${docs_root ?? '/'}docs/${page}`;
const DocsLink = ({ page, children = 'Learn more' }) => {
  const { docs_root } = useSiteConfig();
  return <a href={docs(docs_root, page)} target="_blank" rel="noopener noreferrer">{children}</a>;
};

// The run's files, from its manifest: shared with the run's card, and followed while the run goes on
export const useRunFiles = run => {
  const query = useQuery({ ...manifestQuery(run), enabled: !!run?.output_dir_url });
  return { ...query, files: query.data ? run_files(query.data) : null };
};

const row_style = { display: 'flex', alignItems: 'center', gap: 8, padding: '3px 4px', borderBottom: '1px solid var(--qa-border)', minWidth: 0 };
const path_style = { flex: '1 1 auto', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' };
const size_style = { flex: 'none', fontVariantNumeric: 'tabular-nums', fontSize: 12 };

const Checkpoint = ({ run, file, page_url }) => {
  const url = file_url(run.output_dir_url, file.path);
  return <div style={row_style} data-testid="checkpoint">
    <Icon icon="cube" size={14} className={Classes.TEXT_MUTED} style={{ flex: 'none' }}/>
    <code style={path_style} title={file.path}>{file.path}</code>
    <span className={Classes.TEXT_MUTED} style={size_style}>{formatBytes(file.size)}</span>
    <AnchorButton size="small" icon="download" text="Download" href={url} download={file.path.split('/').pop()}/>
    <Tooltip content="Copy its path on the shared storage" hoverOpenDelay={300}>
      <Button size="small" variant="minimal" icon="duplicate" aria-label={`Copy the path of ${file.path}`} onClick={() => copyDirectory(url)}/>
    </Tooltip>
    <Tooltip content="Copy the Python code that loads it" hoverOpenDelay={300}>
      <Button size="small" variant="minimal" icon="code" aria-label={`Copy the Python code that loads ${file.path}`}
        onClick={() => copyWithToast(load_snippet(page_url, file.path), { what: 'the code' })}/>
    </Tooltip>
  </div>;
};

const MAX_FILES_SHOWN = 100;

export const RunFiles = ({ run, page_url }) => {
  const { files, isPending, error, refetch } = useRunFiles(run);
  const [search, setSearch] = useState('');
  if (!run.output_dir_url) return <NonIdealState icon="folder-close" title="The run has no folder" description="Its files were not saved on the shared storage."/>;
  if (error) return <NonIdealState icon="error" title="Could not list the run's files" description={errorMessage(error)}
    action={<Button icon="refresh" text="Retry" onClick={() => refetch()}/>}/>;
  if (isPending) return <NonIdealState icon={<Spinner size={24}/>} title="Listing the run's files…"/>;
  const needle = search.trim().toLowerCase();
  const others = needle ? files.others.filter(file => file.path.toLowerCase().includes(needle)) : files.others;
  return <div>
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, marginBottom: 12 }}>
      <span><strong>{files.count}</strong> {files.count === 1 ? 'file' : 'files'}, {formatBytes(files.total_size)}</span>
      {run.is_running && <Tag minimal intent="primary" icon="refresh">follows the run</Tag>}
      <span style={{ flex: 1 }}/>
      <AnchorButton size="small" icon="folder-shared-open" text="Open the run's folder" href={folder_url(run.output_dir_url)} target="_blank" rel="noopener noreferrer"/>
      <Button size="small" icon="duplicate" text="Copy its path" onClick={() => copyDirectory(run.output_dir_url)}/>
    </div>

    <h6 className={Classes.HEADING} style={{ margin: '0 0 4px' }}>Checkpoints and models</h6>
    {files.checkpoints.length > 0
      ? <div style={{ marginBottom: 16 }}>{files.checkpoints.map(file => <Checkpoint key={file.path} run={run} file={file} page_url={page_url}/>)}</div>
      : <Callout compact icon="cube" style={{ marginBottom: 16 }} title={run.is_running ? 'No checkpoint yet' : 'No checkpoint'}>
          Save them in the run's folder, and they show here: <code>{'torch.save(state, run.output_dir / "checkpoints/last.pt")'}</code>,
          or <code>{'run.log_artifact("checkpoints/best.pt", path)'}</code>. <DocsLink page="log-from-code#artifacts"/>
        </Callout>}

    <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '0 0 4px' }}>
      <h6 className={Classes.HEADING} style={{ margin: 0 }}>Other files</h6>
      {files.others.length > 12 && <InputGroup size="small" leftIcon="search" placeholder="Find a file" value={search} onChange={e => setSearch(e.target.value)} aria-label="Find a file" style={{ width: 220 }}/>}
    </div>
    {others.slice(0, MAX_FILES_SHOWN).map(file => <div key={file.path} style={row_style}>
      <Icon icon="document" size={12} className={Classes.TEXT_MUTED} style={{ flex: 'none' }}/>
      <a href={file_url(run.output_dir_url, file.path)} target="_blank" rel="noopener noreferrer" style={path_style} className={Classes.MONOSPACE_TEXT} title={file.path}>{file.path}</a>
      <span className={Classes.TEXT_MUTED} style={size_style}>{formatBytes(file.size)}</span>
    </div>)}
    {others.length === 0 && <p className={Classes.TEXT_MUTED}>{needle ? `No file matches “${search}”.` : 'None.'}</p>}
    {others.length > MAX_FILES_SHOWN && <p className={Classes.TEXT_MUTED} style={{ marginTop: 6 }}>
      And {others.length - MAX_FILES_SHOWN} more: find them, or <a href={folder_url(run.output_dir_url)} target="_blank" rel="noopener noreferrer">open the run's folder</a>.
    </p>}
  </div>;
};

export const RunConfiguration = ({ run }) => {
  const { texts, params } = configuration_rows(run);
  if (texts.length === 0 && params.length === 0)
    return <NonIdealState icon="settings" title="No configuration" description={<span>
      Give the run its configuration when it starts, <code>{'start_run(configs=[{"lr": 1e-3}])'}</code>: it shows here, and tells the runs apart. <DocsLink page="log-from-code#identity"/>
    </span>}/>;
  return <div>
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, marginBottom: 10 }}>
      <span className={Classes.TEXT_MUTED}>What it ran with: its <code>configs</code>, which name the run, then what it <strong>logged</strong> with <code>run.log_params()</code>.</span>
      <span style={{ flex: 1 }}/>
      <Button size="small" icon="duplicate" text="Copy as JSON" disabled={params.length === 0} onClick={() => copyWithToast(configuration_json(run), { what: 'the configuration' })}/>
    </div>
    {texts.length > 0 && <p style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {texts.map(text => <Tag key={text} minimal icon="document">{text}</Tag>)}
    </p>}
    {params.length > 0 && <HTMLTable compact striped style={{ width: '100%', tableLayout: 'fixed' }} aria-label="The run's parameters">
      <tbody>
        {params.map(([key, value, source]) => <tr key={key}>
          <td style={{ width: '35%', overflowWrap: 'anywhere' }}><code>{key}</code></td>
          <td style={{ overflowWrap: 'anywhere' }} className={Classes.MONOSPACE_TEXT}>{param_text(value)}</td>
          <td style={{ width: 90, textAlign: 'right' }}>
            {source === 'logged' && <Tooltip content="Reported by run.log_params(): it doesn't tell runs apart" hoverOpenDelay={300}>
              <Tag minimal className={Classes.TEXT_MUTED}>logged</Tag>
            </Tooltip>}
          </td>
        </tr>)}
      </tbody>
    </HTMLTable>}
  </div>;
};

const LinkButton = ({ link }) => <AnchorButton icon={link.icon ?? 'link'} endIcon="share" text={link.text} href={link.href} target="_blank" rel="noopener noreferrer"/>;

// The links: the run's own, then its commit's. The more a project sets up, the less people have to search.
export const useRunLinks = ({ run, commit, batch, project, project_data, user }) => {
  const own = run_links(run);
  const integrations = integration_links({ integrations: integrations_of({ batch, commit, project_data }), project, project_data, commit, batch, user });
  return { own, integrations, count: own.length + integrations.length };
};

export const RunLinks = ({ run, commit, links }) => {
  return <div>
    <h6 className={Classes.HEADING} style={{ margin: '0 0 6px' }}>From the run</h6>
    {links.own.length > 0
      ? <p style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>{links.own.map((link, index) => <LinkButton key={index} link={link}/>)}</p>
      : <Callout compact icon="link" style={{ marginBottom: 16 }}>
          Link the run to its model card, its dataset, its training dashboard: <code>{'run.log_link("Model card", url)'}</code>. <DocsLink page="log-from-code#links"/>
        </Callout>}
    <h6 className={Classes.HEADING} style={{ margin: '16px 0 6px' }}>From its commit</h6>
    <p style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
      {links.integrations.map((link, index) => <LinkButton key={index} link={link}/>)}
      {commit?.artifacts_url && <>
        <Tooltip content="What the commit's CI saved with qa save-artifacts: its build, models, configurations. Empty when nothing was saved." hoverOpenDelay={300}>
          <AnchorButton icon="folder-shared-open" text="The commit's artifacts" href={folder_url(commit.artifacts_url)} target="_blank" rel="noopener noreferrer"/>
        </Tooltip>
        <Tooltip content="Copy the path of the commit's artifacts" hoverOpenDelay={300}>
          <Button variant="minimal" icon="duplicate" aria-label="Copy the path of the commit's artifacts" onClick={() => copyDirectory(commit.artifacts_url)}/>
        </Tooltip>
      </>}
      {!commit && <Spinner size={16}/>}
    </p>
    {commit && links.integrations.length === 0 && <p className={Classes.TEXT_MUTED}>
      The links of the project's Integrations menu show here too, filled for the run's commit. <DocsLink page="triggering-third-party-tools">Add links in qaboard.yaml</DocsLink>
    </p>}
    {run.batch?.origin !== 'ci' && <p className={Classes.TEXT_MUTED} style={{ fontSize: 12 }}>
      Runs started from code save their own files: the commit's artifacts are what its CI saved, if it does.
    </p>}
  </div>;
};
