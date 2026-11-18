// What people coming from other trackers ask about: where are the runs' ids, and how do I group runs? Two help
// popovers, next to what they explain: the table's Run column and the experiments' tree.
import { Button, Classes, Popover } from "@blueprintjs/core";

import { useSiteConfig } from "../hooks";

const Help = ({ label, title, children, anchor }) => {
  const { docs_root } = useSiteConfig();
  return <Popover placement="bottom-start" content={<div style={{ padding: 12, maxWidth: 380, fontSize: 13, lineHeight: 1.5 }}>
    <h5 className={Classes.HEADING} style={{ marginBottom: 6 }}>{title}</h5>
    {children}
    <a href={`${docs_root ?? '/'}docs/log-from-code#${anchor}`} target="_blank" rel="noopener noreferrer">Learn more</a>
  </div>}>
    <Button size="small" variant="minimal" icon="help" className={Classes.TEXT_MUTED} aria-label={label} style={{ minHeight: 18, minWidth: 18, padding: 1 }}/>
  </Popover>;
};

export const RunNamesHelp = () => <Help label="How runs are named" title="Runs are named by what they ran" anchor="identity">
  <p>
    A run is its experiment, its input and its configuration: what you pass to <code>start_run()</code>. There is no
    random id to keep track of. Run the same configuration on another commit, and it is the same run: QA-Board
    compares the two.
  </p>
  <p>
    To start several runs of one configuration, give each its own seed:{" "}
    <code style={{ whiteSpace: 'nowrap' }}>{'configs=[{"lr": 1e-3, "seed": 1}]'}</code>. To point someone to a run, open it and copy its link.
  </p>
</Help>;

export const ExperimentsHelp = () => <Help label="How to group runs" title="Experiments nest with slashes" anchor="experiments">
  <p>
    <code style={{ whiteSpace: 'nowrap' }}>{'experiment="detector/lr-sweep/warmup"'}</code> files the run under <strong>detector</strong> ›{" "}
    <strong>lr-sweep</strong> › <strong>warmup</strong>. Each level lists and plots every run under it: there are no
    groups or tags to set up, a new name is a new group.
  </p>
</Help>;
