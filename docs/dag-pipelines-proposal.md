# Proposal: pipelines (DAGs) of batches

Status: **partly implemented**. Pipelines shipped as an ordered list of steps, batches and commands
(`qaboard/pipelines.py`, user docs in `website/docs/pipelines.mdx`): it's the "user-side" implementation mentioned
below, upstreamed, with named steps and `{steps.<name>...}` in commands. What is left of this proposal is the graph:
steps that say what they need, to run independent steps in parallel. See "What shipped, and what's next" at the end.
Discussion: [issue #10](https://github.com/Samsung/qaboard/issues/10).

## The problem
QA-Board lacks expressivity for a common use-case:
1. Run an algorithm on some images,
2. calibrate from those results,
3. validate the calibration on other images.

Likewise, it doesn't handle machine learning workflows (training, then validation) nicely.

**The goal is a solution that is simple, expressive, and enables caching.** Feedback from people who used flow
engines (e.g. DVC) is welcome.

## Workarounds used today
1. **Run `qa batch` several times**, each batch expecting the results of the previous one to be available.
   - (+) Rather easy to do.
   - (-) The logic is outside QA-Board: it can't easily be used for tuning from the web app.
2. **A "meta" run**, whose heavy `run()` function takes care of everything.
   - (+) Easy to do.
   - (-) Very custom, and not easy to use.
   - (-) Many tricky corners (running locally, filesystem issues on LSF) that should not be the engineer's concern.

## Proposal: built-in support with `needs:`
Extend the syntax used to define batches with a `needs:` keyword.

Simple example:
```yaml
batch1:
  inputs:
  - A.jpg
  - B.jpg
  configurations:
  - base

batch2:
  needs: batch1
  type: script
  configurations:
  - python my_script.py {o.output_dir for o in depends["batch1"]}

# ? there is not really an "input" for a script, especially if it uses "depends"
# ? if we depend on something, we could do without inputs, and provide None...
```

"Real-world" example:
```yaml
my-calibration-images:
    configurations:
    - base
    inputs:
    - DL50.raw
    - DL55.raw
    - DL65.raw
    - DL75.raw

my-calibration:
    needs:
      calibration_images: my-calibration-images
    type: script
    configurations:
    - python calibration.py ${o.output_directory for o in depends[calibration_images]}

my-evaluation-batch:
    needs:
      calibration: my-calibration
    inputs:
    - test_image_1.raw
    - test_image_2.raw
    - test_image_3.raw
    configurations:
    - base
    - ${depends[calibration].output_directory}/calibration.yaml
```

```shell
$ qa batch my-evaluation-batch
#=> qa batch my-calibration-images
#=> qa batch my-calibration
#=> qa batch my-evaluation-batch
```

- (+) Simple for users (to be confirmed).
- (+) Caching for free.
- (?) Needs a clear API: how each job looks up the results of earlier jobs. Parameter tuning will likely need
  naming conventions too.


## What shipped, and what's next
Pipelines are a batch of `type: pipeline` whose run does `steps:` in order:

```yaml
my-calibration:
  type: pipeline
  steps:
  - batch: my-calibration-images
  - name: calibrate
    run: python calibration.py {steps.my-calibration-images.output_dirs} --output {output_dir}/calibration.yaml
  - batch: my-evaluation-batch
    configs: ["{output_dir}/calibration.yaml"]
```

Compared with `needs:` on batches above:
- The pipeline is the container: batches stay what they are, and can be run alone.
- Commands get results as arguments (`{steps.<name>.output_dirs}`, `.files[...]`, `.runs_file`), with plain
  `str.format` fields, not a new expression language.
- Caching is explicit, per step (`use-cache: true`): it reuses `qa batch --action-on-existing=skip`.

Not done, by order of value:
1. **`qa batch my-pipeline --from <step>` / `--only <step>`**, to work on a late step without editing the YAML. It
   needs the pipeline's directory to survive a rerun (today `qa run` cleans it, or uses a new one), so that the files
   of earlier commands are still there.
2. **`needs:` between steps**, defaulting to the previous step: steps are named, so it is a compatible extension, and
   `graphlib` sorts them. The gain is small (the runs of a batch already run in parallel); the work is running
   independent steps together and showing a graph in the web app instead of a line.
3. **Caching commands**: knowing that a script and its inputs didn't change. It's where DVC-like tools get
   complicated; out of scope until someone needs it.
4. **Steps as runs of their own** in the database, to compare a step with the reference's and rerun one from the web
   app. Today the web app reads the steps from the pipeline's `pipeline.json`.
