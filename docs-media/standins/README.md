# Stand-ins
Pictures shown in the docs until a planned shot is captured from the demo stack: see "Stand-ins" in
[`../README.md`](../README.md). A shot's `standin:` key in `../shots.yaml` names its files here.

`make.py` makes the performance viewers' stand-ins (`commit/benchmarks`, `commit/flame-top-functions`): run it again
when their viewer changes, then update the sizes and the numbers of `alt` and `caption` in `../shots.yaml`.

`pipelines.py` makes the pipelines' (`commit/pipeline`, `commit/pipeline-step`, `commit/pipeline-running`,
`commit/pipeline-failed`): it runs the camera ISP example on its charts, then takes pictures of the web app's production
build with the API mocked (`webapp/e2e/standins/pipelines.standin.js`, run with `STANDINS=1`). After `npm run build` in
`webapp/`: `uv run docs-media/standins/pipelines.py`.

When a shot is published, delete its files and its `standin:` key in the same pull request. Keep this folder, even
empty: the docs image copies it (`website/Dockerfile`).
