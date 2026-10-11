# Run the processing service

The phone-friendly website in `web/` can run on Vercel. Its ZIP uploads and film
playback connect directly to this persistent Python service. Vercel Functions
have request-size and execution-duration limits that are unsuitable for the
season-processing worker. See [Vercel's function limits](https://vercel.com/docs/functions/limitations).

Run the service on a machine that stays online, with persistent storage and an
HTTPS address. It runs one edit at a time. The access token opens one shared
workspace, including its projects and downloads; separate user accounts are not
implemented.

## Backend configuration

Copy `.env.example` to `.env` **on the processing server**, then configure:

```dotenv
STS_WEB_ACCESS_TOKEN=
STS_WEB_ALLOWED_ORIGINS=https://your-site.vercel.app
STS_WEB_PUBLIC_URL=https://processing.example.com
STS_WEB_JOBS_DIR=/app/data/jobs
LLM_PROVIDER=ollama
LLM_MODEL=your-installed-ollama-model
OLLAMA_BASE_URL=http://your-private-ollama-service:11434
TRANSCRIPT_SOURCE=auto
SCENE_METHOD=subtitles
EXCLUDE_CREDITS=true
SELECTION_STRATEGY=story
```

Replace the sample addresses and model name with your real settings. Set
`STS_WEB_ACCESS_TOKEN` to a randomly generated value of at least 32 characters
without spaces. Store it only in `.env`; the service rejects a missing token and
does not read this secret from process environment variables. Give the key only
to people who should be able to access every project in this workspace. They
enter it in the website's connection screen. Never put the key in
`web/config.js`, a Vercel frontend variable, or the repository.

List the exact frontend origins separated by commas in
`STS_WEB_ALLOWED_ORIGINS`; a wildcard is rejected. Add a custom domain or a
specific preview origin when it needs access. Set `STS_WEB_PUBLIC_URL` to the
public HTTPS backend origin so generated movie URLs work behind your proxy.

Ollama runs separately and must already have the configured model downloaded.
No paid API key is needed for that option. An endpoint inside a container must
be reachable **from that container**: `localhost:11434` refers to the container
itself, not the host laptop. Use a private container-network service name or a
reachable private host address. Keep the Ollama endpoint private. For Docker
Desktop, `host.docker.internal` can reach a host service if it listens on a
reachable interface; Linux hosts may require Docker's `host-gateway` mapping.
See [Ollama's Docker setup](https://docs.ollama.com/docker).

Alternatively, set `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, and
`ENABLE_OPENAI=true`, with `OPENAI_API_KEY` stored in the same backend `.env`.
`LLM_PROVIDER=mock` is for deterministic checks and does not provide real story
understanding.

## Docker

From the repository root, build the CPU image:

```sh
docker build -f deploy/Dockerfile -t season-to-screen-backend .
```

The image installs FFmpeg/FFprobe, the project's Python requirements, and the
spaCy English model during setup. It installs CPU PyTorch from its official
wheel index; see [PyTorch's installation guide](https://pytorch.org/get-started/locally/).
Whisper and sentence-transformer models download when first needed and remain
in the `/models` volume. Building and first-time model loading require network
access. The image uses the repository's version ranges; preserve a tested image
tag or digest when deploying an update.

Start it on the backend host, keeping the published port on loopback for your
HTTPS reverse proxy:

```sh
docker run -d --name season-to-screen-backend --init \
  --restart unless-stopped --stop-timeout 120 \
  -p 127.0.0.1:8765:8765 \
  --mount "type=bind,source=/absolute/path/to/.env,target=/app/.env,readonly" \
  --mount type=volume,source=sts-data,target=/app/data \
  --mount type=volume,source=sts-models,target=/models \
  season-to-screen-backend
```

Replace `/absolute/path/to/.env` with the existing file on that Docker host.
The file must be readable by the container's user (UID 10001). The image keeps
data and models writable by that user. Named volumes survive container
replacement; keep both volumes when updating. If you use host directories
instead, give that user write access. See [Docker's bind-mount documentation](https://docs.docker.com/engine/storage/bind-mounts/).

The CPU image sets `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE=int8`, and
`RENDER_ENCODER=libx264`. It does not include or automatically install GPU
support. Its API uses one process and one pipeline worker. Do not start multiple
replicas or additional API workers against the same jobs volume: there is no
distributed worker lock. Run either this backend or Streamlit for a given jobs
directory, so two runners cannot start the same edit concurrently.

Size storage for the uploaded ZIPs, extracted episodes, audio, checkpoints,
rendered clips, and finished films. Uploads are spooled to temporary disk before
being saved, so allow room for that additional temporary copy. Jobs retain their
input ZIP at `data/jobs/<job>/input/source.zip`; keep it at least until ingestion
finishes. Keep each saved job directory together when backing up or moving it.
Restore it under the same container paths: checkpoints include source paths.
This deployment creates a server workspace; GitHub and the Docker build do not
include existing laptop episodes or finished films.
An interrupted job stays saved; reconnect and use **Resume edit** after restart.

For a non-Docker installation, use Python 3.10 or newer, install FFmpeg and
FFprobe on the host, then run from the repository root:

```sh
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m sts.app.web_api
```

The native service binds to `127.0.0.1:8765` by default. Set `STS_WEB_HOST` and
`STS_WEB_PORT` for your hosting platform if its private proxy requires another
binding. Keep `.env` in the process working directory.

## HTTPS and large uploads

Point your backend HTTPS reverse proxy at port 8765. Forward `/api/` paths
unchanged; do not route uploaded ZIPs through a Vercel function. The API accepts
ZIP archives up to **2 GiB**. Configure the proxy's request-body limit slightly
above that to allow multipart overhead, and its upload/read timeout for a large
upload over a phone connection. A CDN in front of the backend may impose a lower
upload limit; use a direct HTTPS backend origin that accepts the chosen size.

Allow `OPTIONS` requests and forward `Authorization`, `Content-Type`, and
`Range`. Preserve status codes and `Content-Range`/`Accept-Ranges` headers for
video seeking. Avoid buffering full multi-gigabyte exports in a proxy. Media
links contain scoped access tickets, so avoid logging their query strings.
Set proxy logging accordingly, and maintain `Cache-Control: private, no-store`.

Check readiness at `https://processing.example.com/api/health`, then set the
public backend address in `web/config.js` and redeploy the Vercel frontend.
Health readiness confirms that the API is serving; it does not validate your
Ollama model or pre-download the speech/embedding models. Test a small synthetic
project before processing a full season. The protected workspace APIs require
the access key, and export playback uses short-lived media tickets.

## Vercel frontend

Import the GitHub repository into Vercel and leave **Root Directory** empty; the
root `vercel.json` serves only `web/`. Existing projects rooted at `web` are also
supported by `web/vercel.json`. Use the static site's deployment settings documented in
[web/README.md](../web/README.md). Deploying the frontend alone provides the
interface; upload and processing become available after its persistent backend
is deployed and connected.
