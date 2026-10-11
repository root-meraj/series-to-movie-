# Season to Screen — web workspace

A responsive frontend for phones, tablets, and computers. It uses plain HTML,
CSS, and JavaScript, so Vercel needs no build step or installed packages.

The frontend connects to the real Python processing service. It does not invent
jobs or simulate progress. With no connected service, it clearly asks the user
to connect their workspace.

## Deploy the frontend on Vercel

1. Import the GitHub repository into Vercel.
2. Leave **Root Directory** empty (the repository root).
3. Choose **Framework Preset: Other**.
4. The root `vercel.json` skips installation and building and serves only `web/`.
   If entering settings manually, use empty **Build Command** and **Install
   Command**, and **Output Directory** `web`.
5. Deploy. This publishes the frontend; processing needs the service below.

An existing project whose **Root Directory** is already `web` also works:
`web/vercel.json` applies the same static settings with **Output Directory** `.`.
The configuration files override framework, command, and output settings for new
deployments. See [Vercel's configuration reference](https://vercel.com/docs/project-configuration/vercel-json).

### Fix a "No python entrypoint found" deployment

That error means the deployment detected the repository's Python package instead
of the static frontend. Pull the latest commit containing the root `vercel.json`,
then deploy that commit. In **Settings → Build and Deployment**, set **Framework
Preset** to **Other** and use one of these matching directory pairs:

| Root Directory | Output Directory |
| --- | --- |
| Empty (repository root) | `web` |
| `web` | `.` |

Save the settings and deploy the latest `main` commit. Redeploying an older failed
commit still uses that commit's files. The frontend does not require a Python
entrypoint; the processing service is deployed separately below.

To preset the processing service address, edit `web/config.js`:

```js
window.STS_CONFIG = Object.freeze({
  apiBaseUrl: "https://your-workspace.example.com"
});
```

This address is public. Never put an access key, model API key, or other secret
in `config.js`, `config.example.js`, Vercel browser variables, or the repository.
The interface also accepts a service address at runtime; it remembers only that
address on the current device. The access key stays in tab memory and is cleared
when the tab reloads or disconnects.

## Run the processing service

Deploy the Python project on a persistent server that can run FFmpeg, store
uploads and films, and access the configured NLP models. See
[`../deploy/README.md`](../deploy/README.md) for the server deployment workflow.
Vercel serves the frontend; its request handlers are not used for episode
uploads or long-running video processing.

Configure the server's private `.env`:

```dotenv
STS_WEB_ACCESS_TOKEN=replace-with-a-random-secret-of-at-least-32-characters
STS_WEB_ALLOWED_ORIGINS=https://your-frontend.vercel.app
STS_WEB_PUBLIC_URL=https://your-workspace.example.com
STS_WEB_HOST=0.0.0.0
STS_WEB_PORT=8765
STS_WEB_JOBS_DIR=data/jobs
```

Use a strong, randomly generated access token. The example token above is a
placeholder, not a credential. Keep the jobs directory on a persistent volume.
Configure `LLM_PROVIDER`, `LLM_MODEL`, Whisper, and other pipeline settings in
the same private server `.env`. The browser does not receive model credentials.

Start the service from the project directory:

```sh
python -m sts.app.web_api
```

Expose it through HTTPS and set its allowed origins to the exact frontend
origins, including any preview domain you intend to use. A hosted phone browser
cannot reach another computer using `127.0.0.1`; that address refers to the phone
itself. It needs the processing server's reachable HTTPS address.

## Use it on a phone

1. Open the Vercel website and tap **Connect workspace**.
2. Enter the processing service address and your workspace access key.
3. Open **New film**, choose a ZIP from your phone's file picker, name the film,
   and choose its length. Uploads go directly to the processing service, up to
   the limit reported by that service.
4. Keep the page open while the ZIP uploads. After upload completes, the server
   continues the edit and saves its progress even when you leave the page.
5. Open **Your films** to follow progress, stop or resume an edit, watch the
   result, or download the MP4 and SRT.
6. **Refine this film** creates a separate edit. Add episode ranges in minutes
   to guarantee a specific moment; the previous film remains available.

This is one shared editing workspace, not a multi-user account service. Anyone
with its access key can see and operate its jobs. Share it only with people who
should have that access. Public multi-tenant use requires separate accounts,
per-user storage and permissions, and appropriate server capacity.

## Local preview

From the project directory:

```sh
python -m http.server 3000 --bind 127.0.0.1 --directory web
```

Allow `http://127.0.0.1:3000` in the processing service's origins for local work.
Open that address, then connect to `http://127.0.0.1:8765` with the local server's
access key. Keep loopback HTTP for local use; online deployments require HTTPS.

## Implementation notes

- Native file picker, keyboard navigation, labels, visible focus, modal dialogs,
  and narrow-screen layouts with large touch controls.
- ZIP size validation uses `GET /api/health`; upload progress uses a native XHR
  request and sends the access key only in the Authorization header.
- Polling runs only while the selected job is processing or stopping, slows down
  in a hidden tab, and stops when the job is ready, paused, or failed.
- The movie streams from disk through the backend media endpoint. Polling
  updates status in place without replacing the video player or its position.
- Short-lived, job-scoped media URLs come from `POST /media-ticket`. The
  workspace access key never appears in media addresses or browser storage.
- Download links request attachment disposition directly from the processing
  service; large media does not pass through Vercel functions.
- No service worker caches credentials, API responses, or private media.
