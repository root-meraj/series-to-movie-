# Azure student backend: first private demo

Keep the website on Vercel and run this Docker deployment on one Ubuntu server.
It includes the existing processing API, a private Ollama service, and Caddy for
HTTPS. Persistent volumes keep jobs, downloaded models, and certificates across
container restarts. This is a shared workspace for trusted testers. Separate
accounts, ownership checks, a public job queue, and automatic storage cleanup
are still required before accepting arbitrary public users.

## 1. Review the server in Azure

Open [Azure Portal](https://portal.azure.com), search for **Virtual machines**,
then choose **Create → Azure virtual machine**. Configure:

| Setting | Starting value |
| --- | --- |
| Subscription | Azure for Students; keep its spending limit |
| Resource group | `season-to-screen-demo` |
| VM name | `sts-processor` |
| Region | A region allowed by the student subscription |
| Image | Ubuntu Server 24.04 LTS, x64, Gen2, without paid add-ons |
| Size | `Standard_B2as_v2`: 2 vCPU, 8 GiB RAM; review the displayed price |
| Authentication | SSH public key; username `azureuser` |
| OS disk | 128 GiB Standard SSD, persistent; review the displayed price |
| Availability | One instance; no extra redundancy for this demo |
| Networking | Standard public IP; TCP 80 and 443; SSH 22 restricted to your IP |

If B2as_v2 is unavailable, B2s_v2 has the same CPU and memory count, but review
its actual price and eligibility before selecting it. B2ats_v2 has only 1 GiB
RAM and is unsuitable for this stack. Do not select the similarly named ARM
image for this first deployment; the existing image was tested on x86 Linux.

Use **Review + create** to inspect the complete configuration and estimate
before creating the VM. The 8 GiB size is a starting point for a small demo;
full-season speed and memory use still need benchmarking on the actual server.
B-series CPU performance drops to its baseline after burst credits run out.
The existing Qwen3 provider requests a 32K context. Even a small model can use
substantial memory at that context size; this bundle does not impose a small
container memory cap that would force it to fail. Monitor real use on the VM.
Ollama's configured cache quantization depends on runtime support and does not
prove that an entire season fits in 8 GiB.
See [Azure's size specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/basv2-series)
and [VM creation guide](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal).

Student credit is a finite balance, not a year of unlimited compute. VM runtime,
managed disks, public IPs, and applicable transfer charges consume it. Keep the
student spending limit enabled and do not upgrade to Pay As You Go. Configure a
small monthly budget alert in **Cost Management**; alerts do not stop resources.
See the [student offer](https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0170p/)
and [budget behavior](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets).

## 2. Give the backend an HTTPS address

On the VM's public IP resource, open **Configuration** and set an available
**DNS name label**, for example a unique label based on `sts-processor`.
Copy the resulting full hostname, such as `your-label.centralindia.cloudapp.azure.com`.
The example is a placeholder: use the hostname Azure actually assigns. This
avoids purchasing a domain. The public IP itself can still consume credit.
See [Azure public IP DNS labels](https://learn.microsoft.com/en-us/azure/virtual-network/ip-services/public-ip-addresses).

Confirm that this hostname resolves to the VM's public IP, and that Azure's
network rules allow TCP 80 and 443. Keep TCP 8765 and 11434 closed to the public.
Restrict SSH to your own public IP. Caddy obtains and renews the TLS certificate
after the hostname is reachable; it stores certificate data in its own volume.
Do not place a CDN with a smaller upload limit in front of this hostname.

## 3. Install Docker and get the project

Use the VM's **Connect → Native SSH** instructions to connect from your laptop.
Keep the private SSH key on your laptop. Install Docker Engine and the Compose
plugin using [Docker's official Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/).
Use its apt-repository installation steps. Then run on the VM:

```sh
sudo apt-get update
sudo apt-get install -y git python3
git clone https://github.com/root-meraj/series-to-movie-.git
cd series-to-movie-
cp deploy/azure/.env.example deploy/azure/.env
```

Generate the workspace key directly into the private file, without displaying it:

```sh
python3 - <<'PY'
from pathlib import Path
import secrets

path = Path("deploy/azure/.env")
text = path.read_text(encoding="utf-8")
text = text.replace("STS_WEB_ACCESS_TOKEN=", "STS_WEB_ACCESS_TOKEN=" + secrets.token_urlsafe(48), 1)
path.write_text(text, encoding="utf-8")
PY
```

Edit `deploy/azure/.env` on the server. Set `STS_BACKEND_DOMAIN` to the full Azure
DNS hostname (no `https://`), and `STS_WEB_ALLOWED_ORIGINS` to your exact Vercel
site origin (including `https://`, with no path or trailing slash). The remaining
settings configure subtitles first, credit removal, story selection, and the
small local `qwen3:1.7b` model. No paid model API calls are enabled.

For the current website, the allowed origin is
`https://series-to-movie-854n.vercel.app`. It is already set in the Azure example;
change it if the Vercel domain changes.

The key must stay private. It grants access to every job in this demo workspace.
Never place it in frontend files, Vercel browser variables, screenshots, or Git.
The backend reads it from the mounted `.env` file; an environment variable alone
does not satisfy the backend's authentication configuration.

After editing, grant only the backend user and root access to the file:

```sh
sudo chown 10001:10001 deploy/azure/.env
sudo chmod 0400 deploy/azure/.env
```

Use `sudo` for the Compose commands below so Compose can read this file. Use
`sudoedit deploy/azure/.env` for later changes. Do not regenerate the key or
copy the example over the file during routine updates.

## 4. Start the services and install the local model

From the repository root on the VM:

```sh
sudo docker compose --env-file deploy/azure/.env -f deploy/azure/compose.yaml config --quiet
sudo docker compose --env-file deploy/azure/.env -f deploy/azure/compose.yaml up -d --build
sudo docker compose --env-file deploy/azure/.env -f deploy/azure/compose.yaml exec ollama ollama pull qwen3:1.7b
sudo docker compose --env-file deploy/azure/.env -f deploy/azure/compose.yaml ps
```

The first build installs the Python stack and can take time on a small VM.
Ollama downloads the model separately. Speech and embedding models download
when first needed; their cache is retained. Existing laptop projects and films
are not uploaded by cloning or building the server.

Open `https://YOUR_AZURE_HOSTNAME/api/health` and confirm the API responds.
Health checks verify the API and Ollama server, not that every model is ready or
that a real film has rendered. Do not start a job before the Ollama model pull
has completed. If the proxy has no certificate, inspect DNS and inbound ports
before retrying; do not bypass certificate warnings.

## 5. Connect Vercel and test one small film

Open the existing Vercel site and tap **Connect workspace**. Enter the backend
address `https://YOUR_AZURE_HOSTNAME` and the generated private workspace key.
This connects the site without rebuilding it. The website remembers only the
address; the key remains in tab memory and must be entered after a reload.

For a preset backend address, set only `apiBaseUrl` in `web/config.js` to that
HTTPS address and redeploy the Vercel frontend. Keep the key private.

Test a small synthetic ZIP first. Verify upload, progress, stop/resume, MP4
playback with seeking, and MP4/SRT downloads before trying a full season.
The application accepts ZIPs up to 2 GiB; the proxy permits multipart overhead.
Keep the phone page open while uploading. There is one edit worker, with no
public waiting queue: another concurrent edit can be rejected as busy.

Monitor disk space and memory. Files remain on the server until deliberately
cleaned up; this deployment does not add automatic retention or user isolation.
Do not distribute the workspace key to arbitrary visitors.

## 6. Preserve credit and saved work

For occasional demos, use **Stop** in Azure Portal and confirm **Stopped
(Deallocated)** when the demo and processing have finished. The website cannot
upload or process while the server is off; start it again in Azure before the
next demo. Set auto-shutdown to a time that suits your demo hours in the correct
time zone, allowing for long renders. An interrupted edit may need **Resume
edit** after restart. Disks and some networking resources still consume credit
while compute is deallocated. See [Azure billing states](https://learn.microsoft.com/en-us/azure/virtual-machines/states-billing)
and [auto-shutdown](https://learn.microsoft.com/en-us/azure/virtual-machines/auto-shutdown-vm).

Update the checkout and rebuild services with the same Compose project name and
volumes. Do not use `docker compose down -v`, remove volumes, or delete the VM
and its managed disks if saved jobs must remain available. Back up job files and
models separately before destructive infrastructure changes.
