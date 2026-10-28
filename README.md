# LFM3 video chat

Install the locked dependencies:

```bash
uv sync --locked
```

Run with HTTPS:

To create a self-signed certificate for localhost:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1"
chmod 600 key.pem
```

Start the server, then accept the self-signed certificate when opening the page:

```bash
uv run --python 3.12 serve.py \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem
```

Open https://localhost:8000 and allow camera and microphone access. Supply your
certificate and private key at those paths. Use a certificate trusted by the
browser. The page connects over a secure WebSocket automatically.

For HTTP on localhost:

```bash
uv run --python 3.12 serve.py
```

The model loads from the Hugging Face cache by default. Use `--model /path/to/model`
for a local directory, or `--download` to allow downloading missing model files.

The browser streams microphone audio and camera frames while you speak. After a
pause, the model processes the utterance and streams back text and speech. Use
**Interrupt & speak** to stop a reply, or enable voice interruption with headphones.

Run the tests:

```bash
uv run pytest
```
