# LFM3 Omni ONNX chat

Camera and voice chat with
[LiquidAI/LFM3-Omni-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX).
The model receives microphone audio and camera frames, then responds with text
and speech. Manual interruption and optional voice interruption are supported.

This version has its own uv environment in `src-onnx/.venv`. It uses ONNX Runtime
without installing PyTorch or Transformers. Run commands from the repository
root with [uv](https://docs.astral.sh/uv/) and Python 3.12 available.

## Start locally

Choose one command for your hardware.

CPU with the q4 bundle:

```sh
uv run --project src-onnx --python 3.12 --extra cpu serve \
  --device cpu --precision q4 --download
```

NVIDIA GPU with the q4f16 bundle:

```sh
uv run --project src-onnx --python 3.12 --extra gpu serve \
  --device cuda --precision q4f16 --download
```

Open **http://localhost:8000** on the same computer. Camera and microphone access
on localhost works without SSL certificates. Allow access when the browser asks.

For optional HTTPS, use the [OpenSSL certificate instructions](../README.md#optional-https)
and add `--ssl-certfile cert.pem --ssl-keyfile key.pem` to your ONNX server command.

The first run installs dependencies and downloads the selected model bundle.
Omit `--download` on later runs. The gpu extra includes CUDA and cuDNN runtime
libraries; a compatible NVIDIA driver is still required.

Use only one runtime extra at a time. The `cpu` and `gpu` packages provide the
same Python module, so uv treats these extras as mutually exclusive.

## Choose a precision

| Precision | Device | Bundle size |
| --- | --- | --- |
| `q4` | CPU | About 2.91 GB |
| `q4f16` | NVIDIA GPU | About 2.47 GB |

Both bundles use int4 and int8 weights. q4f16 uses float16 activations for CUDA.
Plain q4 uses float32 activations, and CUDA lacks the float32 GroupQueryAttention
kernel required by these graphs. Use q4f16 for GPU execution.

`--device auto --precision auto` chooses q4f16 when the CUDA provider is available,
otherwise q4. Explicit `--device cuda` fails if CUDA cannot load.

## Model storage

Downloads stay in `resources/models/lfm3-omni-2b-onnx/`. The server fetches only
the selected bundle, tokenizer, and configuration files. Both bundles can share
this folder. Keep it to reuse downloads.

For an existing model, add `--model /path/to/LFM3-Omni-2B-ONNX` to your command.
An explicit local model folder never triggers model downloads. It must contain:

- The selected bundle's five graphs under `onnx/` and their `.onnx_data` files.
- `config.json` and `processor_config.json`.
- `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`.
- `audio_tokenizer/config.json`.

## Conversation and turn detection

Click **Start conversation**, speak, then pause for a reply. Use **Send** to
finish your turn manually. Speak over a reply or press **Interrupt & speak**
to interrupt. Headphones help prevent speaker echo from triggering interruptions.

The WebSocket carries microphone PCM and camera JPEGs to the server, and streamed
text and speech back. The server buffers each utterance until the turn ends.
Conversation media stays in `resources/turns/` until its session is cleared or
expires. Model downloads remain cached.

To enable Smart Turn v3.2, add `--turn-detector smart --download` to the server
command once. Its model is saved in `resources/models/smart-turn-v3/` and runs
on CPU. Later runs can omit `--download`. Auto mode uses a cached classifier or
falls back to silence detection. `--turn-model` accepts an existing classifier.

## Interface preview

Serve the interface without loading weights:

```sh
uv run --project src-onnx --python 3.12 serve --ui-only
```

This mode does not generate replies. For a labelled preview with synthetic
responses, see [the main README](../README.md#development-checks-and-preview).

## Validation and limitations

The tests use small graph fixtures and synthetic media. Full model inference,
speech quality, and CPU/CUDA latency have not been verified on the development
machine.

The published ONNX codec has no streaming cache inputs. This server decodes
accumulated speech codes every four frames and holds the inverse STFT overlap
before sending audio. That repeats codec work as replies grow and may be slower
than the PyTorch streaming codec.

Runtime generation and cache handling adapt Liquid AI's
[official ONNX runtime](https://github.com/Liquid4All/onnx-export).
See [NOTICE](lfm3_onnx/NOTICE) and
[LICENSE.liquidonnx](lfm3_onnx/LICENSE.liquidonnx) for attribution and license terms.
