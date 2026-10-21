# LiftSafe

LiftSafe is an experimental Streamlit application for reviewing lifting motion. It contains saved INAIL lifting-condition classifiers, uploaded-video pose measurements, and browser-based live pose measurements.

## Run locally

Use a Python 3.12 environment. This repository has been exercised with its existing `.venv`:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Uploaded-video conversion invokes FFmpeg. Install it in the operating system image before running that feature; Streamlit Community Cloud reads the `ffmpeg` entry in `packages.txt`. The browser camera feature needs camera permission and a secure page (HTTPS or localhost). Some restrictive networks may also require a TURN relay.

For offline data preparation and Temporal CNN training, install `requirements-training.txt` instead of the runtime requirements alone. Training scripts may require access to the project's authorized datasets.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q app.py pages src tests
.venv/bin/python -m pip check
```

The repository's tested MediaPipe 0.10.21 universal2 PyPI wheel has inconsistent embedded platform metadata: on the audited Apple Silicon environment, `pip check` reports that MediaPipe is unsupported even though its native bindings import. Pose graph startup also failed in the audited non-GUI shell because its `kGpuService` could not create an `NSOpenGLPixelFormat`. Do not edit installed wheel metadata to suppress these findings; repeat Pose initialization in the target interactive macOS environment and review the current MediaPipe release artifacts before changing the pin.

## What the outputs mean

- Saved Logistic Regression and Random Forest models classify experimental INAIL LI1/LI2/LI3 conditions from motion-capture sequences. Their outputs are model scores and are not calibrated probabilities.
- Video and live-camera pages calculate approximate, image-space 2D joint angles and optional threshold-based review flags. These measurements are not clinically validated and are not inputs to the saved LI classifiers.
- The available annotation file does not provide verified safe/unsafe labels. LiftSafe does not establish that a lift is safe or unsafe. A safety classifier requires authorized aligned data, expert labels under a documented protocol, and subject-wise external validation.

See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for the inspected data, evaluation details, validation evidence, and remaining environment limitations. See [TODO.md](TODO.md) for outstanding work.
