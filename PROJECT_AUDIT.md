# LiftSafe project audit

Audit date: 2026-10-19

## Project and architecture

- Python 3 / Streamlit application. `app.py` is the trial selection and LI-condition prediction dashboard. Streamlit pages provide uploaded-video analysis, saved-model evaluation, and browser live-camera analysis.
- Computer vision uses MediaPipe Pose and OpenCV. The live page receives browser frames through `streamlit-webrtc`; it does not open a server-side webcam.
- Classical LI-condition models are saved as joblib artifacts. A separate temporal CNN uses PyTorch. Model inputs are 64 time steps by four features.
- Processing and training scripts are in `src/`. No automated `tests/` suite, README, `packages.txt`, runtime declaration, or local Streamlit configuration existed at audit time.

## Data and model artifacts

- `data/processed/inail_lifting_sequences/metadata.csv` contains 132 trials from 17 subjects. `sequences.npy` has shape `(132, 64, 4)` and `float32` values. LI class counts in this dataset are 51, 51, and 30.
- The saved Logistic Regression and Random Forest artifacts were fit on 90 trials from subjects S01–S10. Their metadata reports five-fold subject-wise GroupKFold estimates on those 90 trials: Logistic Regression accuracy 76.67%; Random Forest accuracy 80.00%, macro F1 0.798.
- Saved CNN grouped cross-validation and held-out predictions also cover 90 trials from S01–S10. The held-out CNN prediction file has 18 trials from two subjects. These results are not measurements on all 132 trials and must not be merged with the 132-trial dataset's counts.
- The model task is experimental LI condition classification. There is no validated safe/unsafe target or calibrated ergonomic risk score.
- Logistic Regression input layout is per feature: mean, standard deviation, minimum, maximum. Random Forest input layout is all means, then all standard deviations, minima, and maxima. Both have 16 features; feeding one layout to the other is incorrect.

## Confirmed issues and constraints

- The dashboard displayed both model outputs but did not state whether they agreed or call attention to disagreement.
- Trial matching used a global trial-name lookup after subject selection, which could select a different subject's row if trial names repeat.
- Live angle helpers were embedded in a page that imports optional camera/ML packages, so they could not be unit-tested in a lightweight environment.
- The live camera page uses browser WebRTC and does not call Streamlit UI methods inside the frame-processing worker. The original code had no explicit MediaPipe resource cleanup hook and the history view only refreshed on normal Streamlit reruns.
- The original WebRTC call had no ICE/STUN configuration. The streamlit-webrtc deployment guidance requires STUN for remote-host streaming; a public STUN server is now configured. TURN may still be needed on restrictive networks.
- Deployment failure logs were not included in the repository or request. The original dependency file pinned Streamlit, WebRTC, MediaPipe, and contrib OpenCV, left several direct dependencies unpinned, and installed PyTorch in the deployed app despite no Streamlit page importing it. The local host has Python 3.14.3 and lacks the app dependencies; this is not the target deployment runtime. MediaPipe 0.10.21 publishes Linux wheels through CPython 3.12. Streamlit Community Cloud currently defaults to Python 3.12.
- The OpenCV import failure cannot be diagnosed to a specific missing system library without the actual deployment traceback. No Linux system packages are added speculatively.
- Git status at audit time included pre-existing untracked generated outputs and `data/processed/video_manifest.csv`. They are user data and were preserved.

## Baseline verification

- Dataset files, model metadata, source scripts, saved prediction CSVs, requirements, and git status were inspected before edits.
- Existing model metadata and saved prediction files agree on the classical model evaluation subset size (90) and subjects (10); CNN held-out output has 18 rows from two subjects.
- The local Python 3.14 environment cannot import the app dependencies. `pip check` reports that its global `wheel` install lacks `packaging`; this environment is separate from the project and was not modified.
- No cloud logs or deployment credentials were available. A remote deployment was not attempted.

## Evaluation artifacts verified from saved predictions

| Artifact | Trials | Subjects | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression subject-wise OOF | 90 | 10 | 0.7667 | 0.7648 |
| Random Forest subject-wise OOF | 90 | 10 | 0.8000 | 0.7985 |
| Temporal CNN subject-wise OOF | 90 | 10 | 0.7444 | 0.7438 |
| Temporal CNN held-out subjects | 18 | 2 | 0.8889 | 0.8857 |

The metrics above were recomputed from each saved prediction CSV. The final line is a small two-subject test and must not be compared as if it were the same evaluation design as the 5-fold OOF estimates. The Random Forest OOF confusion counts were LI 1: 21 correct / 9 predicted LI 2; LI 2: 21 correct / 8 predicted LI 1 / 1 predicted LI 3; LI 3: 30 correct.

## Dependency resolution

- Runtime pins now constrain NumPy and Protobuf to MediaPipe-compatible versions and pin scikit-learn to the version recorded in both joblib model artifacts (1.9.1). PyTorch and SciPy are in `requirements-training.txt` because they are used by offline CNN/data-preparation scripts and are not imported by the Streamlit pages.
- A binary-wheel-only dry run for Linux CPython 3.12 resolved successfully when both `manylinux_2_28_x86_64` and `manylinux2014_x86_64` tags were included. This verifies package metadata and wheel availability for those tags; it does not replace an actual Streamlit Cloud build or OpenCV import test.
- The isolated local environment imports Streamlit, WebRTC, MediaPipe, OpenCV, NumPy, pandas, scikit-learn, joblib, and PyAV. `pip check` reports the installed MediaPipe wheel tag as unsupported for the local host, despite successful imports; the Linux CPython 3.12 wheel resolver succeeds separately.

## Implementation verification

- Six `unittest` cases pass for valid/degenerate angles, visibility and incomplete landmarks, expected sequence dimensions, model-specific feature order, and computed agreement/disagreement.
- Python 3.12 syntax compilation passes for the application, pages, helpers, and tests.
- Streamlit AppTest passes on the main dashboard, uploaded-video page, and model-evaluation page. The main dashboard also confirms both model agreement and disagreement messages against saved models/data. The live-camera page AppTest passes with WebRTC mocked and displays the stopped-camera guidance. AppTest cannot establish a physical camera stream.
- A local headless Streamlit server started successfully and its `/_stcore/health` endpoint returned `ok`. Actual browser permission, connection, stop/restart, and deployment behavior remain unverified.

## Browser camera check procedure

1. Start the app with `streamlit run app.py`, open the Live Camera Analysis page, and press Start. Use `localhost` locally or the HTTPS deployment URL remotely.
2. Grant the browser camera permission. Confirm the page reports a connection, body landmarks are drawn, and measurements appear with a visible full body.
3. Move out of frame or occlude landmarks; confirm the page waits for visible landmarks without labelling the frame safe/unsafe.
4. Check that measurement history updates and its CSV download opens. Press Stop, press Start again, and confirm the stream restarts.
5. If remote streaming remains disconnected after camera permission is granted, inspect the browser console and deployment logs. Some networks require a TURN relay in addition to STUN; configure one only with a verified need and appropriate service credentials.

## Follow-up audit and implementation (2026-10-19)

- Inspected the full Streamlit pages, all source filenames, model/evaluation artifacts, requirements, ignore rules, and annotation JSON. `data/raw/annotations.json` has 182 records and fields for subjective judgment, demographics, and box/action measurements. Its judgment values are 176 `unknown` and 6 `intermediate`; there are no fields naming safe, unsafe, injury, or risk. This is not defensible safe/unsafe ground truth. Do not map LI classes or `intermediate` into safety labels. A real safety classifier needs an explicit labeling protocol, appropriately authorized video/kinematic data, and independent occupational-safety expert labels with adjudication and subject-wise validation.
- The uploaded-video workflow now rejects empty and over-500-MB payloads, reports missing FFmpeg, uses a bounded FFmpeg conversion, validates output, releases OpenCV handles in `finally`, and scopes temporary files to `TemporaryDirectory`. The session stores the resulting video bytes rather than a path into a leaked temporary directory. Per-frame measurements now use the same visibility-filtered angle helper as live camera analysis. The detector remains a research review flag.
- Dashboard and evaluation artifact paths now resolve from the repository files rather than the process working directory. Added shared restrained glass-inspired styling with strong text contrast, visible keyboard focus, responsive Streamlit-native layout, and reduced-motion handling.
- Privacy review: the video manifest contains video identifiers and source paths; measurement CSVs contain per-frame individual motion/body measurements; plots are derived research results. These artifacts are not needed to run the Streamlit app and remain on disk. Exact local exports were added to `.gitignore`; this is not blanket ignoring. Existing tracked processed data, metadata, prediction files, and model artifacts still require confirmation of redistribution rights before publishing the repository.
- MediaPipe investigation: `.venv` is CPython 3.12 on arm64, while the installed distribution's internal WHEEL tag says `macosx_14_0_x86_64`; `direct_url.json` says it was installed from a temporary `universal2` wheel, and its main framework binding is a universal binary. `pip check` still reports it unsupported for this platform. An intermediate Pose initialization attempt did not proceed because the lite model asset was absent and network access was unavailable. A later attempt with the full model asset present failed while creating MediaPipe's `kGpuService` / `NSOpenGLPixelFormat` in this non-GUI shell. The user reports Pose previously initialized successfully on this Mac. These are distinct observations; the graphics-context failure is not proven to be caused by the wheel tag inconsistency. No package was reinstalled, metadata edited, or warning suppressed.
- Linux wheel-only resolution was previously successful for CPython 3.12 manylinux x86_64. No Linux runtime/container was available, so a Linux app startup is not verified. The complete Streamlit Cloud traceback is absent; do not infer an OS-library failure from that absence.
- Annotation JSON includes potentially identifying demographic/video metadata and is ignored through `data/raw/`. No credentials were found by a filename-only secret-pattern scan of source/config text. This scan is not a complete secret audit of arbitrary binary artifacts.
- Final post-change checks: `streamlit run app.py --server.headless true --server.port 8503` started; `curl http://127.0.0.1:8503/_stcore/health` returned `ok`; the server was stopped. AppTest passed for dashboard, video, evaluation, and live page with `webrtc_streamer` mocked. All six unit tests, compilation, and `git diff --check` passed. Saved LR/RF artifacts loaded and predicted from `sequences.npy` shape `(132, 64, 4)`.

## Final end-to-end review (2026-10-19)

### New regression coverage and video pipeline

- Moved uploaded-video processing into `src/video_processing.py` so it can be tested outside Streamlit UI execution. The page calls this shared function.
- Added an explicit 10-minute processing deadline in addition to the 500 MB upload limit and five-minute FFmpeg timeout.
- Added regression tests for empty/oversized uploads, corrupt input, missing FFmpeg, timeout cleanup, successful short-video decode/encode, and temporary directory removal. The successful synthetic clip test uses a fake MediaPipe Pose result (no landmarks); OpenCV decodes the source and result and FFmpeg produces the output. This verifies video I/O and cleanup, not real pose inference.
- Added inference checks for all three expected LI classes, finite inputs, prediction shape/class range, and model exceptions. Dashboard score shape/range/normalization is checked. Downloaded values are named model scores, not calibrated probabilities.
- Added `packages.txt` with `ffmpeg`, which the upload page directly invokes for MP4 conversion. This addresses a known runtime command dependency; it does not assume or add `libGL` packages without the Cloud traceback.

### MediaPipe packaging root cause

- The cached wheel `/tmp/liftsafe-mediapipe-check/mediapipe-0.10.21-cp312-cp312-macosx_11_0_universal2.whl` has SHA-256 `a65508f8c0e28a73f519c8a717f95949b9d5b2fb3bf4a632b181cdb71224d62e`. This matches the hash published by [PyPI for the official universal2 0.10.21 release file](https://pypi.org/project/mediapipe/0.10.21/).
- The wheel's own `mediapipe-0.10.21.dist-info/WHEEL` declares `cp312-cp312-macosx_14_0_x86_64`. The same internal tag is present in `.venv` and `/tmp/liftsafe-mp-test`; the wheel was not locally modified. The `.venv` interpreter reports `arm64`, and `pip debug --verbose` includes arm64/universal2 tags but not x86_64.
- `lipo -info` and `file` show the main `_framework_bindings.cpython-312-darwin.so` and the other principal MediaPipe extension modules have x86_64 and arm64 slices. The exact downloaded package imports, and its native framework binding loads under the arm64 temporary interpreter when the existing environment's dependencies are available via `PYTHONPATH`.
- Therefore the immediate `pip check` warning is caused by a **published wheel metadata inconsistency**: the official filename and PyPI record it as universal2, but the embedded WHEEL tag is Intel-only. It is not an altered local wheel. Pip 26.2.1 in `.venv` flags it; the isolated temp environment's older pip 23.2.1 does not report the platform warning (it does report dependencies missing because it was installed with `--no-deps`). Using the older pip would hide the bad metadata and is not a repair.
- No corrected official 0.10.21 wheel was obtained. A CLI download retry failed because the environment could not resolve `pypi.org`. No package metadata was edited and no package was reinstalled in `.venv`.
- The model asset `pose_landmark_full.tflite` exists. A real Pose 1 graph initialization in this shell failed with `Could not create an NSOpenGLPixelFormat` / missing `kGpuService`; native import succeeds but full graph execution is not verified in this non-GUI shell. The user reports earlier successful Pose initialization on the Mac. This graphics-context failure is a separate runtime limitation; it does not show the arm64 binary slice is missing.

### Final verification and deployment boundaries

- Thirteen unit tests pass. The suite includes the six original tests plus inference and video edge cases.
- `compileall` passes. Streamlit AppTest passes on all four pages, with `webrtc_streamer` mocked for the live-camera page.
- Saved LR/RF model artifacts load and predict on a stored sequence. Saved evaluation CSVs were independently recomputed: LR OOF 90 trials/10 subjects accuracy 0.7667, macro F1 0.7648; RF OOF 0.8000, 0.7985; Temporal CNN OOF 0.7444, 0.7438; separate CNN holdout 18 trials/2 subjects 0.8889, 0.8857. The two-subject holdout is not comparable to the 90-trial OOF evaluation.
- A local Streamlit process was previously started and its health endpoint returned `ok`; rerun after these latest model-score and video-module changes below.
- Docker CLI exists, but no Docker daemon is running. The local shell cannot reach PyPI, so no fresh Linux CPython 3.12 container/build was executed. The earlier binary-wheel metadata resolution for Linux CPython 3.12 x86_64 remains only a wheel-availability check, not a Linux runtime test.
- No Cloud traceback is stored in this repository. Request the complete app logs from the failed deployment, especially the final exception and preceding import/build lines; do not diagnose a specific system library without them.
- Uploaded-video processing is verified with synthetic frames and a mocked Pose result. Real MediaPipe pose estimation, real camera permission/streaming, and Streamlit Cloud remain unverified.
- There is no end-to-end camera/video-to-trained-model path: uploaded/live camera pages emit approximate 2D angles and experimental threshold review flags; saved classical/CNN models classify INAIL LI classes from motion-capture sequences. The current app correctly does not call these safe/unsafe outputs. Bridging modalities requires aligned training examples and ground truth, not a code-only mapping.
- After the final implementation changes, reran `streamlit run app.py --server.headless true --server.port 8503`; `curl http://127.0.0.1:8503/_stcore/health` returned `ok`. The process was stopped. AppTest was rerun afterward and all four pages passed with WebRTC mocked on the live page. On the latest finalization pass, all 13 unit tests passed, `compileall` passed, `git diff --check` passed, and `pip check` reported only `mediapipe 0.10.21 is not supported on this platform`.
