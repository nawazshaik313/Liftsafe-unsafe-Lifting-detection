# LiftSafe completion status

## Completed and verified

- [x] Preserved all datasets, model artifacts, evaluation results, and user exports; no model or dataset was regenerated.
- [x] Kept experimental LI1/LI2/LI3 outputs distinct from safe/unsafe assessments. Annotation review found 176 `unknown`, 6 `intermediate`, and no documented safe/unsafe ground truth.
- [x] Validated model-specific Logistic Regression and Random Forest feature ordering, 64×4 input shape, finite values, expected classes, prediction shape/range, model exceptions, and model-score values.
- [x] Dashboard displays agreement/disagreement, training-subject caveats, and model scores without presenting them as calibrated probabilities.
- [x] Centralized camera/video landmark visibility checks for trunk, hip, and knee geometry.
- [x] Video upload validates empty/oversized/corrupt inputs; checks FFmpeg; applies a 10-minute processing deadline and five-minute encode timeout; releases OpenCV resources and cleans temporary storage.
- [x] Video I/O regression test generates a three-frame synthetic clip, mocks only the Pose graph, confirms FFmpeg output decodes, and checks temporary cleanup.
- [x] Added explicit `packages.txt` entry for FFmpeg, required by uploaded-video MP4 conversion.
- [x] Added shared accessible styling and repository-relative app data paths.
- [x] Reviewed local research exports and added exact `.gitignore` entries; files remain on disk.
- [x] Recomputed metrics from saved prediction CSVs and checked subject-wise split code.
- [x] Thirteen unit tests pass; Python compilation passes; AppTest passes for all pages (WebRTC mocked); actual saved LR/RF inference succeeds; the final local server health endpoint returned `ok`; the server was stopped.
- [x] Verified wheel hash against PyPI. Root cause of `pip check`: the published universal2 wheel contains an x86_64-only internal WHEEL tag even though its key native binaries include both arm64 and x86_64 slices. No metadata was edited and no package change was made.

## Implemented but not fully verified

- [ ] Full MediaPipe Pose graph initialization in this non-GUI shell fails with `NSOpenGLPixelFormat` / `kGpuService`; native module import works and the model asset is present. User reports it initialized previously on the Mac. Verify from the intended interactive environment.
- [ ] `pip check` remains nonzero in the project `.venv` due to the published WHEEL metadata inconsistency. Pip 23.2.1 in the no-dependencies temp environment did not flag the platform tag, but this only suppresses detection and is not a fix. No corrected official ARM64 wheel was downloaded.
- [ ] The Linux wheel resolver passed earlier for CPython 3.12 x86_64. No Linux runtime check: Docker daemon is unavailable and CLI networking to PyPI fails.
- [x] Final local Streamlit startup and `/_stcore/health` check passed after the model-score and video-module changes; server stopped cleanly.

## Blocked by unavailable resources or access

- [ ] Physical camera connect/stop/restart requires an HTTPS or localhost browser with camera permission. WebRTC AppTest is mocked only.
- [ ] Real-video MediaPipe estimation requires the Pose graph to initialize in an environment with its required graphics context. Current synthetic video test mocks Pose and verifies decode/encode/cleanup only.
- [ ] Cloud startup cause is unknown because no complete traceback was provided. Obtain Streamlit Cloud build and app logs through the final exception; do not add guessed OS packages.
- [ ] Sharing rights for the tracked processed dataset, subject/trial metadata, predictions, and models must be confirmed before publishing. Local manifest/measurement exports remain ignored and on disk.

## Requires data or expert validation

- [ ] A genuine unsafe-lifting classifier needs authorized, aligned camera or kinematic data, explicit expert labels with a documented labeling protocol and adjudication, and subject-wise external validation. The current video/live threshold flags are experimental and are not connected to the LI-condition classifiers.

No commit, push, public deployment, or publication was performed.
