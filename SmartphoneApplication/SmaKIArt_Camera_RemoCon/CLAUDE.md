# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research-grade Android camera app for clinical ophthalmic image acquisition.

Core goals:
- Manual control of every camera parameter
- Reproducible imaging conditions across sessions
- Patient-based, eye-separated image organization

This is NOT a general-purpose camera app. Do not introduce features, abstractions, or UX patterns that belong in consumer camera apps.

---

## Build & Test Commands

```bash
# Windows
gradlew.bat assembleDebug         # Build debug APK
gradlew.bat assembleRelease       # Build release APK
gradlew.bat test                  # Run unit tests (local JVM)
gradlew.bat connectedAndroidTest  # Run instrumented tests (requires device/emulator)
gradlew.bat lint                  # Run Android Lint
gradlew.bat clean                 # Clean build outputs
gradlew.bat installDebug          # Build and install on connected device
```

---

## Project Structure

```
com/example/smakiartclinical/
├── MainActivity.kt               — sole entry point; hosts Compose tree + OrientationManager
├── ui/
│   ├── CameraScreen.kt           — full-screen overlay UI (preview + controls)
│   ├── CameraViewModel.kt        — state owner; bridges UI ↔ camera ↔ storage
│   └── components/
│       └── CameraPreviewView.kt  — AndroidView wrapper for TextureView
├── camera/
│   ├── CameraController.kt       — Camera2 lifecycle, preview, capture, transform
│   ├── ManualCameraSettingsApplier.kt — applies CameraSettings to a CaptureRequest.Builder
│   └── OrientationManager.kt     — sensor-based orientation detection
├── data/
│   ├── model/
│   │   ├── CameraSettings.kt     — ISO, exposure, focus, AE/AF flags (no AWB — always auto)
│   │   └── Preset.kt             — named snapshot of CameraSettings
│   ├── PresetDataStore.kt        — DataStore-backed preset persistence (max 4)
│   └── PhotoFileManager.kt       — MediaStore / File output stream factory
└── ui/theme/                     — Material 3 colors, typography, theme wrapper
```

---

## Critical Rules (MUST FOLLOW)

### Camera System

- Use **Camera2 API ONLY** — do not use CameraX under any circumstances
- All camera parameters must remain manually controllable
- Required manual controls: ISO, exposure time, focus distance, exposure compensation, AE/AF on/off
- **AWB is always `CONTROL_AWB_MODE_AUTO`** — do not expose AWB as a user setting; `CameraSettings` has no `awbEnabled` field

### Architecture

- Keep implementation **simple and minimal**
- Do NOT introduce: repository layer, use-case layer, dependency injection frameworks
- Do NOT create empty placeholder classes
- Do NOT duplicate responsibilities across files
- The ViewModel is the single source of truth for all state

### File Saving

- Standard shot path: `Pictures/SmaKIArtClinical/{patientId}/{eye}/IMG_YYYYMMDD_HHMMSS_SSS.jpg`
- Fixed-focus shot path: `Pictures/SmaKIArtClinical/{patientId}/{eye}/{patientId}_{tag}_IMG_YYYYMMDD_HHMMSS_SSS.jpg`
- Use MediaStore on API 29+; use `File` API on API 28
- Do NOT flatten the folder structure

---

## Orientation Architecture

This is the most complex area of the codebase. Read carefully before touching anything orientation-related.

### How orientation works

The activity uses `screenOrientation="sensorLandscape"` with `configChanges="orientation|screenSize|screenLayout|smallestScreenSize"`.

- The **window physically rotates** between `ROTATION_90` (normal landscape) and `ROTATION_270` (reverse landscape)
- The activity is NOT recreated — `onConfigurationChanged` fires instead
- `OrientationManager` detects rotation via `OrientationEventListener` and calls `viewModel.onOrientationChanged()`

### What `onOrientationChanged` does

```
LANDSCAPE       → displayRotationDegrees = 90,  selectedEye = RIGHT
REVERSE_LANDSCAPE → displayRotationDegrees = 270, selectedEye = LEFT
```

Setting `displayRotationDegrees` immediately triggers `startPreview()` → `applyPreviewTransform()`.

### Preview transform (CameraController.applyPreviewTransform)

The preview transform is **completely independent of the UI layer**.

The TextureView lives inside the physically-rotating window. When the window rotates 180° (landscape → reverse landscape), the TextureView's effective rendering flips. The transform compensates:

```
displayRotationDegrees = 90  (normal landscape)   → postRotate(-90°)
displayRotationDegrees = 270 (reverse landscape)  → postRotate(+90°)
```

The sign flip is the 180° compensation for the window-level rotation. The `bufferRect` width/height swap in the same function corrects the sensor's portrait-oriented frame buffer into landscape.

**DO NOT change this transform without understanding both the buffer-swap step and the window-rotation compensation.**

### UI rotation

The UI does NOT use `Modifier.rotate()`. Because the window itself rotates, all Compose content (including the keyboard for Patient ID input) rotates naturally. There is no Compose-level rotation applied to the overlay.

### Common orientation mistakes to avoid

- Do not apply `Modifier.rotate()` to the entire screen — the window already handles it
- Do not assume the preview transform needs a Compose-level rotation — it does not
- Do not hardcode `-90f` in `applyPreviewTransform()` — it must respect `displayRotationDegrees`
- Do not remove the `configChanges` attribute from the manifest — that would cause activity recreation on rotation, breaking the camera session

---

## Camera Logic Rules

- `CameraController.startPreview()` → `applyPreviewTransform()` must be called together; never call one without eventually calling the other
- `ManualCameraSettingsApplier.apply()` is called for both preview and still capture — changes there affect both
- Exposure time for the preview is capped at 200 ms inside `startPreview()` (`MAX_PREVIEW_EXPOSURE_NS`) — the slider and stored value may not exceed this cap
- `captureStillImage()` uses the full stored `exposureTimeNs` without additional capping beyond the hardware range
- `ImageReader` receives only JPEG frames; do not add other formats without considering memory pressure
- AWB is hardcoded to `CONTROL_AWB_MODE_AUTO` in `ManualCameraSettingsApplier` — do not add conditional AWB logic

### Camera error recovery

`CameraController` automatically retries `openCamera()` up to **2 times** (with an 800 ms delay) when `CameraDevice.StateCallback.onError` fires. `retryCount` resets to 0 on `onOpened`. Only after all retries are exhausted is `listener.onCameraError()` called.

Do not remove this retry logic — transient errors on app resume are expected on some devices.

---

## Focus Pair Capture (CaptureMode state machine)

`CameraViewModel` implements a sequential two-shot capture for the 3D and 10D focus buttons.

```
CaptureMode.NONE          — idle; normal captureImage() path
CaptureMode.FOCUS_PAIR_FIRST  — waiting for the first (current-settings) shot to complete
CaptureMode.FOCUS_PAIR_SECOND — waiting for the second (fixed-focus) shot to complete
```

State transitions:
1. `captureFocusPair(focusDistance, tag)` → sets `pendingFocusDistance`, `pendingFocusTag`, mode → `FOCUS_PAIR_FIRST`, fires first capture
2. `onCaptureSaved` with `FOCUS_PAIR_FIRST` → overrides controller settings with `focusDistance=pendingFocusDistance, afEnabled=false` (**without** changing `_settings`), fires second capture, mode → `FOCUS_PAIR_SECOND`
3. `onCaptureSaved` with `FOCUS_PAIR_SECOND` → restores controller to `_settings.value`, mode → `NONE`

Key invariant: `_settings` (the ViewModel's StateFlow) is **never mutated** during a focus pair. The focus override is applied only to `cameraController.updateSettings()`. This ensures the UI and presets are unaffected and restoration is always `cameraController.updateSettings(_settings.value)`.

Error handling: both `onCaptureError` and `onCameraError` reset `captureMode` to `NONE`, set `isCapturing = false`, and call `cameraController.updateSettings(_settings.value)` if a focus override was in effect.

Fixed focus distances: `FOCUS_3D = 3.00f`, `FOCUS_10D = 10.00f` (diopters).

---

## Slider Scale Rules

### ISO slider — logarithmic, rounded to nearest 100

The `Slider` composable operates in log space:
- `valueRange = ln(isoMin)..ln(isoMax)`
- `value = ln(currentIso)`
- `onValueChange`: `iso = (exp(logVal) / 100).roundToInt() * 100`, then coerced to hardware range

Do not switch this to linear — the log scale gives clinically useful granularity (fine at low ISO, coarse at high ISO).

### Exposure time slider — logarithmic, max 200 ms

- `valueRange = ln(expMinMs)..ln(200f)`
- `value = ln(currentExpMs)`
- `onValueChange`: `exposureTimeNs = (exp(logVal) * 1_000_000).toLong()`

`MAX_SLIDER_EXPOSURE_MS = 200f` in `CameraScreen.kt` and `MAX_PREVIEW_EXPOSURE_NS = 200_000_000L` in `CameraController.kt` must be kept in sync.

### Focus and EV sliders — linear (unchanged)

Focus distance and EV compensation sliders remain linear.

---

## Preset Auto-load

`CameraViewModel.init` launches a coroutine that reads `presetDataStore.presetsFlow.first()`. If index-0 is non-null, `updateSettings(preset1.settings)` is called before the camera opens. This sets `CameraController.currentSettings` so that the first `startPreview()` call (triggered by `onConfigured`) uses the preset values immediately.

Do not move this logic out of `init` — it must run before `openCamera()` is called by the UI.

---

## UI Rules

- `CameraScreen.kt` is a single full-screen `Box` with two layers: preview (Layer 1) and overlay (Layer 2)
- The settings panel is a conditional right-edge column — it does not navigate away; no Compose Navigation is used
- `BottomControlsBar` is a single `Row` — do not split it into multiple rows
- Patient ID field is intentionally narrow (120 dp) — real IDs are short
- The shutter buttons are custom circular `Box` composables, not Material `Button`s
  - Normal shutter: 64 dp, 3 dp border, no label
  - Focus shutter (3D / 10D): 56 dp, 2 dp border, text label inside
- Do NOT add a loading spinner, progress bar, or modal during capture — all shutter buttons dim to grey via `isCapturing`

---

## Common Pitfalls

| Area | Pitfall |
|---|---|
| Preview transform | Hardcoding `-90f` breaks reverse landscape |
| Preview transform | Changing `bufferRect` dimensions breaks aspect ratio |
| Orientation | Adding `Modifier.rotate()` to the screen double-rotates the UI |
| Orientation | Removing `configChanges` causes activity recreation, closes camera |
| Exposure slider | Always clamp to 200 ms (`MAX_SLIDER_EXPOSURE_MS`) — raw hardware range can be seconds |
| Slider updates | Every `onValueChange` call triggers `updateSettings()` → `startPreview()` — debouncing is a future improvement |
| MediaStore | On API 29+, `RELATIVE_PATH` creates folders automatically — do not call `mkdirs()` |
| Camera session | Do not close and reopen the camera for settings changes — use `setRepeatingRequest()` only |
| Focus pair | Do not call `updateSettings()` (ViewModel) during a focus pair — only call `cameraController.updateSettings()` directly |
| AWB | Do not re-add `awbEnabled` to `CameraSettings` — AWB is always auto and must stay that way |
| Log sliders | Do not convert ISO or exposure sliders back to linear — the log range is a deliberate clinical decision |

---

## Design Philosophy

- Determinism > convenience
- Accuracy > UI polish
- Simplicity > abstraction

---

## When Modifying Camera Logic

Be extremely careful with:

1. `applyPreviewTransform()` — any change to the matrix affects the live preview visually
2. `startPreview()` — called on every settings update; must be fast
3. `createCaptureSession()` — expensive; only called on camera open, not on settings changes
4. `captureStillImage()` — uses a separate `TEMPLATE_STILL_CAPTURE` request; do not merge with preview logic
5. Buffer/surface sizes — `ImageReader` size is set once at open time; changing it requires closing and reopening the session

---

## Output Requirements

- Provide COMPLETE Kotlin files when making changes
- No placeholders, no pseudo-code
- Include all necessary imports
- Keep changes localized — do not refactor unrelated code when fixing a specific issue
