# CLAUDE.md — SmaKIArt_Screen_FrontCamera

This file provides guidance to Claude Code when working with this project.

## Project Overview

Patient-facing Android app in the SmaKIArt Dual clinical ophthalmic imaging system.

Responsibilities:
- Display a hot-air balloon gaze-fixation stimulus over a landscape background
- Receive and execute BLE commands from `Camera_RemoCon` (balloon move, resize, preset, video start/stop, fog)
- Record the patient's eye with the front camera when commanded
- Manage 4 balloon position+size presets, auto-applying Preset 1 on launch
- Play a fog (雲霧) blur animation over the stimulus panel, triggered locally or via the BLE `FOG` command

This is NOT a general-purpose display or camera app. All UI decisions serve clinical correctness.

---

## Build Commands

```bash
gradlew.bat assembleDebug
gradlew.bat installDebug
gradlew.bat test
gradlew.bat lint
```

---

## Project Structure

```
com/example/smakiart_screen_frontcamera/
├── MainActivity.kt                — entry point; OrientationEventListener; permissions
├── bluetooth/
│   └── BluetoothServer.kt         — BLE GATT server; addService → onServiceAdded → advertise
├── camera/
│   └── FrontCameraRecorder.kt     — Camera2 front-camera recording; SessionConfiguration
├── data/
│   └── BalloonPresetStore.kt      — SharedPreferences persistence for 4 balloon presets
└── ui/
    ├── MainScreen.kt              — Compose UI: StimulusPanel + ControlPanel
    └── MainViewModel.kt           — State owner; bridges BLE commands ↔ UI state ↔ recorder
```

---

## Critical Rules

### BLE Server Initialization Order

`BluetoothGattServer.addService()` is **asynchronous**.  
Advertising MUST NOT start until `onServiceAdded()` fires with `GATT_SUCCESS`.  
If advertising starts before the service is registered, clients will connect but fail to discover the characteristic.

```
start() → gattServer.addService(service)
              ↓  [async]
         onServiceAdded(GATT_SUCCESS)
              ↓
         startAdvertising()   ← only here
```

Do NOT move advertising out of `onServiceAdded`.

### Orientation Detection

Uses `OrientationEventListener` (sensor-based), NOT `onConfigurationChanged` + display rotation.  
The reason: within `sensorLandscape`, switching between ROTATION_90 and ROTATION_270 does not reliably trigger `onConfigurationChanged` on all Android versions.

Angle mapping (clockwise degrees from natural/portrait orientation):
- **225–315°** → normal landscape (ROTATION_90) → front camera on LEFT → `isReverseLandscape = false`
- **45–135°** → reverse landscape (ROTATION_270) → front camera on RIGHT → `isReverseLandscape = true`

`OrientationEventListener` is enabled in `onResume()` and disabled in `onPause()`.

### StimulusPanel Layout

The balloon is positioned using `BoxWithConstraints` with dp offsets:
- Center point: `(maxWidth/2, maxHeight * 0.42f)` — slightly above vertical center
- Clamping: balloon is clamped so it cannot leave the panel bounds
- Size: `balloonW = balloonSizeDp.dp`, `balloonH = (balloonSizeDp * 1.18f).dp` (preserves aspect ratio)

Do NOT replace `BoxWithConstraints` with a fixed-size Box — the panel width varies with orientation.

### Preset Indexing

Presets are **0-indexed** internally (List, array), but **1-indexed** in the BLE protocol and SharedPreferences keys.

| Layer | Indexing |
|---|---|
| `_presets: List<BalloonPreset?>` | 0-indexed (0–3) |
| `BalloonPresetStore.save(index, ...)` | 1-indexed (1–4) |
| BLE command `PRESET:N` | 1-indexed (1–4) |
| UI display ("P1"–"P4") | 1-indexed |

Conversion: `onPresetApply(presetNumber: Int)` → `applyPreset(presetNumber - 1)`

### Camera Recording

- Uses Camera2 API with `SessionConfiguration` + `OutputConfiguration` (no deprecated `createCaptureSession`)
- `SessionConfiguration` is available since API 28 = minSdk
- `MediaRecorder` uses `MediaRecorder(context)` constructor (API 31+) or deprecated no-arg constructor
- On Android 10+: `MediaStore` with `IS_PENDING = 1` during recording, set to `0` on stop
- Front camera identified by `LENS_FACING_FRONT`; best size ≤ 1280×720

### Fog (雲霧) Animation

Simulates autorefractor fogging by blurring the whole `StimulusPanel` (landscape + balloon).

- The trigger lives in the ViewModel as `fogTrigger: StateFlow<Int>` (NOT a `MainScreen`-local
  `remember`). Both the local Fog button and the BLE `FOG` command call `viewModel.triggerFog()`,
  which increments it. This is what makes the feature remotely controllable — keep it in the ViewModel.
- `MainScreen` collects `fogTrigger` and drives an `Animatable` blur in `LaunchedEffect(fogTrigger)`.
  `fogTrigger == 0` is the initial no-op state; the effect returns early so no animation plays on launch.
- Timing/blur constants live at the top of `MainScreen.kt` (`FOG_*`). Sequence (≈3.3 s):
  `0 →(0.8 s)→ 9 dp →(0.5 s hold)→(0.5 s)→ 15 dp →(1.0 s hold)→(0.5 s)→ 0`.
- The blur is applied with `Modifier.blur(radius, edgeTreatment = Rectangle)`; `0 dp` is a no-op.
- Display-only: it does not touch the recorded front-camera video.

---

## State Flows in MainViewModel

| StateFlow | Type | Description |
|---|---|---|
| `balloonOffset` | `Offset` | Current balloon position (dp offset from center) |
| `balloonSizeDp` | `Int` | Current balloon width in dp (50–250) |
| `presets` | `List<BalloonPreset?>` | 4 preset slots; null = empty |
| `isRecording` | `Boolean` | Front camera recording active |
| `isConnected` | `Boolean` | BLE client connected |
| `patientId` | `String` | Current patient ID (set by VIDEO_START command) |
| `advertisingState` | `Boolean?` | null=starting, true=active, false=failed |
| `advertisingError` | `Int` | Error code when advertising fails |
| `isReverseLandscape` | `Boolean` | Set by OrientationEventListener in MainActivity |
| `cameraReady` | `Boolean` | Front camera opened successfully |
| `fogTrigger` | `Int` | Incrementing counter; each change plays one fog blur animation. Bumped by `triggerFog()` (local Fog button + BLE `FOG`) |

---

## BLE Command Protocol

Commands are UTF-8 strings written to `COMMAND_CHAR_UUID`.

| Command | Action |
|---|---|
| `BALLOON:dx:dy` | Move balloon by (dx, dy) dp |
| `BALLOON_RESET` | Return balloon to Offset.Zero |
| `BALLOON_SIZE:delta` | Change size by delta dp (clamp 50–250) |
| `PRESET:N` | Apply preset N (1-indexed); no-op if slot is empty |
| `VIDEO_START:patientId:eye` | Start recording; eye = "RIGHT" or "LEFT" |
| `VIDEO_STOP` | Stop recording |
| `FOG` | Trigger the fog (雲霧) blur animation once (calls `triggerFog()`) |

---

## Common Pitfalls

| Area | Pitfall |
|---|---|
| BLE advertising | Do NOT start advertising in `start()` — wait for `onServiceAdded()` |
| GATT cache | Client must call `BluetoothGatt.refresh()` before `discoverServices()` to clear stale cache |
| Orientation | Do NOT use `windowManager.defaultDisplay.rotation` on Android 11+ — use `OrientationEventListener` |
| Preset index | Do NOT pass BLE `presetNumber` (1-based) directly to `_presets[i]` — subtract 1 first |
| StateFlow update | To update a `List` in `MutableStateFlow`, create a new list with `toMutableList().also { ... }` |
| `addService` | Do NOT call `addService` more than once; close and re-open `gattServer` to reset |
| Fog trigger | Do NOT move `fogTrigger` back into a `MainScreen`-local `remember` — it must stay in the ViewModel so the BLE `FOG` command can drive the same animation |

---

## Output Requirements

- Provide COMPLETE Kotlin files when making changes
- No placeholders, no pseudo-code
- Include all necessary imports
- Keep changes localized — do not refactor unrelated code
