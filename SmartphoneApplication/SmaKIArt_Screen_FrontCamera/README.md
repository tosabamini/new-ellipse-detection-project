# SmaKIArt Screen FrontCamera

Patient-facing Android app for the SmaKIArt Dual clinical ophthalmic imaging system.

Displays a hot-air balloon gaze-fixation stimulus over a landscape background, records the patient's eye with the front camera, and receives commands from the examiner's `Camera_RemoCon` device over BLE.

---

## Features

| Feature | Description |
|---|---|
| Gaze fixation stimulus | Hot-air balloon image (`hotballon01.png`) over landscape background (`landscape_mountain.png`) |
| Balloon position control | Moved remotely via D-pad commands from Camera_RemoCon |
| Balloon size control | Resized remotely ±15 dp per step; clamped 50–250 dp |
| Balloon preset system | 4 presets (position + size); saved locally, applied remotely via `PRESET:N` |
| Preset auto-load | Preset 1 applied automatically on launch |
| Fog (雲霧) stimulus | Staged blur over the stimulus panel (autorefractor fogging); triggered locally or via BLE `FOG` |
| Orientation-aware layout | Stimulus panel always on front-camera side; flips between ROTATION_90 and ROTATION_270 using `OrientationEventListener` |
| Front camera recording | Starts/stops via BLE `VIDEO_START:patientId:eye` / `VIDEO_STOP` command |
| Local recording | Manual Start/Stop Recording button in ControlPanel |
| BLE GATT server | Advertises Service UUID; accepts WRITE commands from Camera_RemoCon |

---

## UI Layout

```
┌────────────────────────────────────────────────────────────────┐
│  [Stimulus Panel — landscape + balloon]    [ControlPanel 280dp]│
│                                                                │
│   🏔 landscape background                  ● BLE Active       │
│                                            ● Remote: Connected │
│       🎈 balloon (movable, resizable)       Patient ID: _____ │
│                                                                │
│                                            Balloon Position   │
│                                            [↑5][↑10]          │
│                                            [←5][←10]          │
│                                            [→5][→10]          │
│                                            [↓5][↓10]          │
│                                            [Reset to Center]  │
│                                            [☁ 雲霧 (Fog)]     │
│                                                                │
│                                            Presets            │
│                                            [1●][x:0 y:0 s:110][Save]│
│                                            [2 ][— empty —    ][Save]│
│                                            [3 ][— empty —    ][Save]│
│                                            [4 ][— empty —    ][Save]│
│                                                                │
│                                            [● Start Recording]│
└────────────────────────────────────────────────────────────────┘
```

The ControlPanel side is always opposite to the front camera.  
In reverse landscape, the panels swap automatically.

---

## Balloon Preset System

Presets are stored in `SharedPreferences` (`balloon_presets`).  
Each preset stores: `offsetX` (float), `offsetY` (float), `sizeDp` (int).

| Action | How |
|---|---|
| Save current state | Tap **Save** next to the preset slot in ControlPanel |
| Apply locally | Tap the numbered button (blue = saved, grey = empty) |
| Apply remotely | Camera_RemoCon sends `PRESET:N`; ignored if slot is empty |
| Auto-apply on launch | Preset 1 (index 0) is applied if saved |

---

## Fog (雲霧) Stimulus

Simulates the fogging step of an autorefractor by blurring the whole stimulus panel
(landscape + balloon) for one staged animation (≈3.3 s):

```
0 →(0.8s)→ mid blur (9 dp) →(0.5s hold)→(0.5s)→ max blur (15 dp) →(1.0s hold)→(0.5s)→ 0
```

| Trigger | How |
|---|---|
| Local | Tap **☁ 雲霧 (Fog)** in the ControlPanel |
| Remote | Camera_RemoCon sends `FOG` (☁ button left of the 3D shutter) |

State is owned by `MainViewModel.fogTrigger` (incrementing `Int`); both paths call
`triggerFog()`. The blur is display-only and is not captured in the recorded video.
Timing/blur constants are the `FOG_*` values at the top of `MainScreen.kt`.

---

## Video Recording

Front-camera recordings are saved to `Movies/SmaKIArtClinical/`:

```
{patientId}_{eye}_{YYYYMMDD_HHmmss_SSS}.mp4
```

- `eye` = `R` or `L` (from `VIDEO_START:patientId:eye` command)
- Local recordings use `LOCAL` as the eye tag
- Uses `MediaRecorder` with H264 video + AAC audio, up to 1280×720 @ 30 fps
- Android 10+: saved via `MediaStore` with `IS_PENDING` flag for atomic write

---

## Orientation Detection

Uses `OrientationEventListener` (sensor-based, more reliable than `onConfigurationChanged` for detecting ROTATION_90 ↔ ROTATION_270 within `sensorLandscape`):

| Sensor angle | Meaning | `isReverseLandscape` |
|---|---|---|
| 225–315° | Normal landscape (ROTATION_90) — front camera LEFT | `false` |
| 45–135° | Reverse landscape (ROTATION_270) — front camera RIGHT | `true` |

---

## Build

```bash
gradlew.bat assembleDebug
gradlew.bat installDebug
```

**minSdk:** 28 · **targetSdk:** 36 · **Device tested:** Android 16
