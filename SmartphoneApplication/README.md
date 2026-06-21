# SmaKIArt Dual — Clinical Ophthalmic Imaging System

A two-Android-device system for clinical ophthalmic research.  
One device controls the examination and captures fundus images; the other displays a gaze-fixation stimulus and records the patient's frontal eye video.

---

## System Overview

```
┌─────────────────────────────┐          BLE GATT          ┌──────────────────────────────┐
│  SmaKIArt_Camera_RemoCon    │ ─────────────────────────► │  SmaKIArt_Screen_FrontCamera │
│  (Examiner's device)        │          commands           │  (Patient-facing device)     │
│                             │                             │                              │
│  • Fundus camera (Camera2)  │                             │  • Hot-air balloon stimulus  │
│  • Manual ISO / Exp / Focus │                             │  • Landscape background      │
│  • R/L eye selection        │                             │  • Front camera video rec.   │
│  • D-pad balloon control    │                             │  • Preset position recall    │
│  • Preset ①②③④ buttons    │                             │  • Fog (雲霧) blur stimulus  │
│  • Fog (雲霧) button        │                             │  • BLE GATT server           │
│  • Start/Stop REC button    │                             │                              │
│  • BLE GATT client          │                             │                              │
└─────────────────────────────┘                             └──────────────────────────────┘
        Android 11 (API 30)                                         Android 16 (API 36)
```

---

## Sub-projects

| Directory | Role | Device |
|---|---|---|
| [`SmaKIArt_Camera_RemoCon`](./SmaKIArt_Camera_RemoCon/) | Examiner's fundus camera + remote control | Android 11 |
| [`SmaKIArt_Screen_FrontCamera`](./SmaKIArt_Screen_FrontCamera/) | Patient-facing stimulus display + front camera recorder | Android 16 |

---

## BLE Communication Protocol

`Screen_FrontCamera` acts as the **BLE GATT server** (advertises).  
`Camera_RemoCon` acts as the **BLE GATT client** (scans and connects).

### UUIDs
| Role | UUID |
|---|---|
| Service | `e0cb5db0-afac-11de-8a39-0800200c9a66` |
| Command characteristic (WRITE) | `e0cb5db1-afac-11de-8a39-0800200c9a66` |

### Command Set
| Command | Sent by | Effect |
|---|---|---|
| `BALLOON:dx:dy` | Camera_RemoCon | Move balloon by (dx, dy) dp |
| `BALLOON_RESET` | Camera_RemoCon | Return balloon to center |
| `BALLOON_SIZE:delta` | Camera_RemoCon | Resize balloon ±delta dp (clamped 50–250 dp) |
| `PRESET:N` | Camera_RemoCon | Apply stored preset N (1–4) on Screen_FrontCamera |
| `VIDEO_START:patientId:eye` | Camera_RemoCon | Begin front-camera recording (`eye` = `RIGHT`/`LEFT`) |
| `VIDEO_STOP` | Camera_RemoCon | Stop front-camera recording |
| `FOG` | Camera_RemoCon | Trigger the fog (雲霧) blur animation on the stimulus panel (simulates autorefractor fogging) |

### Connection Flow
1. `Screen_FrontCamera` starts GATT server → registers service → **then** starts BLE advertising  
   (advertising is deferred to `onServiceAdded()` to ensure the characteristic exists before any client connects)
2. `Camera_RemoCon` scans (no filter; UUID verified in `onScanResult`), connects, calls `BluetoothGatt.refresh()` to clear stale GATT cache, requests MTU 512, then discovers services
3. Commands are UTF-8 strings written to the command characteristic

---

## Balloon Preset System

Presets 1–4 store the balloon's **position (offsetX, offsetY)** and **size (dp)**.  
They are persisted on `Screen_FrontCamera` via `SharedPreferences`.

- **Save**: tap a preset's **Save** button in the `Screen_FrontCamera` ControlPanel
- **Apply remotely**: tap **①②③④** in `Camera_RemoCon`'s RemotePanel → sends `PRESET:N`
- **Auto-load**: Preset 1 is applied automatically when `Screen_FrontCamera` starts

---

## Fog (雲霧) Stimulus

Simulates the **fogging step of an autorefractor** by applying a staged blur to the
entire stimulus panel (landscape + balloon) on `Screen_FrontCamera`.

- **Trigger locally**: tap **☁ 雲霧 (Fog)** in the `Screen_FrontCamera` ControlPanel
- **Trigger remotely**: tap the **☁** button (left of the **3D** shutter) in `Camera_RemoCon`'s
  BottomShutterBar → sends `FOG`
- **Animation** (≈3.3 s, one play per trigger):
  `0 →(0.8 s)→ mid blur (9 dp) →(0.5 s hold)→(0.5 s)→ max blur (15 dp) →(1.0 s hold)→(0.5 s)→ 0`
- State is owned by `MainViewModel.fogTrigger` (an incrementing `Int`); both the local button
  and the BLE `FOG` command call `triggerFog()`, so they share one animation path.
- The blur is display-only — it does **not** affect the recorded front-camera video.

---

## Camera_RemoCon — Capture, Gallery & On-Device SCA

The examiner's device runs the full clinical workflow end-to-end:
capture → save → re-open from gallery → per-image ellipse analysis → patient-level
S/C/A estimation via cosine fit.

### Live preview overlay (real-time)

While the camera is open, a background loop samples the TextureView (≈5 Hz),
runs `EllipseAnalyzer` (port of `pipeline_v150526`'s AdaptDoG core), and
overlays the detected ellipse in green on the preview.

**Critical implementation detail.** `TextureView.getBitmap()` does **not**
apply the `setTransform` matrix, so the bitmap is in *raw GL-texture*
orientation — which for `SENSOR_ORIENTATION = 90°` is portrait
(e.g. `864 × 1920`). Requesting `getBitmap(previewSize.width, previewSize.height)`
forces portrait content into a landscape destination and produces a non-uniform
stretch.

  - **Correct call**: `tv.getBitmap(previewSize.height, previewSize.width)` →
    portrait bitmap with no distortion.
  - **Overlay mapping**: The Compose `EllipseCanvas` mirrors `applyPreviewTransform`'s
    `-90° CCW` rotation and uniform scale to map portrait bitmap coordinates onto
    the landscape display.  `REVERSE_LANDSCAPE` flips the rotation to `+90° CW`.

### Capture flow

1. **Shutter** → saves a JPEG only.  No automatic transition to an analysis screen.
   Files go to `Pictures/SmaKIArtClinical/{patientId}/{eye}/IMG_{timestamp}.jpg`
   (via MediaStore on Android Q+, direct file path on older devices).
2. **3D / 10D shutters** → focus-pair capture (saves two JPEGs at preset focus distances).

### Gallery (4-level navigation)

Reachable from the **Gallery** button next to **End** in the session panel.

```
Patient list          → newest capture first; shows R/L counts per patient
   ↓ tap patient
Eye selector          → RIGHT / LEFT (disabled if no images)
   ↓ tap eye
Image list            → thumbnails + per-image "Analyze" button
                       + "All Analyze" button in the top bar
   ↓ tap "Analyze"            ↓ tap "All Analyze"
Single-image           SCA result screen
analysis (existing     - S, C, A, SE, R², n
PhotoAnalysisScreen)   - Cos-curve plot (drawn in Compose Canvas)
```

EXIF Orientation is read in `PhotoFileManager.loadBitmap` so re-loaded photos
analyse identically to live capture regardless of which physical orientation
the device was held at when the JPEG was taken.

### On-device SCA estimation

`analysis/SCAEstimator.kt` ports `pipeline_v150526`'s cosine fit verbatim:

```
For each image i: run EllipseAnalyzer → (αᵢ = angleDeg, Dᵢ = dEst)
Least-squares fit (3×3 normal equations, Cramer's rule):
    D(α) = P₀ + P₁·cos(2α) + P₂·sin(2α)
SE = P₀
C  = -2·√(P₁² + P₂²)        (cylinder, minus notation)
S  = SE - C/2
A  = ½·atan2(-P₂, -P₁) mod 180°
R² = 1 - SSres / SStot
```

Minimum sample count: **3 valid (α, D) pairs** (`SCAEstimator.MIN_VALID`).
Photos for which `EllipseAnalyzer` fails or `dEst == null` are skipped.

### Files added

| File | Role |
|---|---|
| `analysis/EllipseAnalyzer.kt` | AdaptDoG → ratio → p → D₂ (per image) |
| `analysis/SCAEstimator.kt` | Cosine fit → S/C/A (per patient × eye) |
| `data/CapturedPhoto.kt` | `CapturedPhoto`, `PatientSummary` data classes |
| `data/PhotoFileManager.kt` | Save (existing) + MediaStore enumeration + EXIF-aware load |
| `ui/GalleryScreen.kt` | 4-level gallery navigation + cos-curve plot (no Coil dep.) |
| `ui/PhotoAnalysisScreen.kt` | Single-image overlay analysis (reused from gallery) |
| `ui/CameraViewModel.kt` | Live loop, gallery state machine, `runAllAnalyze` |

---

## Video File Naming

Front-camera recordings are saved to `Movies/SmaKIArtClinical/` with the pattern:

```
{patientId}_{eye}_{YYYYMMDD_HHmmss_SSS}.mp4
```

Examples: `P001_R_20260508_143022_456.mp4`, `P001_L_20260508_143055_789.mp4`

Eye tag is `R` (RIGHT) or `L` (LEFT), as specified in `VIDEO_START`.

---

## Build

Both projects use the same build toolchain:

```bash
# Windows
gradlew.bat assembleDebug
gradlew.bat installDebug
```

| Property | Value |
|---|---|
| minSdk | 28 (Android 9) |
| targetSdk | 36 (Android 16) |
| compileSdk | 36 |
| Language | Kotlin |
| UI | Jetpack Compose + Material3 |

---

## Required Permissions

### Camera_RemoCon (Android 11)
- `CAMERA`, `RECORD_AUDIO`
- `BLUETOOTH`, `BLUETOOTH_ADMIN`, `ACCESS_FINE_LOCATION` (API ≤ 30, for BLE scan)
- `BLUETOOTH_SCAN` (neverForLocation), `BLUETOOTH_CONNECT` (API ≥ 31)

### Screen_FrontCamera (Android 16)
- `CAMERA`, `RECORD_AUDIO`
- `BLUETOOTH`, `BLUETOOTH_ADMIN` (API ≤ 30)
- `BLUETOOTH_ADVERTISE`, `BLUETOOTH_CONNECT` (API ≥ 31)
