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
│  • Preset ①②③④ buttons    │                             │  • BLE GATT server           │
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
