# SmaKIArt Camera RemoCon

Examiner-side Android app for the SmaKIArt Dual clinical ophthalmic imaging system.

Provides a full-screen Camera2 fundus camera with manual parameter control, and a BLE remote control panel for the patient-facing `Screen_FrontCamera` device.

---

## Features

### Camera (this device)
| Feature | Description |
|---|---|
| Manual ISO | Logarithmic slider, hardware range, steps of ~100 ISO |
| Manual exposure | Up to 200 ms; logarithmic slider |
| Manual focus | Diopter-based slider (`LENS_FOCUS_DISTANCE`) |
| EV compensation | `CONTROL_AE_EXPOSURE_COMPENSATION` |
| AE / AF toggle | Fully disable or enable auto exposure / auto focus |
| AWB | Always `CONTROL_AWB_MODE_AUTO` — not user-configurable |
| Standard shutter | Single shot at current settings |
| 3D focus shutter | Current-settings shot + 3.00 D fixed-focus shot |
| 10D focus shutter | Current-settings shot + 10.00 D fixed-focus shot |
| Camera presets | Save / load up to 4 camera setting presets (DataStore) |
| Preset auto-load | Preset 1 applied automatically on launch |
| Error recovery | Auto-retries camera open up to 2 times on hardware error |
| Orientation | `sensorLandscape`; RIGHT eye in normal landscape, LEFT in reverse |
| Storage | `Pictures/SmaKIArtClinical/{PatientID}/{RIGHT\|LEFT}/` |

### BLE Remote Control (for Screen_FrontCamera)
| Feature | Description |
|---|---|
| D-pad | Move balloon stimulus in 4 directions; step = 5 or 10 dp |
| Balloon size | Resize balloon with − / + buttons (±15 dp per tap) |
| Preset buttons | ①②③④ — send `PRESET:N` to apply stored position/size preset |
| Start / Stop REC | Green/red circle button; sends `VIDEO_START:patientId:eye` or `VIDEO_STOP` |
| Fog (雲霧) | ☁ button (left of the 3D shutter); sends `FOG` to blur the stimulus panel |
| Disconnect | One-tap disconnect when connected |

---

## UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  [RemotePanel 196dp]   [Camera Preview — full screen]  [SessionPanel 176dp] │
│                                                              │
│   Remote            ·····preview·····         Session ⚙     │
│   ● Connected  [X]                            Patient ID     │
│   [D-pad ↑]                                   [R] [L]        │
│   [← ] [→ ]                                   End Session    │
│   [D-pad ↓]                                                  │
│   Step: [5][10]                                              │
│   Size: [−][+]                                               │
│   Preset: ①②③④                                             │
│                                                              │
│         [●REC]  [☁]  [3D]  [⬤ shutter]  [10D]            │
└──────────────────────────────────────────────────────────────┘
```

- **RemotePanel** (left, always visible): BLE connection + balloon control + presets
- **SessionPanel** (right): Patient ID, eye selection, settings icon
- **BottomShutterBar**: REC button (left) + ☁ Fog + 3D / main shutter / 10D (center-right, +10dp offset)

---

## Storage Structure

```
Pictures/
└── SmaKIArtClinical/
    └── {PatientID}/
        ├── RIGHT/
        │   ├── IMG_YYYYMMDD_HHMMSS_SSS.jpg
        │   ├── {PatientID}_3D_IMG_YYYYMMDD_HHMMSS_SSS.jpg
        │   └── {PatientID}_10D_IMG_YYYYMMDD_HHMMSS_SSS.jpg
        └── LEFT/
            └── ...
```

---

## Build

```bash
gradlew.bat assembleDebug
gradlew.bat installDebug
```

**minSdk:** 28 · **targetSdk:** 36 · **Device tested:** Android 11

---

## BLE Notes

- Scans without a filter; verifies Service UUID in `onScanResult`
- On Android 11 (API 30): requires `ACCESS_FINE_LOCATION` at runtime for BLE scan
- After connection: calls `BluetoothGatt.refresh()` via reflection to clear GATT cache before `discoverServices()`
- MTU negotiated to 512 before service discovery
