package com.example.smakiartclinical.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.view.TextureView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import android.net.Uri
import com.example.smakiartclinical.analysis.EllipseAnalyzer
import com.example.smakiartclinical.analysis.EllipseResult
import com.example.smakiartclinical.analysis.SCAEstimator
import com.example.smakiartclinical.analysis.SCAResult
import com.example.smakiartclinical.bluetooth.BluetoothClient
import com.example.smakiartclinical.camera.CameraController
import com.example.smakiartclinical.camera.DeviceOrientation
import com.example.smakiartclinical.data.CapturedPhoto
import com.example.smakiartclinical.data.PatientSummary
import com.example.smakiartclinical.data.PhotoFileManager
import com.example.smakiartclinical.data.PresetDataStore
import com.example.smakiartclinical.data.model.CameraSettings
import com.example.smakiartclinical.data.model.Preset
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class SessionState(
    val patientId: String = "",
    val selectedEye: String = "RIGHT",
    val capturedFiles: List<String> = emptyList()
)

data class UiMessage(val text: String, val id: Long = System.currentTimeMillis())

enum class BtConnectionState { DISCONNECTED, SCANNING, CONNECTING, CONNECTED }

sealed class GalleryView {
    data object None         : GalleryView()
    data object PatientList  : GalleryView()
    data class  EyeSelector(val patientId: String) : GalleryView()
    data class  ImageList(val patientId: String, val eye: String) : GalleryView()
    data class  AllAnalyzeResult(val patientId: String, val eye: String, val result: SCAResult) : GalleryView()
}

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        val DEFAULT_SETTINGS = CameraSettings(
            iso = 100,
            exposureTimeNs = 10_000_000L,
            focusDistance = 0f,
            exposureCompensation = 0,
            aeEnabled = false,
            afEnabled = false
        )
        private const val FOCUS_3D = 3.00f
        private const val FOCUS_10D = 10.00f
    }

    private enum class CaptureMode { NONE, FOCUS_PAIR_FIRST, FOCUS_PAIR_SECOND }
    @Volatile private var captureMode = CaptureMode.NONE
    @Volatile private var pendingFocusDistance = 0f
    @Volatile private var pendingFocusTag = ""

    val cameraController = CameraController(application)
    private val photoFileManager = PhotoFileManager(application)
    private val presetDataStore = PresetDataStore(application)
    private val ellipseAnalyzer = EllipseAnalyzer()

    private var previewTextureView: TextureView? = null
    private var analysisJob: Job? = null

    private val _ellipseResult = MutableStateFlow<EllipseResult?>(null)
    val ellipseResult: StateFlow<EllipseResult?> = _ellipseResult.asStateFlow()

    private val _isAnalysisRunning = MutableStateFlow(false)
    val isAnalysisRunning: StateFlow<Boolean> = _isAnalysisRunning.asStateFlow()

    // ── Photo analysis screen (entered from gallery) ─────────────────────────

    private val _showAnalysisScreen = MutableStateFlow(false)
    val showAnalysisScreen: StateFlow<Boolean> = _showAnalysisScreen.asStateFlow()

    private val _capturedBitmap = MutableStateFlow<Bitmap?>(null)
    val capturedBitmap: StateFlow<Bitmap?> = _capturedBitmap.asStateFlow()

    private val _captureAnalysisResult = MutableStateFlow<EllipseResult?>(null)
    val captureAnalysisResult: StateFlow<EllipseResult?> = _captureAnalysisResult.asStateFlow()

    private val _isAnalyzingCapture = MutableStateFlow(false)
    val isAnalyzingCapture: StateFlow<Boolean> = _isAnalyzingCapture.asStateFlow()

    private val _captureAnalysisAttempted = MutableStateFlow(false)
    val captureAnalysisAttempted: StateFlow<Boolean> = _captureAnalysisAttempted.asStateFlow()

    // ── Gallery navigation ────────────────────────────────────────────────────

    private val _galleryView = MutableStateFlow<GalleryView>(GalleryView.None)
    val galleryView: StateFlow<GalleryView> = _galleryView.asStateFlow()

    private val _patientSummaries = MutableStateFlow<List<PatientSummary>>(emptyList())
    val patientSummaries: StateFlow<List<PatientSummary>> = _patientSummaries.asStateFlow()

    private val _galleryPhotos = MutableStateFlow<List<CapturedPhoto>>(emptyList())
    val galleryPhotos: StateFlow<List<CapturedPhoto>> = _galleryPhotos.asStateFlow()

    private val _isRunningAllAnalyze = MutableStateFlow(false)
    val isRunningAllAnalyze: StateFlow<Boolean> = _isRunningAllAnalyze.asStateFlow()

    private val _allAnalyzeProgress = MutableStateFlow(0 to 0)  // (done, total)
    val allAnalyzeProgress: StateFlow<Pair<Int, Int>> = _allAnalyzeProgress.asStateFlow()

    // ─────────────────────────────────────────────────────────────────────────

    private val _settings = MutableStateFlow(CameraSettings())
    val settings: StateFlow<CameraSettings> = _settings.asStateFlow()

    private val _session = MutableStateFlow(SessionState())
    val session: StateFlow<SessionState> = _session.asStateFlow()

    private val _message = MutableStateFlow<UiMessage?>(null)
    val message: StateFlow<UiMessage?> = _message.asStateFlow()

    private val _isCapturing = MutableStateFlow(false)
    val isCapturing: StateFlow<Boolean> = _isCapturing.asStateFlow()

    private val _currentOrientation = MutableStateFlow(DeviceOrientation.LANDSCAPE)
    val currentOrientation: StateFlow<DeviceOrientation> = _currentOrientation.asStateFlow()

    val presets: StateFlow<List<Preset?>> = presetDataStore.presetsFlow
        .stateIn(viewModelScope, SharingStarted.Eagerly, List(PresetDataStore.MAX_PRESETS) { null })

    // ── Wi-Fi 接続 ────────────────────────────────────────────────────────────

    val bluetoothClient = BluetoothClient(application)

    private val _btState = MutableStateFlow(BtConnectionState.DISCONNECTED)
    val btState: StateFlow<BtConnectionState> = _btState.asStateFlow()

    private val _isRemoteRecording = MutableStateFlow(false)
    val isRemoteRecording: StateFlow<Boolean> = _isRemoteRecording.asStateFlow()

    // ─────────────────────────────────────────────────────────────────────────

    init {
        viewModelScope.launch {
            val preset1 = presetDataStore.presetsFlow.first()[0]
            if (preset1 != null) updateSettings(preset1.settings)
        }

        bluetoothClient.onConnecting = {
            _btState.value = BtConnectionState.CONNECTING
        }
        bluetoothClient.onConnected = {
            _btState.value = BtConnectionState.CONNECTED
        }
        bluetoothClient.onDisconnected = {
            _btState.value = BtConnectionState.DISCONNECTED
            _isRemoteRecording.value = false
            if (bluetoothClient.lastError.isNotBlank()) {
                postMessage("接続エラー: ${bluetoothClient.lastError}")
                bluetoothClient.lastError = ""
            }
        }
    }

    // --- Camera lifecycle ---

    fun openCamera(textureView: TextureView) {
        previewTextureView = textureView
        cameraController.setListener(object : CameraController.Listener {
            override fun onCameraOpened() {}

            override fun onCameraError(error: String) {
                val wasInFocusPair = captureMode != CaptureMode.NONE
                captureMode = CaptureMode.NONE
                _isCapturing.value = false
                postMessage(error)
                if (wasInFocusPair) cameraController.updateSettings(_settings.value)
            }

            override fun onCaptureSaved(filename: String, jpegBytes: ByteArray, jpegOrientationDeg: Int) {
                when (captureMode) {
                    CaptureMode.NONE -> {
                        _isCapturing.value = false
                        _session.update { it.copy(capturedFiles = it.capturedFiles + filename) }
                        postMessage("Saved: $filename")
                    }
                    CaptureMode.FOCUS_PAIR_FIRST -> {
                        _session.update { it.copy(capturedFiles = it.capturedFiles + filename) }
                        postMessage("Saved: $filename")
                        captureMode = CaptureMode.FOCUS_PAIR_SECOND
                        val s = _session.value
                        cameraController.updateSettings(
                            _settings.value.copy(focusDistance = pendingFocusDistance, afEnabled = false)
                        )
                        val (outputStream, filename2) = photoFileManager.createOutputStream(
                            s.patientId, s.selectedEye, pendingFocusTag
                        )
                        cameraController.captureStillImage(outputStream, filename2)
                    }
                    CaptureMode.FOCUS_PAIR_SECOND -> {
                        captureMode = CaptureMode.NONE
                        _isCapturing.value = false
                        _session.update { it.copy(capturedFiles = it.capturedFiles + filename) }
                        postMessage("Saved: $filename")
                        cameraController.updateSettings(_settings.value)
                    }
                }
            }

            override fun onCaptureError(error: String) {
                val wasInFocusPair = captureMode != CaptureMode.NONE
                captureMode = CaptureMode.NONE
                _isCapturing.value = false
                postMessage(error)
                if (wasInFocusPair) cameraController.updateSettings(_settings.value)
            }
        })
        cameraController.openCamera(textureView)
    }

    fun closeCamera() {
        _isAnalysisRunning.value = false
        analysisJob?.cancel()
        analysisJob = null
        previewTextureView = null
        _ellipseResult.value = null
        cameraController.close()
    }

    fun toggleAnalysis() {
        if (_isAnalysisRunning.value) {
            _isAnalysisRunning.value = false
            analysisJob?.cancel()
            analysisJob = null
            _ellipseResult.value = null
        } else {
            _isAnalysisRunning.value = true
            startAnalysisLoop()
        }
    }

    // --- Photo analysis screen (entered from gallery) ---

    /** Load a saved photo by Uri, then show PhotoAnalysisScreen. */
    fun openPhotoForAnalysis(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val bmp = photoFileManager.loadBitmap(uri)
            if (bmp == null) { postMessage("Failed to load image"); return@launch }
            _capturedBitmap.value = bmp
            _captureAnalysisResult.value = null
            _captureAnalysisAttempted.value = false
            _showAnalysisScreen.value = true
        }
    }

    fun analyzeCapture() {
        val bitmap = _capturedBitmap.value ?: return
        if (_isAnalyzingCapture.value) return
        _isAnalyzingCapture.value = true
        _captureAnalysisResult.value = null
        viewModelScope.launch(Dispatchers.Default) {
            try {
                _captureAnalysisResult.value = ellipseAnalyzer.analyze(bitmap)
            } finally {
                _captureAnalysisAttempted.value = true
                _isAnalyzingCapture.value = false
            }
        }
    }

    fun dismissAnalysisScreen() {
        _showAnalysisScreen.value = false
        _capturedBitmap.value?.recycle()
        _capturedBitmap.value = null
        _captureAnalysisResult.value = null
        _captureAnalysisAttempted.value = false
    }

    // --- Gallery navigation ---

    fun openGallery() {
        refreshPatientList()
        _galleryView.value = GalleryView.PatientList
    }

    fun closeGallery() { _galleryView.value = GalleryView.None }

    fun galleryOpenPatient(patientId: String) {
        _galleryView.value = GalleryView.EyeSelector(patientId)
    }

    fun galleryOpenEye(patientId: String, eye: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _galleryPhotos.value = photoFileManager.listPhotosFor(patientId, eye)
            _galleryView.value = GalleryView.ImageList(patientId, eye)
        }
    }

    fun galleryBack() {
        when (val v = _galleryView.value) {
            is GalleryView.AllAnalyzeResult -> _galleryView.value = GalleryView.ImageList(v.patientId, v.eye)
            is GalleryView.ImageList        -> _galleryView.value = GalleryView.EyeSelector(v.patientId)
            is GalleryView.EyeSelector      -> { refreshPatientList(); _galleryView.value = GalleryView.PatientList }
            GalleryView.PatientList         -> _galleryView.value = GalleryView.None
            GalleryView.None                -> Unit
        }
    }

    private fun refreshPatientList() {
        viewModelScope.launch(Dispatchers.IO) {
            _patientSummaries.value = photoFileManager.listPatientSummaries()
        }
    }

    suspend fun loadThumbnail(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        photoFileManager.loadBitmap(uri, inSampleSize = 8)
    }

    fun runAllAnalyze(patientId: String, eye: String) {
        if (_isRunningAllAnalyze.value) return
        _isRunningAllAnalyze.value = true
        _allAnalyzeProgress.value = 0 to 0
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val photos = photoFileManager.listPhotosFor(patientId, eye)
                _allAnalyzeProgress.value = 0 to photos.size
                if (photos.isEmpty()) { postMessage("No images for $patientId / $eye"); return@launch }
                val samples = mutableListOf<SCAResult.Sample>()
                photos.forEachIndexed { i, photo ->
                    val bmp = photoFileManager.loadBitmap(photo.uri, inSampleSize = 2)
                    if (bmp != null) {
                        val res = ellipseAnalyzer.analyze(bmp)
                        bmp.recycle()
                        val d = res?.dEst
                        if (res != null && d != null) {
                            samples += SCAResult.Sample(res.angleDeg, d)
                        }
                    }
                    _allAnalyzeProgress.value = (i + 1) to photos.size
                }
                val sca = SCAEstimator.fit(samples)
                if (sca == null) {
                    postMessage("Need ≥${SCAEstimator.MIN_VALID} valid D samples (got ${samples.size}/${photos.size})")
                } else {
                    _galleryView.value = GalleryView.AllAnalyzeResult(patientId, eye, sca)
                }
            } finally {
                _isRunningAllAnalyze.value = false
            }
        }
    }

    private fun startAnalysisLoop() {
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch(Dispatchers.Default) {
            delay(800L)
            while (isActive && _isAnalysisRunning.value) {
                try {
                    val bitmap = withContext(Dispatchers.Main) {
                        val tv = previewTextureView
                        val ps = cameraController.getPreviewSize()
                        // GL texture content is portrait-oriented (sensor-rotation-corrected).
                        // Request a portrait-shaped destination so it fits without distortion.
                        if (tv != null && tv.isAvailable) tv.getBitmap(ps.height, ps.width) else null
                    }
                    if (bitmap != null) {
                        val result = ellipseAnalyzer.analyze(bitmap)
                        _ellipseResult.value = result
                        bitmap.recycle()
                    }
                } catch (_: Exception) {}
                delay(200L)
            }
        }
    }

    // --- Settings ---

    fun updateSettings(settings: CameraSettings) {
        _settings.value = settings
        cameraController.updateSettings(settings)
    }

    fun resetSettings() {
        updateSettings(DEFAULT_SETTINGS)
        postMessage("Settings reset to defaults")
    }

    // --- Capture ---

    fun captureImage() {
        val s = _session.value
        if (s.patientId.isBlank()) {
            postMessage("Enter a patient ID before capturing")
            return
        }
        _isCapturing.value = true
        val (outputStream, filename) = photoFileManager.createOutputStream(s.patientId, s.selectedEye)
        cameraController.captureStillImage(outputStream, filename)
    }

    private fun captureFocusPair(focusDistance: Float, tag: String) {
        val s = _session.value
        if (s.patientId.isBlank()) {
            postMessage("Enter a patient ID before capturing")
            return
        }
        _isCapturing.value = true
        pendingFocusDistance = focusDistance
        pendingFocusTag = tag
        captureMode = CaptureMode.FOCUS_PAIR_FIRST
        val (outputStream, filename) = photoFileManager.createOutputStream(s.patientId, s.selectedEye)
        cameraController.captureStillImage(outputStream, filename)
    }

    fun captureFocusPair3D() = captureFocusPair(FOCUS_3D, "3D")
    fun captureFocusPair10D() = captureFocusPair(FOCUS_10D, "10D")

    // --- Session ---

    fun setPatientId(id: String) = _session.update { it.copy(patientId = id) }

    fun setSelectedEye(eye: String) = _session.update { it.copy(selectedEye = eye) }

    fun finishSession() {
        _session.value = SessionState()
        postMessage("Session finished")
    }

    // --- Orientation ---

    fun onOrientationChanged(orientation: DeviceOrientation) {
        if (orientation == DeviceOrientation.OTHER) return
        _currentOrientation.value = orientation
        cameraController.displayRotationDegrees = when (orientation) {
            DeviceOrientation.LANDSCAPE -> 90
            DeviceOrientation.REVERSE_LANDSCAPE -> 270
            else -> 90
        }
        val eye = when (orientation) {
            DeviceOrientation.LANDSCAPE -> "RIGHT"
            DeviceOrientation.REVERSE_LANDSCAPE -> "LEFT"
            else -> return
        }
        _session.update { it.copy(selectedEye = eye) }
    }

    // --- Presets ---

    fun savePreset(index: Int, name: String) {
        viewModelScope.launch {
            presetDataStore.savePreset(index, name, _settings.value)
            postMessage("Preset ${index + 1} saved")
        }
    }

    fun loadPreset(preset: Preset) {
        updateSettings(preset.settings)
        postMessage("Loaded: ${preset.name}")
    }

    fun deletePreset(index: Int) {
        viewModelScope.launch {
            presetDataStore.deletePreset(index)
            postMessage("Preset ${index + 1} deleted")
        }
    }

    // --- Wi-Fi 接続 ---

    fun startBluetoothScan() {
        _btState.value = BtConnectionState.SCANNING
        bluetoothClient.startScan()
    }

    fun stopBluetoothScan() {
        bluetoothClient.stopScan()
        _btState.value = BtConnectionState.DISCONNECTED
    }

    fun disconnectBluetooth() {
        bluetoothClient.disconnect()
        _btState.value = BtConnectionState.DISCONNECTED
        _isRemoteRecording.value = false
    }

    fun sendBalloonMove(dx: Int, dy: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("BALLOON:$dx:$dy")
        }
    }

    fun sendVideoStart() {
        val patientId = _session.value.patientId
        if (patientId.isBlank()) {
            postMessage("Enter a Patient ID before starting remote recording.")
            return
        }
        val eye = _session.value.selectedEye   // "RIGHT" or "LEFT"
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("VIDEO_START:$patientId:$eye")
        }
        _isRemoteRecording.value = true
    }

    fun sendVideoStop() {
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("VIDEO_STOP")
        }
        _isRemoteRecording.value = false
    }

    fun sendBalloonReset() {
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("BALLOON_RESET")
        }
    }

    fun sendBalloonSizeChange(delta: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("BALLOON_SIZE:$delta")
        }
    }

    /** プリセット適用コマンドを送信する（n: 1〜4, 1-indexed） */
    fun sendPreset(n: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            bluetoothClient.sendCommand("PRESET:$n")
        }
    }

    // --- Helpers ---

    fun clearMessage() { _message.value = null }

    private fun postMessage(text: String) {
        _message.value = UiMessage(text)
    }

    override fun onCleared() {
        super.onCleared()
        bluetoothClient.disconnect()
    }
}
