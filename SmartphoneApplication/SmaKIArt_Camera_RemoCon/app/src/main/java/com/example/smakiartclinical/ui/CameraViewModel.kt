package com.example.smakiartclinical.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.view.TextureView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.smakiartclinical.analysis.EllipseAnalyzer
import com.example.smakiartclinical.analysis.EllipseResult
import com.example.smakiartclinical.bluetooth.BluetoothClient
import com.example.smakiartclinical.camera.CameraController
import com.example.smakiartclinical.camera.DeviceOrientation
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

    // ── Photo analysis screen ─────────────────────────────────────────────────
    @Volatile private var lastCapturedBytes: ByteArray? = null
    @Volatile private var lastCapturedOrientationDeg: Int = 0

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
                        // Navigate to photo analysis screen
                        lastCapturedBytes = jpegBytes
                        lastCapturedOrientationDeg = jpegOrientationDeg
                        _capturedBitmap.value = null
                        _captureAnalysisResult.value = null
                        _captureAnalysisAttempted.value = false
                        _showAnalysisScreen.value = true
                        loadCapturedBitmapForDisplay(jpegBytes, jpegOrientationDeg)
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

    // --- Photo analysis screen ---

    private fun loadCapturedBitmapForDisplay(bytes: ByteArray, orientationDeg: Int) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val opts = BitmapFactory.Options().apply { inSampleSize = 2 }
                var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return@launch
                if (orientationDeg != 0) {
                    val m = Matrix(); m.postRotate(orientationDeg.toFloat())
                    val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                    bmp.recycle(); bmp = rotated
                }
                _capturedBitmap.value = bmp
            } catch (_: Exception) {}
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
        _capturedBitmap.value = null
        _captureAnalysisResult.value = null
        _captureAnalysisAttempted.value = false
    }

    private fun startAnalysisLoop() {
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch(Dispatchers.Default) {
            delay(800L)
            while (isActive && _isAnalysisRunning.value) {
                try {
                    val bitmap = withContext(Dispatchers.Main) {
                        val tv = previewTextureView
                        if (tv != null && tv.isAvailable) tv.bitmap else null
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
