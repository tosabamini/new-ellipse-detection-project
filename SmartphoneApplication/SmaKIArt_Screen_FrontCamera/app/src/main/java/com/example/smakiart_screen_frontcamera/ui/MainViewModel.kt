package com.example.smakiart_screen_frontcamera.ui

import android.app.Application
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import com.example.smakiart_screen_frontcamera.bluetooth.BluetoothServer
import com.example.smakiart_screen_frontcamera.camera.FrontCameraRecorder
import com.example.smakiart_screen_frontcamera.data.BalloonPreset
import com.example.smakiart_screen_frontcamera.data.BalloonPresetStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val bluetoothServer    = BluetoothServer(application)
    private val frontCameraRecorder = FrontCameraRecorder(application)
    private val presetStore        = BalloonPresetStore(application)

    private val _balloonOffset      = MutableStateFlow(Offset.Zero)
    val balloonOffset: StateFlow<Offset> = _balloonOffset.asStateFlow()

    private val _isRecording        = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isConnected        = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected.asStateFlow()

    private val _patientId          = MutableStateFlow("")
    val patientId: StateFlow<String> = _patientId.asStateFlow()

    private val _cameraReady        = MutableStateFlow(false)
    val cameraReady: StateFlow<Boolean> = _cameraReady.asStateFlow()

    private val _advertisingState   = MutableStateFlow<Boolean?>(null)
    val advertisingState: StateFlow<Boolean?> = _advertisingState.asStateFlow()

    private val _advertisingError   = MutableStateFlow(0)
    val advertisingError: StateFlow<Int> = _advertisingError.asStateFlow()

    private val _isReverseLandscape = MutableStateFlow(false)
    val isReverseLandscape: StateFlow<Boolean> = _isReverseLandscape.asStateFlow()

    // balloon display size in dp (min 50, max 250, default 110)
    private val _balloonSizeDp      = MutableStateFlow(110)
    val balloonSizeDp: StateFlow<Int> = _balloonSizeDp.asStateFlow()

    // 4 presets (0-indexed list; null = empty slot)
    private val _presets            = MutableStateFlow<List<BalloonPreset?>>(List(4) { null })
    val presets: StateFlow<List<BalloonPreset?>> = _presets.asStateFlow()

    // 雲霧（フォグ）トリガー。インクリメントするたびに MainScreen 側で
    // ブラーアニメーションが 1 回再生される（ローカルボタン／BLE FOG の両方から発火）。
    private val _fogTrigger         = MutableStateFlow(0)
    val fogTrigger: StateFlow<Int> = _fogTrigger.asStateFlow()

    init {
        // ── プリセット読み込み & 起動時プリセット 1 を自動適用 ──────────────
        val loaded = presetStore.loadAll()   // 0-indexed list
        _presets.value = loaded
        loaded[0]?.let { p ->               // プリセット 1 (index 0) が保存済みなら適用
            _balloonOffset.value = Offset(p.offsetX, p.offsetY)
            _balloonSizeDp.value = p.sizeDp
        }

        // ── BLE アドバタイズコールバック ───────────────────────────────────
        bluetoothServer.onAdvertisingStarted = { _advertisingState.value = true }
        bluetoothServer.onAdvertisingFailed  = { errorCode ->
            _advertisingState.value = false
            _advertisingError.value = errorCode
        }

        // ── BLE コマンドリスナー ───────────────────────────────────────────
        bluetoothServer.setCommandListener(object : BluetoothServer.CommandListener {
            override fun onConnected()    { _isConnected.value = true }
            override fun onDisconnected() { _isConnected.value = false }

            override fun onBalloonMove(dx: Int, dy: Int) {
                _balloonOffset.update { Offset(it.x + dx, it.y + dy) }
            }
            override fun onBalloonReset() {
                _balloonOffset.value = Offset.Zero
            }
            override fun onBalloonSizeChange(delta: Int) {
                _balloonSizeDp.value = (_balloonSizeDp.value + delta).coerceIn(50, 250)
            }
            override fun onPresetApply(presetNumber: Int) {
                applyPreset(presetNumber - 1)   // BLE は 1-indexed → 0-indexed に変換
            }
            override fun onVideoStart(patientId: String, eye: String) {
                _patientId.value = patientId
                frontCameraRecorder.startRecording(patientId, eye)
                _isRecording.value = true
            }
            override fun onVideoStop() {
                frontCameraRecorder.stopRecording()
                _isRecording.value = false
            }
            override fun onFog() {
                triggerFog()
            }
        })
        bluetoothServer.start()

        frontCameraRecorder.openCamera(
            onReady = { _cameraReady.value = true },
            onError = { _cameraReady.value = false }
        )
    }

    // ── 気球操作 ──────────────────────────────────────────────────────────────

    fun moveBalloon(dx: Int, dy: Int) { _balloonOffset.update { Offset(it.x + dx, it.y + dy) } }
    fun resetBalloon()                 { _balloonOffset.value = Offset.Zero }
    fun setPatientId(id: String)       { _patientId.value = id }

    fun setReverseLandscape(value: Boolean) { _isReverseLandscape.value = value }

    /** 雲霧（フォグ）演出を 1 回トリガーする。ローカルボタン／BLE FOG コマンド共用。 */
    fun triggerFog() { _fogTrigger.update { it + 1 } }

    // ── プリセット操作 ────────────────────────────────────────────────────────

    /** 現在の気球位置・サイズを index（0-indexed）のスロットに保存する */
    fun savePreset(index: Int) {
        val preset = BalloonPreset(
            offsetX = _balloonOffset.value.x,
            offsetY = _balloonOffset.value.y,
            sizeDp  = _balloonSizeDp.value
        )
        presetStore.save(index + 1, preset)          // store は 1-indexed
        _presets.value = _presets.value.toMutableList().also { it[index] = preset }
    }

    /** index（0-indexed）のプリセットを適用する。未保存なら何もしない。 */
    fun applyPreset(index: Int) {
        if (index !in 0..3) return
        val preset = _presets.value[index] ?: return
        _balloonOffset.value = Offset(preset.offsetX, preset.offsetY)
        _balloonSizeDp.value = preset.sizeDp
    }

    // ── ローカル録画 ──────────────────────────────────────────────────────────

    fun startLocalRecording() {
        val id = _patientId.value.ifBlank { "unknown" }
        frontCameraRecorder.startRecording(id, "LOCAL")
        _isRecording.value = true
    }

    fun stopLocalRecording() {
        frontCameraRecorder.stopRecording()
        _isRecording.value = false
    }

    override fun onCleared() {
        super.onCleared()
        bluetoothServer.stop()
        frontCameraRecorder.close()
    }
}
