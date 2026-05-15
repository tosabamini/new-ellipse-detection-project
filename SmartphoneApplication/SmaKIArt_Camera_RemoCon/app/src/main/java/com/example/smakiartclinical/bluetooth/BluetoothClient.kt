package com.example.smakiartclinical.bluetooth

import android.annotation.SuppressLint
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.util.UUID

/**
 * BLE GATT クライアント。
 * Android 11: ACCESS_FINE_LOCATION + BLUETOOTH が必要（manifest で宣言済み）。
 * Android 12+: BLUETOOTH_SCAN + BLUETOOTH_CONNECT が必要。
 *
 * 接続フロー:
 *   connectGatt → STATE_CONNECTED → refreshCache() → requestMtu(512)
 *   → onMtuChanged → discoverServices() → onServicesDiscovered → 完了
 *
 * refreshCache() はリフレクション経由で BluetoothGatt.refresh() を呼び出す。
 * これにより Android の GATT サービスキャッシュをクリアし、
 * 同一デバイスへの再接続時に古い（空の）サービスリストが返るのを防ぐ。
 */
class BluetoothClient(private val context: Context) {

    companion object {
        val SERVICE_UUID: UUID = UUID.fromString("e0cb5db0-afac-11de-8a39-0800200c9a66")
        val COMMAND_CHAR_UUID: UUID = UUID.fromString("e0cb5db1-afac-11de-8a39-0800200c9a66")
        private const val SCAN_TIMEOUT_MS = 20_000L
    }

    var lastError: String = ""

    /** NSD 解決済み → 接続開始直前（CONNECTING 状態に遷移させるために使用） */
    var onConnecting: (() -> Unit)? = null
    /** GATT 接続・サービス検出完了 */
    var onConnected: (() -> Unit)? = null
    /** 切断またはエラー */
    var onDisconnected: (() -> Unit)? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private var gatt: BluetoothGatt? = null
    private var commandChar: BluetoothGattCharacteristic? = null
    private var leScanner: android.bluetooth.le.BluetoothLeScanner? = null
    @Volatile private var scanning = false

    // ── スキャン ──────────────────────────────────────────────────────────────

    @SuppressLint("MissingPermission")
    fun startScan() {
        if (scanning || gatt != null) return
        val adapter = (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
        leScanner = adapter.bluetoothLeScanner ?: run {
            lastError = "BLE スキャナーが利用できません"
            mainHandler.post { onDisconnected?.invoke() }
            return
        }
        scanning = true
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        try {
            // フィルターなしでスキャン → onScanResult 内で UUID を手動確認
            leScanner!!.startScan(null, settings, scanCallback)
        } catch (e: SecurityException) {
            scanning = false
            lastError = "スキャン権限エラー: ${e.message}"
            mainHandler.post { onDisconnected?.invoke() }
            return
        }

        // タイムアウト
        mainHandler.postDelayed({
            if (scanning) {
                stopScan()
                lastError = "Screen_FrontCamera が見つかりませんでした（20秒タイムアウト）"
                onDisconnected?.invoke()
            }
        }, SCAN_TIMEOUT_MS)
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        if (!scanning) return
        scanning = false
        try { leScanner?.stopScan(scanCallback) } catch (_: Exception) {}
        leScanner = null
    }

    // ── コマンド送信 ──────────────────────────────────────────────────────────

    @SuppressLint("MissingPermission")
    fun sendCommand(command: String) {
        val char = commandChar ?: return
        val g = gatt ?: return
        val bytes = command.toByteArray(Charsets.UTF_8)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            g.writeCharacteristic(char, bytes, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)
        } else {
            @Suppress("DEPRECATION")
            char.value = bytes
            @Suppress("DEPRECATION")
            g.writeCharacteristic(char)
        }
    }

    // ── 切断 ──────────────────────────────────────────────────────────────────

    @SuppressLint("MissingPermission")
    fun disconnect() {
        stopScan()
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        commandChar = null
    }

    val isConnected: Boolean get() = commandChar != null

    // ── GATTキャッシュリフレッシュ ────────────────────────────────────────────

    /**
     * Android の GATT サービスキャッシュをクリアする（リフレクション使用）。
     * 同一デバイスへの再接続時にキャッシュされた古いサービスリストが
     * onServicesDiscovered で返るのを防ぐ。
     * 失敗しても接続フローは継続する（ベストエフォート）。
     */
    private fun BluetoothGatt.refreshCache(): Boolean = try {
        val method = javaClass.getMethod("refresh")
        method.invoke(this) as? Boolean ?: false
    } catch (_: Exception) { false }

    // ── コールバック ──────────────────────────────────────────────────────────

    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            // サービス UUID を手動確認（フィルター不使用のため）
            val uuids = result.scanRecord?.serviceUuids
            if (uuids == null || uuids.none { it.uuid == SERVICE_UUID }) return
            stopScan()
            mainHandler.post { onConnecting?.invoke() }
            gatt = result.device.connectGatt(
                context, false, gattCallback, android.bluetooth.BluetoothDevice.TRANSPORT_LE
            )
        }

        override fun onScanFailed(errorCode: Int) {
            scanning = false
            lastError = "BLE スキャン失敗 errorCode=$errorCode"
            mainHandler.post { onDisconnected?.invoke() }
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    // キャッシュをクリアしてから MTU 交渉 → サービス探索へ
                    gatt.refreshCache()
                    gatt.requestMtu(512)
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    commandChar = null
                    this@BluetoothClient.gatt?.close()
                    this@BluetoothClient.gatt = null
                    mainHandler.post { onDisconnected?.invoke() }
                }
            }
        }

        @SuppressLint("MissingPermission")
        override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
            // status に関わらずサービス探索を進める（MTU 失敗でも接続は有効）
            gatt.discoverServices()
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                lastError = "サービス検出失敗: $status"
                mainHandler.post { onDisconnected?.invoke() }
                return
            }
            val char = gatt.getService(SERVICE_UUID)?.getCharacteristic(COMMAND_CHAR_UUID)
            if (char == null) {
                lastError = "コマンド特性が見つかりません（Screen_FrontCamera が起動しているか確認）"
                mainHandler.post { onDisconnected?.invoke() }
                return
            }
            commandChar = char
            mainHandler.post { onConnected?.invoke() }
        }
    }
}
