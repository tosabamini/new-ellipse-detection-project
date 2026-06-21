package com.example.smakiart_screen_frontcamera.bluetooth

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattServer
import android.bluetooth.BluetoothGattServerCallback
import android.bluetooth.BluetoothGattService
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import java.util.UUID

/**
 * BLE GATT サーバー。
 * SERVICE_UUID をアドバタイズし、Camera_RemoCon からの WRITE を受け付ける。
 *
 * 重要: addService() は非同期。onServiceAdded() コールバック後にアドバタイズを
 * 開始することで、クライアントが接続時にサービス/特性を確実に発見できるようにする。
 *
 * Android 12+: BLUETOOTH_ADVERTISE + BLUETOOTH_CONNECT が必要（manifest で宣言済み）。
 */
class BluetoothServer(private val context: Context) {

    companion object {
        val SERVICE_UUID: UUID = UUID.fromString("e0cb5db0-afac-11de-8a39-0800200c9a66")
        val COMMAND_CHAR_UUID: UUID = UUID.fromString("e0cb5db1-afac-11de-8a39-0800200c9a66")
    }

    interface CommandListener {
        fun onConnected()
        fun onDisconnected()
        fun onBalloonMove(dx: Int, dy: Int)
        fun onBalloonReset()
        fun onBalloonSizeChange(delta: Int)
        fun onPresetApply(presetNumber: Int)   // 1-indexed
        fun onVideoStart(patientId: String, eye: String)
        fun onVideoStop()
        fun onFog()
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private var commandListener: CommandListener? = null
    private var gattServer: BluetoothGattServer? = null
    private var bluetoothAdapter: BluetoothAdapter? = null
    private var advertiser: android.bluetooth.le.BluetoothLeAdvertiser? = null

    var onAdvertisingStarted: (() -> Unit)? = null
    var onAdvertisingFailed: ((Int) -> Unit)? = null

    fun setCommandListener(l: CommandListener) { commandListener = l }

    @SuppressLint("MissingPermission")
    fun start() {
        val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter

        gattServer = bluetoothManager.openGattServer(context, gattServerCallback)

        val service = BluetoothGattService(SERVICE_UUID, BluetoothGattService.SERVICE_TYPE_PRIMARY)
        val characteristic = BluetoothGattCharacteristic(
            COMMAND_CHAR_UUID,
            BluetoothGattCharacteristic.PROPERTY_WRITE,
            BluetoothGattCharacteristic.PERMISSION_WRITE
        )
        service.addCharacteristic(characteristic)

        // addService() は非同期 → 完了は onServiceAdded() で受け取り、そこでアドバタイズ開始
        gattServer?.addService(service)
    }

    @SuppressLint("MissingPermission")
    private fun startAdvertising() {
        val adapter = bluetoothAdapter ?: return
        advertiser = adapter.bluetoothLeAdvertiser
        if (advertiser == null) {
            mainHandler.post { onAdvertisingFailed?.invoke(-1) }
            return
        }

        val settings = AdvertiseSettings.Builder()
            .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
            .setConnectable(true)
            .setTimeout(0)
            .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
            .build()

        val data = AdvertiseData.Builder()
            .setIncludeDeviceName(false)
            .addServiceUuid(ParcelUuid(SERVICE_UUID))
            .build()

        try {
            advertiser!!.startAdvertising(settings, data, advertiseCallback)
        } catch (e: SecurityException) {
            mainHandler.post { onAdvertisingFailed?.invoke(-2) }
        }
    }

    @SuppressLint("MissingPermission")
    fun stop() {
        try { advertiser?.stopAdvertising(advertiseCallback) } catch (_: Exception) {}
        try { gattServer?.close() } catch (_: Exception) {}
        advertiser = null
        gattServer = null
        bluetoothAdapter = null
    }

    // ── アドバタイズコールバック ──────────────────────────────────────────────

    private val advertiseCallback = object : AdvertiseCallback() {
        override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
            mainHandler.post { onAdvertisingStarted?.invoke() }
        }
        override fun onStartFailure(errorCode: Int) {
            mainHandler.post { onAdvertisingFailed?.invoke(errorCode) }
        }
    }

    // ── GATT サーバーコールバック ─────────────────────────────────────────────

    private val gattServerCallback = object : BluetoothGattServerCallback() {

        override fun onServiceAdded(status: Int, service: BluetoothGattService) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                startAdvertising()
            } else {
                mainHandler.post { onAdvertisingFailed?.invoke(-3) }
            }
        }

        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(device: BluetoothDevice, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED    -> mainHandler.post { commandListener?.onConnected() }
                BluetoothProfile.STATE_DISCONNECTED -> mainHandler.post { commandListener?.onDisconnected() }
            }
        }

        @SuppressLint("MissingPermission")
        override fun onCharacteristicWriteRequest(
            device: BluetoothDevice,
            requestId: Int,
            characteristic: BluetoothGattCharacteristic,
            preparedWrite: Boolean,
            responseNeeded: Boolean,
            offset: Int,
            value: ByteArray
        ) {
            if (responseNeeded) {
                gattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, 0, null)
            }
            parseAndDispatch(value.toString(Charsets.UTF_8))
        }
    }

    // ── コマンド解析・ディスパッチ ────────────────────────────────────────────

    private fun parseAndDispatch(command: String) {
        when {
            command.startsWith("BALLOON:") -> {
                val parts = command.removePrefix("BALLOON:").split(":")
                if (parts.size == 2) {
                    val dx = parts[0].toIntOrNull() ?: 0
                    val dy = parts[1].toIntOrNull() ?: 0
                    mainHandler.post { commandListener?.onBalloonMove(dx, dy) }
                }
            }
            command == "BALLOON_RESET" ->
                mainHandler.post { commandListener?.onBalloonReset() }
            command.startsWith("BALLOON_SIZE:") -> {
                val delta = command.removePrefix("BALLOON_SIZE:").toIntOrNull() ?: 0
                mainHandler.post { commandListener?.onBalloonSizeChange(delta) }
            }
            command.startsWith("PRESET:") -> {
                // Format: PRESET:N  (N = 1〜4, 1-indexed)
                val n = command.removePrefix("PRESET:").toIntOrNull() ?: return
                if (n in 1..4) mainHandler.post { commandListener?.onPresetApply(n) }
            }
            command.startsWith("VIDEO_START:") -> {
                val parts     = command.removePrefix("VIDEO_START:").split(":")
                val patientId = parts.getOrElse(0) { "" }
                val eye       = parts.getOrElse(1) { "RIGHT" }
                mainHandler.post { commandListener?.onVideoStart(patientId, eye) }
            }
            command == "VIDEO_STOP" ->
                mainHandler.post { commandListener?.onVideoStop() }
            command == "FOG" ->
                mainHandler.post { commandListener?.onFog() }
        }
    }
}
