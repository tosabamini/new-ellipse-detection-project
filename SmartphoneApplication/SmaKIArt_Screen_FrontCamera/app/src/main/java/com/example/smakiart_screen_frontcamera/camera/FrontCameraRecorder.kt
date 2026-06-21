package com.example.smakiart_screen_frontcamera.camera

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Size
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FrontCameraRecorder(private val context: Context) {

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val backgroundThread = HandlerThread("FrontCameraBackground").also { it.start() }
    private val backgroundHandler = Handler(backgroundThread.looper)

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var mediaRecorder: MediaRecorder? = null
    private var pendingUri: Uri? = null
    private var pendingPfd: ParcelFileDescriptor? = null

    private fun findFrontCamera(): String? =
        cameraManager.cameraIdList.firstOrNull { id ->
            cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
        }

    private fun bestVideoSize(cameraId: String): Size {
        val map = cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = map?.getOutputSizes(MediaRecorder::class.java) ?: return Size(1280, 720)
        return sizes
            .filter { it.width <= 1280 && it.height <= 720 }
            .maxByOrNull { it.width * it.height }
            ?: sizes.minByOrNull { it.width * it.height }
            ?: Size(1280, 720)
    }

    @SuppressLint("MissingPermission")
    fun openCamera(onReady: () -> Unit, onError: (String) -> Unit) {
        val cameraId = findFrontCamera() ?: run { onError("フロントカメラが見つかりません"); return }
        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                onReady()
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close(); cameraDevice = null }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
                onError("カメラエラー: $error")
            }
        }, backgroundHandler)
    }

    fun startRecording(patientId: String, eye: String = "RIGHT") {
        val device = cameraDevice ?: return
        val cameraId = findFrontCamera() ?: return
        val videoSize = bestVideoSize(cameraId)

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val eyeTag    = if (eye == "LEFT") "L" else "R"
        val filename  = "${patientId}_${eyeTag}_${timestamp}.mp4"

        // minSdk=28 のため常に MediaRecorder(context) を使用可能（API 31 推奨版）
        val recorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) MediaRecorder(context)
                        else @Suppress("DEPRECATION") MediaRecorder()

        recorder.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(videoSize.width, videoSize.height)
            setVideoFrameRate(30)
            setVideoEncodingBitRate(3_000_000)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, filename)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/SmaKIArtClinical")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                ?: return
            val pfd = context.contentResolver.openFileDescriptor(uri, "w") ?: return
            pendingUri = uri
            pendingPfd = pfd
            recorder.setOutputFile(pfd.fileDescriptor)
        } else {
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                "SmaKIArtClinical"
            ).also { it.mkdirs() }
            recorder.setOutputFile(File(dir, filename).absolutePath)
        }

        try {
            recorder.prepare()
        } catch (e: Exception) {
            recorder.release()
            return
        }
        mediaRecorder = recorder

        // SessionConfiguration を使用（minSdk=28 以上で常に利用可能、旧 API は API 30 で deprecated）
        val sessionConfig = SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            listOf(OutputConfiguration(recorder.surface)),
            { backgroundHandler.post(it) },
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                        addTarget(recorder.surface)
                    }
                    session.setRepeatingRequest(builder.build(), null, backgroundHandler)
                    recorder.start()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    recorder.release()
                    mediaRecorder = null
                }
            }
        )
        device.createCaptureSession(sessionConfig)
    }

    fun stopRecording() {
        captureSession?.close()
        captureSession = null

        try { mediaRecorder?.stop() } catch (_: Exception) {}
        mediaRecorder?.release()
        mediaRecorder = null

        pendingPfd?.close()
        pendingPfd = null

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            pendingUri?.let { uri ->
                context.contentResolver.update(
                    uri,
                    ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) },
                    null, null
                )
            }
        }
        pendingUri = null
    }

    fun close() {
        stopRecording()
        cameraDevice?.close()
        cameraDevice = null
        backgroundThread.quitSafely()
    }
}
