package com.example.smakiartclinical.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.CameraCaptureSession
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.view.Surface
import android.view.TextureView
import com.example.smakiartclinical.data.model.CameraSettings
import java.io.OutputStream

class CameraController(private val context: Context) {

    interface Listener {
        fun onCameraOpened()
        fun onCameraError(error: String)
        fun onCaptureSaved(filename: String, jpegBytes: ByteArray, jpegOrientationDeg: Int)
        fun onCaptureError(error: String)
    }

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewSurface: Surface? = null
    private var previewTextureView: TextureView? = null
    private var imageReader: ImageReader? = null
    private var characteristics: CameraCharacteristics? = null

    private val backgroundThread = HandlerThread("CameraBackground").also { it.start() }
    private val backgroundHandler = Handler(backgroundThread.looper)
    private val mainHandler = Handler(Looper.getMainLooper())

    private var retryCount = 0

    companion object {
        private const val MAX_PREVIEW_EXPOSURE_NS = 200_000_000L  // 200 ms
        private const val MAX_CAMERA_RETRIES = 2
        private const val RETRY_DELAY_MS = 800L
    }

    private var listener: Listener? = null
    private var currentSettings = CameraSettings()

    /**
     * Degrees the display is rotated from its natural orientation (0, 90, 180, 270).
     * Updating this re-applies the preview transform immediately.
     */
    var displayRotationDegrees: Int = 90
        set(value) {
            field = value
            applyPreviewTransform()
        }

    fun setListener(l: Listener) { listener = l }

    fun getCharacteristics(): CameraCharacteristics? = characteristics

    @SuppressLint("MissingPermission")
    fun openCamera(textureView: TextureView) {
        previewTextureView = textureView
        previewSurface?.release()
        previewSurface = Surface(textureView.surfaceTexture)

        val cameraId = findBackFacingCamera() ?: run {
            listener?.onCameraError("No back-facing camera found")
            return
        }
        characteristics = cameraManager.getCameraCharacteristics(cameraId)

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                retryCount = 0
                cameraDevice = camera
                setupImageReader()
                createCaptureSession()
            }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
            }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
                val tv = previewTextureView
                if (retryCount < MAX_CAMERA_RETRIES && tv != null) {
                    retryCount++
                    mainHandler.postDelayed({ openCamera(tv) }, RETRY_DELAY_MS)
                } else {
                    retryCount = 0
                    listener?.onCameraError("Camera error: $error")
                }
            }
        }, backgroundHandler)
    }

    private fun findBackFacingCamera(): String? {
        return cameraManager.cameraIdList.firstOrNull { id ->
            cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
    }

    private fun setupImageReader() {
        val chars = characteristics ?: return
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
        val largest = map.getOutputSizes(ImageFormat.JPEG).maxByOrNull { it.width * it.height } ?: return
        imageReader = ImageReader.newInstance(largest.width, largest.height, ImageFormat.JPEG, 2)
    }

    private fun createCaptureSession() {
        val device = cameraDevice ?: return
        val preview = previewSurface ?: return
        val reader = imageReader ?: return

        device.createCaptureSession(
            listOf(preview, reader.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    listener?.onCameraOpened()
                    startPreview()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    listener?.onCameraError("Session configuration failed")
                }
            },
            backgroundHandler
        )
    }

    fun startPreview() {
        val session = captureSession ?: return
        val device = cameraDevice ?: return
        val preview = previewSurface ?: return
        val chars = characteristics ?: return

        applyPreviewTransform()

        val previewSettings = currentSettings.copy(
            exposureTimeNs = currentSettings.exposureTimeNs.coerceAtMost(MAX_PREVIEW_EXPOSURE_NS)
        )

        val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(preview)
            ManualCameraSettingsApplier.apply(this, previewSettings, chars)
        }
        session.setRepeatingRequest(builder.build(), null, backgroundHandler)
    }

    /**
     * Applies a TextureView transform so the preview appears upright on screen.
     *
     * The aspect-ratio swap (bufferRect has swapped w/h) combined with setRectToRect corrects
     * for the camera sensor delivering portrait-oriented frames into a landscape-sized surface.
     * postRotate then spins the result so it reads naturally.
     *
     * With sensorLandscape, the window physically rotates between ROTATION_90 and ROTATION_270.
     * The TextureView lives inside that window, so its effective rendering flips 180° when the
     * window does. We compensate by inverting the rotation sign in reverse landscape:
     *
     *   displayRotationDegrees = 90  (normal landscape)    → postRotate(-90°)
     *   displayRotationDegrees = 270 (reverse landscape)   → postRotate(+90°)
     *
     * The sign flip is exactly the 180° needed to cancel the window-level rotation.
     */
    private fun applyPreviewTransform() {
        val tv = previewTextureView ?: return
        if (tv.width == 0 || tv.height == 0) return

        val viewRect = android.graphics.RectF(0f, 0f, tv.width.toFloat(), tv.height.toFloat())
        val bufferRect = android.graphics.RectF(0f, 0f, tv.height.toFloat(), tv.width.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())

        val matrix = Matrix()
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)

        val rotateDeg = if (displayRotationDegrees == 270) 90f else -90f
        matrix.postRotate(rotateDeg, centerX, centerY)

        tv.setTransform(matrix)
    }

    fun updateSettings(settings: CameraSettings) {
        currentSettings = settings
        startPreview()
    }

    fun captureStillImage(outputStream: OutputStream, filename: String) {
        val session = captureSession ?: run {
            listener?.onCaptureError("No active session")
            return
        }
        val device = cameraDevice ?: run {
            listener?.onCaptureError("Camera not open")
            return
        }
        val reader = imageReader ?: run {
            listener?.onCaptureError("ImageReader not ready")
            return
        }
        val chars = characteristics ?: return

        val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val jpegOrientation = (sensorOrientation - displayRotationDegrees + 360) % 360

        reader.setOnImageAvailableListener({ r ->
            val image = r.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                outputStream.write(bytes)
                outputStream.flush()
                outputStream.close()
                listener?.onCaptureSaved(filename, bytes, jpegOrientation)
            } catch (e: Exception) {
                listener?.onCaptureError("Save failed: ${e.message}")
            } finally {
                image.close()
            }
        }, backgroundHandler)

        val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)
            ManualCameraSettingsApplier.apply(this, currentSettings, chars)
            set(CaptureRequest.JPEG_ORIENTATION, jpegOrientation)
        }

        session.capture(builder.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) { }
        }, backgroundHandler)
    }

    fun close() {
        captureSession?.close()
        captureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
        previewTextureView = null
        backgroundThread.quitSafely()
    }
}
