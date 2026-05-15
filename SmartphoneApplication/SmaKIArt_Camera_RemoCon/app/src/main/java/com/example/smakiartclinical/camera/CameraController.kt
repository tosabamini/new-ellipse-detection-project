package com.example.smakiartclinical.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.SurfaceTexture
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
    private var previewSize: android.util.Size = android.util.Size(1920, 1080)

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

    fun getPreviewSize(): android.util.Size = previewSize

    @SuppressLint("MissingPermission")
    fun openCamera(textureView: TextureView) {
        previewTextureView = textureView

        val cameraId = findBackFacingCamera() ?: run {
            listener?.onCameraError("No back-facing camera found")
            return
        }
        characteristics = cameraManager.getCameraCharacteristics(cameraId)

        // Choose preview size from camera capabilities, then declare it to the
        // SurfaceTexture BEFORE wrapping it in a Surface. This ensures the camera
        // driver knows the requested buffer dimensions and getBitmap() returns
        // content at a well-defined resolution.
        characteristics!!
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputSizes(SurfaceTexture::class.java)
            ?.let { sizes ->
                previewSize = choosePreviewSize(sizes, textureView.width, textureView.height)
            }
        textureView.surfaceTexture?.setDefaultBufferSize(previewSize.width, previewSize.height)

        previewSurface?.release()
        previewSurface = Surface(textureView.surfaceTexture)

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

    private fun choosePreviewSize(
        choices: Array<android.util.Size>, viewW: Int, viewH: Int
    ): android.util.Size {
        // Target: closest aspect ratio to the view, longer side ≤ 1920 px.
        val targetAspect = if (viewW > 0 && viewH > 0) viewW.toFloat() / viewH else 16f / 9f
        return choices
            .filter { maxOf(it.width, it.height) <= 1920 }
            .minByOrNull { kotlin.math.abs(it.width.toFloat() / it.height.toFloat() - targetAspect) }
            ?: choices.minByOrNull { it.width * it.height }
            ?: android.util.Size(1920, 1080)
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
     * The camera delivers frames in sensor-native orientation (rotated 90° or 270° from the
     * display). The standard Camera2 pattern swaps the preview dimensions in bufferRect to
     * model that rotated footprint, then postScale fills the view (centre-crop, no black bars),
     * then postRotate corrects the sensor angle.
     *
     *   displayRotationDegrees = 90  (normal landscape)    → postRotate(-90°)
     *   displayRotationDegrees = 270 (reverse landscape)   → postRotate(+90°)
     */
    private fun applyPreviewTransform() {
        val tv = previewTextureView ?: return
        if (tv.width == 0 || tv.height == 0) return

        val viewW = tv.width.toFloat()
        val viewH = tv.height.toFloat()
        val centerX = viewW / 2f
        val centerY = viewH / 2f

        // Swap previewSize dims to represent the sensor-native (portrait) content footprint.
        val viewRect   = android.graphics.RectF(0f, 0f, viewW, viewH)
        val bufferRect = android.graphics.RectF(
            0f, 0f,
            previewSize.height.toFloat(),
            previewSize.width.toFloat()
        )
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())

        val matrix = Matrix()
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)

        // Scale to fill the view (centre-crop): no black bars on either axis.
        val scale = maxOf(viewH / previewSize.height, viewW / previewSize.width)
        matrix.postScale(scale, scale, centerX, centerY)

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
