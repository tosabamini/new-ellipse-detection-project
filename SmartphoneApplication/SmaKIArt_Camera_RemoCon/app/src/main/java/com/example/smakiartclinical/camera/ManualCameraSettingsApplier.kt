package com.example.smakiartclinical.camera

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import com.example.smakiartclinical.data.model.CameraSettings

object ManualCameraSettingsApplier {

    fun apply(
        builder: CaptureRequest.Builder,
        settings: CameraSettings,
        characteristics: CameraCharacteristics
    ) {
        // AE
        builder.set(
            CaptureRequest.CONTROL_AE_MODE,
            if (settings.aeEnabled) CameraMetadata.CONTROL_AE_MODE_ON
            else CameraMetadata.CONTROL_AE_MODE_OFF
        )
        if (!settings.aeEnabled) {
            val clampedIso = clampIso(settings.iso, characteristics)
            val clampedExp = clampExposureTime(settings.exposureTimeNs, characteristics)
            builder.set(CaptureRequest.SENSOR_SENSITIVITY, clampedIso)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, clampedExp)
        } else {
            builder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, settings.exposureCompensation)
        }

        // AF
        builder.set(
            CaptureRequest.CONTROL_AF_MODE,
            if (settings.afEnabled) CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            else CameraMetadata.CONTROL_AF_MODE_OFF
        )
        if (!settings.afEnabled) {
            val clampedFocus = clampFocusDistance(settings.focusDistance, characteristics)
            builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, clampedFocus)
        }

        // AWB — always automatic
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)

        // Disable CONTROL_MODE overriding our manual settings
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
    }

    private fun clampIso(iso: Int, c: CameraCharacteristics): Int {
        val range = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE) ?: return iso
        return iso.coerceIn(range.lower, range.upper)
    }

    private fun clampExposureTime(ns: Long, c: CameraCharacteristics): Long {
        val range = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE) ?: return ns
        return ns.coerceIn(range.lower, range.upper)
    }

    private fun clampFocusDistance(distance: Float, c: CameraCharacteristics): Float {
        val max = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: return distance
        return distance.coerceIn(0f, max)
    }
}
