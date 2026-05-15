package com.example.smakiartclinical.camera

import android.content.Context
import android.hardware.SensorManager
import android.view.OrientationEventListener
import android.view.Surface

enum class DeviceOrientation { LANDSCAPE, REVERSE_LANDSCAPE, OTHER }

class OrientationManager(
    context: Context,
    private val onOrientationChanged: (DeviceOrientation) -> Unit
) {
    private var lastOrientation: DeviceOrientation = DeviceOrientation.OTHER

    // OrientationEventListener reports clockwise degrees from the device's natural (portrait) orientation.
    // ~270° means device rotated 90° CCW from portrait  = home-button right  = ROTATION_90  = LANDSCAPE
    // ~90°  means device rotated 90° CW  from portrait  = home-button left   = ROTATION_270 = REVERSE_LANDSCAPE
    private val eventListener = object : OrientationEventListener(
        context, SensorManager.SENSOR_DELAY_NORMAL
    ) {
        override fun onOrientationChanged(angle: Int) {
            if (angle == ORIENTATION_UNKNOWN) return
            val orientation = when (angle) {
                in 225..315 -> DeviceOrientation.LANDSCAPE
                in 45..135  -> DeviceOrientation.REVERSE_LANDSCAPE
                else        -> DeviceOrientation.OTHER
            }
            if (orientation != DeviceOrientation.OTHER && orientation != lastOrientation) {
                lastOrientation = orientation
                onOrientationChanged(orientation)
            }
        }
    }

    fun start() { eventListener.enable() }
    fun stop()  { eventListener.disable() }

    companion object {
        fun rotationToOrientation(rotation: Int): DeviceOrientation = when (rotation) {
            Surface.ROTATION_90  -> DeviceOrientation.LANDSCAPE
            Surface.ROTATION_270 -> DeviceOrientation.REVERSE_LANDSCAPE
            else                 -> DeviceOrientation.OTHER
        }
    }
}
