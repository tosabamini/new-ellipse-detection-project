package com.example.smakiartclinical.data.model

data class CameraSettings(
    val iso: Int = 100,
    val exposureTimeNs: Long = 33_000_000L,   // 1/30s in nanoseconds
    val focusDistance: Float = 0f,             // 0 = infinity
    val exposureCompensation: Int = 0,
    val aeEnabled: Boolean = false,
    val afEnabled: Boolean = false
)
