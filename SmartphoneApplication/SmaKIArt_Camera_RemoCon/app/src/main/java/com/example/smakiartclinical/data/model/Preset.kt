package com.example.smakiartclinical.data.model

data class Preset(
    val index: Int,       // 0–3
    val name: String,
    val settings: CameraSettings
)
