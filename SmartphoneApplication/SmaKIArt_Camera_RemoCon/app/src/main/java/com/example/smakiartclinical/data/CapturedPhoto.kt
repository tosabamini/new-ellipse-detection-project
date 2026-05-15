package com.example.smakiartclinical.data

import android.net.Uri

/** A single saved JPEG in the app's gallery. */
data class CapturedPhoto(
    val uri: Uri,
    val displayName: String,
    val patientId: String,
    val eye: String,            // "RIGHT" or "LEFT"
    val dateAddedSec: Long      // unix seconds, from MediaStore DATE_ADDED
)

/** Patient-level aggregate for the gallery's top-level list. */
data class PatientSummary(
    val patientId: String,
    val latestCaptureSec: Long,
    val rightCount: Int,
    val leftCount: Int
)
