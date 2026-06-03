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
    val leftCount: Int,
    val right3DCount: Int,
    val left3DCount: Int,
    val right10DCount: Int,
    val left10DCount: Int
)

/** Eye folder names recognised by the gallery. */
object EyeFolders {
    const val RIGHT    = "RIGHT"
    const val LEFT     = "LEFT"
    const val RIGHT3D  = "RIGHT3D"
    const val LEFT3D   = "LEFT3D"
    const val RIGHT10D = "RIGHT10D"
    const val LEFT10D  = "LEFT10D"
    val all = setOf(RIGHT, LEFT, RIGHT3D, LEFT3D, RIGHT10D, LEFT10D)
}
