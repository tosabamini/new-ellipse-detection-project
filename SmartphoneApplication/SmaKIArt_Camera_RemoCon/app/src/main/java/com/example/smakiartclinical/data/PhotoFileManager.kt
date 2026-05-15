package com.example.smakiartclinical.data

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PhotoFileManager(private val context: Context) {

    private val timestampFormat = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)

    // Root under Pictures/ visible to gallery apps
    private val appFolder = "SmaKIArtClinical"

    /**
     * @param tag Optional tag inserted immediately after patientId in the filename.
     *            e.g. tag="3D" → "{patientId}_3D_IMG_{timestamp}.jpg"
     *            null           → "IMG_{timestamp}.jpg"
     */
    fun createOutputStream(patientId: String, eye: String, tag: String? = null): Pair<OutputStream, String> {
        val timestamp = timestampFormat.format(Date())
        val filename = if (tag != null) "${patientId}_${tag}_IMG_${timestamp}.jpg"
                       else "IMG_${timestamp}.jpg"

        // Pictures/SmaKIArtClinical/{patientId}/{eye}/
        val relativePath = "${Environment.DIRECTORY_PICTURES}/$appFolder/$patientId/$eye"

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("Failed to create MediaStore entry: $relativePath/$filename")
            val stream = context.contentResolver.openOutputStream(uri)
                ?: error("Failed to open output stream: $relativePath/$filename")
            Pair(stream, filename)
        } else {
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "$appFolder/$patientId/$eye"
            )
            dir.mkdirs()
            val file = File(dir, filename)
            Pair(file.outputStream(), file.absolutePath)
        }
    }
}
