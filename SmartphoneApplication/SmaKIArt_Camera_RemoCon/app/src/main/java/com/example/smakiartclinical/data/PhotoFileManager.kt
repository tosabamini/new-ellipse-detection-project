package com.example.smakiartclinical.data

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
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

    // ── Gallery enumeration ──────────────────────────────────────────────────

    /** All photos under Pictures/SmaKIArtClinical, newest first. */
    fun listAllPhotos(): List<CapturedPhoto> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) listAllPhotosMediaStore()
        else listAllPhotosFile()
    }

    private fun listAllPhotosMediaStore(): List<CapturedPhoto> {
        val out = mutableListOf<CapturedPhoto>()
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.RELATIVE_PATH,
            MediaStore.Images.Media.DATE_ADDED
        )
        val selection = "${MediaStore.Images.Media.RELATIVE_PATH} LIKE ?"
        val args = arrayOf("${Environment.DIRECTORY_PICTURES}/$appFolder/%")
        val sort = "${MediaStore.Images.Media.DATE_ADDED} DESC"
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection, selection, args, sort
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val pathCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.RELATIVE_PATH)
            val dateCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
            while (c.moveToNext()) {
                val path = c.getString(pathCol) ?: continue
                val parts = path.trim('/').split("/")
                // Expect: Pictures / SmaKIArtClinical / {patientId} / {eye}
                if (parts.size < 4) continue
                if (parts[0] != Environment.DIRECTORY_PICTURES || parts[1] != appFolder) continue
                val patientId = parts[2]
                val eye = parts[3]
                if (eye !in EyeFolders.all) continue
                val id = c.getLong(idCol)
                out += CapturedPhoto(
                    uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id),
                    displayName = c.getString(nameCol) ?: "",
                    patientId = patientId,
                    eye = eye,
                    dateAddedSec = c.getLong(dateCol)
                )
            }
        }
        return out
    }

    private fun listAllPhotosFile(): List<CapturedPhoto> {
        val out = mutableListOf<CapturedPhoto>()
        val root = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            appFolder
        )
        if (!root.exists()) return out
        root.listFiles()?.forEach { patientDir ->
            if (!patientDir.isDirectory) return@forEach
            EyeFolders.all.forEach { eye ->
                val eyeDir = File(patientDir, eye)
                if (!eyeDir.isDirectory) return@forEach
                eyeDir.listFiles { f -> f.isFile && f.extension.equals("jpg", true) }?.forEach { f ->
                    out += CapturedPhoto(
                        uri = Uri.fromFile(f),
                        displayName = f.name,
                        patientId = patientDir.name,
                        eye = eye,
                        dateAddedSec = f.lastModified() / 1000L
                    )
                }
            }
        }
        return out.sortedByDescending { it.dateAddedSec }
    }

    fun listPatientSummaries(): List<PatientSummary> {
        val photos = listAllPhotos()
        val byPatient = photos.groupBy { it.patientId }
        return byPatient.map { (id, list) ->
            PatientSummary(
                patientId        = id,
                latestCaptureSec = list.maxOf { it.dateAddedSec },
                rightCount       = list.count { it.eye == EyeFolders.RIGHT },
                leftCount        = list.count { it.eye == EyeFolders.LEFT },
                right3DCount     = list.count { it.eye == EyeFolders.RIGHT3D },
                left3DCount      = list.count { it.eye == EyeFolders.LEFT3D },
                right10DCount    = list.count { it.eye == EyeFolders.RIGHT10D },
                left10DCount     = list.count { it.eye == EyeFolders.LEFT10D }
            )
        }.sortedByDescending { it.latestCaptureSec }
    }

    fun listPhotosFor(patientId: String, eye: String): List<CapturedPhoto> =
        listAllPhotos().filter { it.patientId == patientId && it.eye == eye }

    /** Decode a stored photo as a Bitmap.  inSampleSize for memory; null on failure. */
    fun loadBitmap(uri: Uri, inSampleSize: Int = 2): Bitmap? = try {
        val bmp = context.contentResolver.openInputStream(uri)?.use { input ->
            val opts = BitmapFactory.Options().apply { this.inSampleSize = inSampleSize }
            BitmapFactory.decodeStream(input, null, opts)
        }
        if (bmp == null) null else applyExifOrientation(uri, bmp)
    } catch (_: Exception) { null }

    /**
     * Rotate the bitmap according to the JPEG's EXIF Orientation tag if needed.
     * Camera2 sets EXIF Orientation via CaptureRequest.JPEG_ORIENTATION; BitmapFactory
     * doesn't apply it automatically, so we do it here so analysis sees an upright image.
     */
    private fun applyExifOrientation(uri: Uri, bmp: Bitmap): Bitmap {
        val orientation = try {
            context.contentResolver.openInputStream(uri)?.use { ExifInterface(it) }
                ?.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
                ?: ExifInterface.ORIENTATION_NORMAL
        } catch (_: Exception) { ExifInterface.ORIENTATION_NORMAL }

        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90  -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL   -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE  -> { m.postRotate(90f);  m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return bmp
        }
        return try {
            val out = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            if (out != bmp) bmp.recycle()
            out
        } catch (_: Exception) { bmp }
    }
}
