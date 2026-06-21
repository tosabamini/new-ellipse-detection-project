package com.example.smakiartclinical.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.Canvas
import com.example.smakiartclinical.analysis.EllipseConstants

@Composable
fun PhotoAnalysisScreen(
    viewModel: CameraViewModel,
    onDismiss: () -> Unit
) {
    val capturedBitmap       by viewModel.capturedBitmap.collectAsState()
    val analysisResult       by viewModel.captureAnalysisResult.collectAsState()
    val isAnalyzing          by viewModel.isAnalyzingCapture.collectAsState()
    val analysisAttempted    by viewModel.captureAnalysisAttempted.collectAsState()

    // View-only 20% center crop toggle. The saved image is never modified; this
    // only changes what is displayed, mirroring EllipseAnalyzer's analysis ROI
    // (central CROP_RATIO × CROP_RATIO of the frame).
    var cropped by remember(capturedBitmap) { mutableStateOf(false) }
    val cropInfo = remember(capturedBitmap) { capturedBitmap?.let { makeCenterCrop(it) } }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // ── Captured image ────────────────────────────────────────────────────
        val bmp = capturedBitmap
        if (bmp != null) {
            // What we actually show: full frame, or the 20% center crop.
            val shown    = if (cropped && cropInfo != null) cropInfo.bitmap else bmp
            val originX  = if (cropped && cropInfo != null) cropInfo.originX else 0
            val originY  = if (cropped && cropInfo != null) cropInfo.originY else 0

            Image(
                bitmap           = shown.asImageBitmap(),
                contentDescription = null,
                modifier         = Modifier.fillMaxSize(),
                contentScale     = ContentScale.Fit
            )

            // ── Ellipse overlay (aligned to Fit letterbox) ────────────────────
            if (analysisResult != null) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val result = analysisResult ?: return@Canvas

                    // Compute letterbox rect (ContentScale.Fit) for the shown bitmap
                    val imgW  = shown.width.toFloat()
                    val imgH  = shown.height.toFloat()
                    val scale = minOf(size.width / imgW, size.height / imgH)
                    val dispW = imgW * scale
                    val dispH = imgH * scale
                    val offX  = (size.width  - dispW) / 2f
                    val offY  = (size.height - dispH) / 2f

                    // Ellipse coords are in FULL-bitmap space; subtract the crop
                    // origin so they align with the (possibly cropped) shown image.
                    val sx = dispW / imgW
                    val sy = dispH / imgH
                    val cx = offX + (result.cxPx - originX) * sx
                    val cy = offY + (result.cyPx - originY) * sy
                    val halfMajor = result.majorPx * sx / 2f
                    val halfMinor = result.minorPx * sy / 2f

                    withTransform({ rotate(result.angleDeg, Offset(cx, cy)) }) {
                        drawOval(
                            color   = Color(0xFF00FF80),
                            topLeft = Offset(cx - halfMajor, cy - halfMinor),
                            size    = Size(halfMajor * 2f, halfMinor * 2f),
                            style   = Stroke(width = 3f)
                        )
                    }
                    val arm = 12f
                    drawLine(Color(0xFF00FF80), Offset(cx - arm, cy), Offset(cx + arm, cy), strokeWidth = 2f)
                    drawLine(Color(0xFF00FF80), Offset(cx, cy - arm), Offset(cx, cy + arm), strokeWidth = 2f)
                }
            }
        } else {
            // Loading spinner while bitmap is being decoded
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.Center),
                color    = Color.White
            )
        }

        // ── Top bar ───────────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopStart)
                .safeDrawingPadding()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Button(
                onClick = onDismiss,
                colors  = ButtonDefaults.buttonColors(containerColor = Color.Black.copy(alpha = 0.65f))
            ) {
                Text("← Back", color = Color.White, fontSize = 13.sp)
            }

            // D estimation badge
            val d = analysisResult?.dEst
            if (d != null) {
                Box(
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 12.dp, vertical = 5.dp)
                ) {
                    Text(
                        text       = "D ≈ ${"%.2f".format(d)} D",
                        color      = Color(0xFF80FF80),
                        fontSize   = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // ── Status / result text ──────────────────────────────────────────────
        if (analysisAttempted && analysisResult == null) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            ) {
                Text("No ellipse detected", color = Color(0xFFFFCC80), fontSize = 14.sp)
            }
        }

        if (analysisAttempted && analysisResult != null && analysisResult!!.dEst == null) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Ellipse found but D estimation failed", color = Color(0xFFFFCC80), fontSize = 13.sp)
                    Text(
                        "ratio = ${"%.3f".format(analysisResult!!.ratio)}",
                        color    = Color.White.copy(alpha = 0.75f),
                        fontSize = 11.sp
                    )
                }
            }
        }

        // ── Bottom-left: 20% crop toggle (view only) ──────────────────────────
        if (capturedBitmap != null) {
            Button(
                onClick = { cropped = !cropped },
                colors  = ButtonDefaults.buttonColors(
                    containerColor = if (cropped) Color(0xFF1565C0) else Color.Black.copy(alpha = 0.65f)
                ),
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .safeDrawingPadding()
                    .padding(start = 12.dp, bottom = 20.dp)
            ) {
                Text(
                    text     = if (cropped) "Full Image" else "20% Crop",
                    color    = Color.White,
                    fontSize = 13.sp
                )
            }
        }

        // ── Bottom: Analyze button ────────────────────────────────────────────
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .safeDrawingPadding()
                .padding(bottom = 20.dp)
        ) {
            Button(
                onClick  = { viewModel.analyzeCapture() },
                enabled  = !isAnalyzing && capturedBitmap != null,
                modifier = Modifier.width(180.dp).height(52.dp)
            ) {
                if (isAnalyzing) {
                    CircularProgressIndicator(
                        modifier    = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color       = Color.White
                    )
                } else {
                    Text(
                        text     = if (analysisAttempted) "Re-Analyze" else "Analyze",
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

/** A center-cropped bitmap plus its origin (top-left) in full-bitmap pixel space. */
private class CropInfo(val bitmap: Bitmap, val originX: Int, val originY: Int)

/**
 * Central CROP_RATIO × CROP_RATIO crop, mirroring `EllipseAnalyzer`'s analysis ROI
 * (same integer math) so the displayed crop matches what the pipeline actually sees.
 * Returns null if the source is too small.
 */
private fun makeCenterCrop(src: Bitmap): CropInfo? {
    val w = src.width
    val h = src.height
    val cropW = (w * EllipseConstants.CROP_RATIO).toInt().coerceAtLeast(1)
    val cropH = (h * EllipseConstants.CROP_RATIO).toInt().coerceAtLeast(1)
    val roiX  = (w - cropW) / 2
    val roiY  = (h - cropH) / 2
    if (cropW <= 0 || cropH <= 0 || roiX < 0 || roiY < 0) return null
    return CropInfo(Bitmap.createBitmap(src, roiX, roiY, cropW, cropH), roiX, roiY)
}
