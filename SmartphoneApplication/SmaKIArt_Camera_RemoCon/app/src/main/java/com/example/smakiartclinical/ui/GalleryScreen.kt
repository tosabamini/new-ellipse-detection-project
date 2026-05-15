package com.example.smakiartclinical.ui

import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smakiartclinical.analysis.SCAEstimator
import com.example.smakiartclinical.analysis.SCAResult
import com.example.smakiartclinical.data.CapturedPhoto
import com.example.smakiartclinical.data.PatientSummary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max

private val tsFmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

@Composable
fun GalleryOverlay(viewModel: CameraViewModel) {
    val view by viewModel.galleryView.collectAsState()
    if (view is GalleryView.None) return

    Box(Modifier.fillMaxSize().background(Color(0xFF101010))) {
        when (val v = view) {
            GalleryView.PatientList               -> PatientListScreen(viewModel)
            is GalleryView.EyeSelector            -> EyeSelectorScreen(viewModel, v.patientId)
            is GalleryView.ImageList              -> ImageListScreen(viewModel, v.patientId, v.eye)
            is GalleryView.AllAnalyzeResult       -> AllAnalyzeResultScreen(viewModel, v)
            GalleryView.None                      -> Unit
        }
    }
}

// ── Top bar with Back + title ─────────────────────────────────────────────────

@Composable
private fun GalleryTopBar(title: String, onBack: () -> Unit, trailing: @Composable () -> Unit = {}) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.Black.copy(alpha = 0.55f))
            .safeDrawingPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Button(
            onClick = onBack,
            colors = ButtonDefaults.buttonColors(containerColor = Color.Black.copy(alpha = 0.65f)),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
        ) { Text("← Back", color = Color.White, fontSize = 12.sp) }
        Spacer(Modifier.width(12.dp))
        Text(title, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
        trailing()
    }
}

// ── 1. Patient list ───────────────────────────────────────────────────────────

@Composable
private fun PatientListScreen(viewModel: CameraViewModel) {
    val patients by viewModel.patientSummaries.collectAsState()
    Column(Modifier.fillMaxSize()) {
        GalleryTopBar("Image Gallery", onBack = { viewModel.closeGallery() })
        if (patients.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No captured images yet", color = Color.White.copy(alpha = 0.7f), fontSize = 14.sp)
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(patients, key = { it.patientId }) { p ->
                    PatientRow(p) { viewModel.galleryOpenPatient(p.patientId) }
                }
            }
        }
    }
}

@Composable
private fun PatientRow(p: PatientSummary, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFF222222))
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f)) {
            Text(p.patientId, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            Text(
                tsFmt.format(Date(p.latestCaptureSec * 1000L)),
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 11.sp
            )
        }
        Text("R:${p.rightCount}  L:${p.leftCount}", color = Color(0xFF80FF80), fontSize = 13.sp)
        Spacer(Modifier.width(8.dp))
        Text("›", color = Color.White, fontSize = 22.sp)
    }
}

// ── 2. Eye selector ───────────────────────────────────────────────────────────

@Composable
private fun EyeSelectorScreen(viewModel: CameraViewModel, patientId: String) {
    val patients by viewModel.patientSummaries.collectAsState()
    val p = patients.firstOrNull { it.patientId == patientId }
    Column(Modifier.fillMaxSize()) {
        GalleryTopBar("$patientId", onBack = { viewModel.galleryBack() })
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalArrangement = Arrangement.spacedBy(24.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            EyeCard("RIGHT", p?.rightCount ?: 0, Modifier.weight(1f).fillMaxHeight()) {
                viewModel.galleryOpenEye(patientId, "RIGHT")
            }
            EyeCard("LEFT",  p?.leftCount  ?: 0, Modifier.weight(1f).fillMaxHeight()) {
                viewModel.galleryOpenEye(patientId, "LEFT")
            }
        }
    }
}

@Composable
private fun EyeCard(eye: String, count: Int, modifier: Modifier, onClick: () -> Unit) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(if (eye == "RIGHT") Color(0xFF1B3A5C) else Color(0xFF5C2A1B))
            .clickable(enabled = count > 0) { onClick() }
            .border(2.dp, Color.White.copy(alpha = if (count > 0) 0.4f else 0.1f), RoundedCornerShape(16.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(eye, color = Color.White, fontSize = 36.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(6.dp))
            Text("$count images", color = Color.White.copy(alpha = 0.75f), fontSize = 14.sp)
        }
    }
}

// ── 3. Image list ─────────────────────────────────────────────────────────────

@Composable
private fun ImageListScreen(viewModel: CameraViewModel, patientId: String, eye: String) {
    val photos by viewModel.galleryPhotos.collectAsState()
    val running by viewModel.isRunningAllAnalyze.collectAsState()
    val progress by viewModel.allAnalyzeProgress.collectAsState()

    Column(Modifier.fillMaxSize()) {
        GalleryTopBar("$patientId / $eye  (${photos.size})", onBack = { viewModel.galleryBack() }) {
            Button(
                onClick = { viewModel.runAllAnalyze(patientId, eye) },
                enabled = !running && photos.size >= SCAEstimator.MIN_VALID,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1B5E20))
            ) {
                if (running) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = Color.White)
                    Spacer(Modifier.width(6.dp))
                    Text("${progress.first}/${progress.second}", color = Color.White, fontSize = 12.sp)
                } else {
                    Text("All Analyze", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
        if (photos.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No images", color = Color.White.copy(alpha = 0.7f))
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                items(photos, key = { it.uri.toString() }) { photo ->
                    ImageRow(
                        photo,
                        onAnalyze = { viewModel.openPhotoForAnalysis(photo.uri) },
                        loadThumb = { uri -> viewModel.loadThumbnail(uri) }
                    )
                }
            }
        }
    }
}

@Composable
private fun ImageRow(photo: CapturedPhoto, onAnalyze: () -> Unit, loadThumb: suspend (Uri) -> Bitmap?) {
    var thumb by remember(photo.uri) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(photo.uri) { thumb = loadThumb(photo.uri) }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFF1E1E1E))
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            Modifier.size(64.dp).clip(RoundedCornerShape(4.dp)).background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            val t = thumb
            if (t != null) {
                Image(bitmap = t.asImageBitmap(), contentDescription = null,
                    modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            } else {
                CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = Color.White)
            }
        }
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text(photo.displayName, color = Color.White, fontSize = 12.sp, maxLines = 1)
            Text(tsFmt.format(Date(photo.dateAddedSec * 1000L)),
                color = Color.White.copy(alpha = 0.55f), fontSize = 10.sp)
        }
        Button(onClick = onAnalyze, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF263238))) {
            Text("Analyze", color = Color.White, fontSize = 12.sp)
        }
    }
}

// ── 4. All-Analyze result ─────────────────────────────────────────────────────

@Composable
private fun AllAnalyzeResultScreen(viewModel: CameraViewModel, v: GalleryView.AllAnalyzeResult) {
    val r = v.result
    Column(Modifier.fillMaxSize()) {
        GalleryTopBar("Result: ${v.patientId} / ${v.eye}", onBack = { viewModel.galleryBack() })
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // Numeric summary
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SCAStat("S",  "%.2f".format(r.sphere),   "diopter")
                SCAStat("C",  "%.2f".format(r.cylinder), "diopter")
                SCAStat("A",  "%.0f°".format(r.axisDeg), "axis")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SCAStat("SE", "%.2f".format(r.se), "spherical eq.")
                SCAStat("R²", "%.3f".format(r.r2), "fit quality")
                SCAStat("n",  "${r.n}",            "samples")
            }
            Divider(color = Color.White.copy(alpha = 0.15f))
            Text("Cos curve fit", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            CosCurvePlot(r, modifier = Modifier.fillMaxWidth().height(220.dp))
            Text(
                "D(α) = ${"%.2f".format(r.p0)} + ${"%.2f".format(r.p1)}·cos(2α) + ${"%.2f".format(r.p2)}·sin(2α)",
                color = Color.White.copy(alpha = 0.7f), fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun SCAStat(label: String, value: String, sub: String) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFF1E1E1E))
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        Text(label, color = Color.White.copy(alpha = 0.6f), fontSize = 11.sp)
        Text(value, color = Color(0xFF80FF80), fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Text(sub, color = Color.White.copy(alpha = 0.45f), fontSize = 9.sp)
    }
}

// ── Cos-curve plot ────────────────────────────────────────────────────────────

@Composable
private fun CosCurvePlot(r: SCAResult, modifier: Modifier = Modifier) {
    val curve = remember(r) { SCAEstimator.curvePoints(r) }
    Canvas(modifier = modifier.clip(RoundedCornerShape(8.dp)).background(Color(0xFF0F1A0F))) {
        val padL = 44f; val padR = 14f; val padT = 14f; val padB = 28f
        val plotW = size.width - padL - padR
        val plotH = size.height - padT - padB
        // Y range: include curve + sample points, with small margin
        val allY = curve.map { it.second } + r.samples.map { it.dEst }
        val yMin0 = (allY.min())
        val yMax0 = (allY.max())
        val pad = max(0.5f, (yMax0 - yMin0) * 0.15f)
        val yMin = yMin0 - pad; val yMax = yMax0 + pad
        val xMin = 0f; val xMax = 180f
        fun sx(x: Float) = padL + (x - xMin) / (xMax - xMin) * plotW
        fun sy(y: Float) = padT + (1f - (y - yMin) / (yMax - yMin)) * plotH

        // Axes
        drawLine(Color.White.copy(alpha = 0.35f),
            Offset(padL, padT), Offset(padL, padT + plotH), strokeWidth = 1.5f)
        drawLine(Color.White.copy(alpha = 0.35f),
            Offset(padL, padT + plotH), Offset(padL + plotW, padT + plotH), strokeWidth = 1.5f)

        // Y zero-line if in range
        if (yMin <= 0f && 0f <= yMax) {
            val y0 = sy(0f)
            drawLine(Color.White.copy(alpha = 0.2f),
                Offset(padL, y0), Offset(padL + plotW, y0), strokeWidth = 1f)
        }

        // X ticks: 0, 45, 90, 135, 180
        val tickPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 22f
            alpha = 180
        }
        listOf(0, 45, 90, 135, 180).forEach { t ->
            val x = sx(t.toFloat())
            drawLine(Color.White.copy(alpha = 0.2f),
                Offset(x, padT + plotH), Offset(x, padT + plotH + 4f), strokeWidth = 1f)
            drawContext.canvas.nativeCanvas.drawText("$t°", x - 12f, padT + plotH + 22f, tickPaint)
        }
        // Y ticks: 3 levels
        val yTicks = listOf(yMin, (yMin + yMax) / 2f, yMax)
        yTicks.forEach { yv ->
            val y = sy(yv)
            drawLine(Color.White.copy(alpha = 0.15f),
                Offset(padL - 4f, y), Offset(padL + plotW, y), strokeWidth = 1f)
            drawContext.canvas.nativeCanvas.drawText("%.1f".format(yv), 4f, y + 7f, tickPaint)
        }

        // Curve path
        val path = Path()
        curve.forEachIndexed { i, (ax, dv) ->
            val px = sx(ax); val py = sy(dv)
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        drawPath(path, Color(0xFF00FF80), style = Stroke(width = 2.5f))

        // Sample points
        r.samples.forEach { s ->
            drawCircle(Color(0xFFFF7043), radius = 4f, center = Offset(sx(s.angleDeg), sy(s.dEst)))
        }
    }
}
