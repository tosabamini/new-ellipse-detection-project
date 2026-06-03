package com.example.smakiartclinical.ui

import android.annotation.SuppressLint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.view.TextureView
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smakiartclinical.analysis.EllipseResult
import com.example.smakiartclinical.data.model.CameraSettings
import com.example.smakiartclinical.data.model.Preset
import com.example.smakiartclinical.ui.components.CameraPreviewView
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt

private const val MAX_SLIDER_EXPOSURE_MS = 200f

// ── Theme: primary action colour + eye-tinted panel background ───────────────
private val PrimaryPurple = Color(0xFF7B1FA2)        // Material Purple 700
private val GalleryRightBlue  = Color(0xFF1B3A5C)    // existing card colour
private val GalleryLeftGreen  = Color(0xFF1B5C2F)    // green replacement for the left card

/** Returns the eye-tinted translucent background for the side panels. */
private fun panelBgFor(eye: String): Color = when (eye) {
    "RIGHT" -> Color(0xFF0D47A1).copy(alpha = 0.55f)  // blue tint
    "LEFT"  -> Color(0xFF1B5E20).copy(alpha = 0.55f)  // green tint
    else    -> Color.Black.copy(alpha = 0.72f)
}

private enum class DPadDir { UP, DOWN, LEFT, RIGHT }

// ── Main screen ───────────────────────────────────────────────────────────────

@Composable
fun CameraScreen(viewModel: CameraViewModel) {
    val settings          by viewModel.settings.collectAsState()
    val session           by viewModel.session.collectAsState()
    val presets           by viewModel.presets.collectAsState()
    val isCapturing       by viewModel.isCapturing.collectAsState()
    val message           by viewModel.message.collectAsState()
    val btState           by viewModel.btState.collectAsState()
    val isRemoteRecording by viewModel.isRemoteRecording.collectAsState()
    val ellipseResult        by viewModel.ellipseResult.collectAsState()
    val isAnalysisRunning    by viewModel.isAnalysisRunning.collectAsState()
    val showAnalysisScreen   by viewModel.showAnalysisScreen.collectAsState()
    val currentOrientation   by viewModel.currentOrientation.collectAsState()

    var showSettings by remember { mutableStateOf(false) }
    var dpadStep     by remember { mutableIntStateOf(10) }

    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(message) {
        message?.let {
            snackbarHostState.showSnackbar(it.text)
            viewModel.clearMessage()
        }
    }

    DisposableEffect(Unit) {
        onDispose { viewModel.closeCamera() }
    }

    val chars = viewModel.cameraController.getCharacteristics()

    Box(modifier = Modifier.fillMaxSize()) {

        // ── Layer 1: Full-screen camera preview ──────────────────────────────
        CameraPreviewView(
            modifier = Modifier.fillMaxSize(),
            onTextureViewReady = { tv ->
                if (tv.isAvailable) {
                    viewModel.openCamera(tv)
                } else {
                    tv.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                        override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) { viewModel.openCamera(tv) }
                        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
                        override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean = true
                        override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
                    }
                }
            }
        )

        // ── Layer 2: Ellipse overlay canvas (live preview analysis) ──────────
        EllipseCanvas(
            result = ellipseResult,
            isReverseLandscape = currentOrientation == com.example.smakiartclinical.camera.DeviceOrientation.REVERSE_LANDSCAPE,
            modifier = Modifier.fillMaxSize()
        )

        // ── Layer 3: Overlay UI ───────────────────────────────────────────────
        Box(modifier = Modifier.fillMaxSize().safeDrawingPadding()) {

            val panelBg = panelBgFor(session.selectedEye)

            // Left: Remote control panel — always visible
            RemotePanel(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .width(196.dp)
                    .fillMaxHeight()
                    .padding(bottom = 80.dp),
                bgColor             = panelBg,
                btState             = btState,
                step                = dpadStep,
                onStepChange        = { dpadStep = it },
                onScan              = { viewModel.startBluetoothScan() },
                onStopScan          = { viewModel.stopBluetoothScan() },
                onDisconnect        = { viewModel.disconnectBluetooth() },
                onBalloonMove       = { dx, dy -> viewModel.sendBalloonMove(dx, dy) },
                onBalloonSizeChange = { delta -> viewModel.sendBalloonSizeChange(delta) },
                onPreset            = { n -> viewModel.sendPreset(n) }
            )

            // Right: Session panel with settings icon in header
            if (!showSettings) {
                SessionPanel(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .width(176.dp),
                    bgColor           = panelBg,
                    patientId         = session.patientId,
                    onPatientIdChange = { viewModel.setPatientId(it) },
                    selectedEye       = session.selectedEye,
                    onEyeSelect       = { viewModel.setSelectedEye(it) },
                    isAnalysisRunning = isAnalysisRunning,
                    onToggleAnalysis  = { viewModel.toggleAnalysis() },
                    onFinishSession   = { viewModel.finishSession() },
                    onOpenGallery     = { viewModel.openGallery() },
                    onOpenSettings    = { showSettings = true }
                )

                // ISO quick-adjust column — just left of the session panel
                IsoQuickPanel(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(end = 180.dp, top = 4.dp)
                        .width(60.dp),
                    bgColor    = panelBg,
                    isoValue   = settings.iso,
                    onBumpUp   = { viewModel.bumpIso(+100) },
                    onBumpDown = { viewModel.bumpIso(-100) }
                )
            }

            // Right: Settings panel (replaces session panel)
            if (showSettings) {
                SettingsPanel(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .fillMaxHeight()
                        .width(300.dp),
                    settings         = settings,
                    presets          = presets,
                    chars            = chars,
                    onSettingsChange = { viewModel.updateSettings(it) },
                    onReset          = { viewModel.resetSettings() },
                    onSavePreset     = { i, n -> viewModel.savePreset(i, n) },
                    onLoadPreset     = { viewModel.loadPreset(it) },
                    onDeletePreset   = { viewModel.deletePreset(it) },
                    onClose          = { showSettings = false }
                )
            }

            // Bottom-right: captured count + snackbar
            Column(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .widthIn(max = 220.dp)
                    .padding(end = 8.dp, bottom = 6.dp),
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                if (session.capturedFiles.isNotEmpty()) {
                    Box(
                        modifier = Modifier
                            .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 3.dp)
                    ) {
                        Text("${session.capturedFiles.size} captured", color = Color.White, fontSize = 11.sp)
                    }
                }
                SnackbarHost(hostState = snackbarHostState, modifier = Modifier.fillMaxWidth())
            }

            // D-estimation badge — shown above shutter bar when ellipse is detected
            ellipseResult?.dEst?.let { d ->
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .offset(x = 10.dp, y = (-88).dp)
                        .background(Color.Black.copy(alpha = 0.60f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text      = "D ≈ ${"%.2f".format(d)} D",
                        color     = Color(0xFF80FF80),
                        fontSize  = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Debug: bitmap size + detected angle (remove when live-preview alignment is fixed)
            ellipseResult?.let { r ->
                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 4.dp)
                        .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(
                        "${r.bitmapW}×${r.bitmapH} angle=${r.angleDeg.toInt()}°",
                        color    = Color.Cyan,
                        fontSize = 10.sp
                    )
                }
            }

            // Bottom: Shutter controls + Start REC — no background, lifted 8dp, shifted 10dp right
            BottomShutterBar(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .offset(x = 10.dp, y = (-8).dp),
                isCapturing       = isCapturing,
                isRemoteRecording = isRemoteRecording,
                onCapture         = { viewModel.captureImage() },
                onCapture3D       = { viewModel.captureFocusPair3D() },
                onCapture10D      = { viewModel.captureFocusPair10D() },
                onVideoStart      = { viewModel.sendVideoStart() },
                onVideoStop       = { viewModel.sendVideoStop() }
            )
        }

        // ── Layer 4: Image gallery ────────────────────────────────────────────
        GalleryOverlay(viewModel = viewModel)

        // ── Layer 5: Photo analysis screen (top-most when active) ─────────────
        if (showAnalysisScreen) {
            PhotoAnalysisScreen(
                viewModel = viewModel,
                onDismiss = { viewModel.dismissAnalysisScreen() }
            )
        }
    }
}

// ── Ellipse overlay canvas ────────────────────────────────────────────────────
//
// The bitmap fed to EllipseAnalyzer is PORTRAIT-oriented (e.g. 864×1920) — getBitmap
// returns the sensor-rotation-corrected GL texture content, which is naturally portrait.
// The display is landscape, with applyPreviewTransform rotating the portrait content
// -90° CCW (normal LANDSCAPE) or +90° CW (REVERSE_LANDSCAPE) and uniformly scaling
// to fill.  We mirror that exact mapping here.
//
// For -90° CCW (normal LANDSCAPE):
//   portrait +x  →  screen -y   (right in portrait = up on screen)
//   portrait +y  →  screen +x   (down in portrait  = right on screen)

@Composable
private fun EllipseCanvas(
    result: EllipseResult?,
    isReverseLandscape: Boolean,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        if (result == null) return@Canvas
        val bmpW = result.bitmapW.toFloat()  // portrait width  (e.g. 864)
        val bmpH = result.bitmapH.toFloat()  // portrait height (e.g. 1920)

        // After -90° rotation the portrait's height (bmpH) maps to display width,
        // its width (bmpW) maps to display height.  Match applyPreviewTransform's
        // centre-crop scale formula.
        val scale = maxOf(size.width / bmpH, size.height / bmpW)
        val centerX = size.width / 2f
        val centerY = size.height / 2f

        val cx: Float
        val cy: Float
        val screenAngle: Float
        if (!isReverseLandscape) {
            cx = centerX + (result.cyPx - bmpH / 2f) * scale
            cy = centerY - (result.cxPx - bmpW / 2f) * scale
            screenAngle = result.angleDeg - 90f
        } else {
            cx = centerX - (result.cyPx - bmpH / 2f) * scale
            cy = centerY + (result.cxPx - bmpW / 2f) * scale
            screenAngle = result.angleDeg + 90f
        }

        val halfMajor = result.majorPx * scale / 2f
        val halfMinor = result.minorPx * scale / 2f
        val pivot = Offset(cx, cy)
        withTransform({ rotate(screenAngle, pivot) }) {
            drawOval(
                color   = Color(0xFF00FF80),
                topLeft = Offset(cx - halfMajor, cy - halfMinor),
                size    = Size(halfMajor * 2f, halfMinor * 2f),
                style   = Stroke(width = 3f)
            )
        }
        val arm = 10f
        drawLine(Color(0xFF00FF80), Offset(cx - arm, cy), Offset(cx + arm, cy), strokeWidth = 2f)
        drawLine(Color(0xFF00FF80), Offset(cx, cy - arm), Offset(cx, cy + arm), strokeWidth = 2f)
    }
}

// ── Remote control panel ──────────────────────────────────────────────────────

@Composable
private fun RemotePanel(
    modifier: Modifier = Modifier,
    bgColor: Color,
    btState: BtConnectionState,
    step: Int,
    onStepChange: (Int) -> Unit,
    onScan: () -> Unit,
    onStopScan: () -> Unit,
    onDisconnect: () -> Unit,
    onBalloonMove: (Int, Int) -> Unit,
    onBalloonSizeChange: (Int) -> Unit,
    onPreset: (Int) -> Unit          // 1-indexed preset number
) {
    val connected = btState == BtConnectionState.CONNECTED

    Column(
        modifier = modifier
            .background(bgColor)
            .padding(horizontal = 10.dp, vertical = 8.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text("Remote", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)

        // BT status row
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            val (dot, label) = when (btState) {
                BtConnectionState.DISCONNECTED -> Color(0xFF757575) to "Disconnected"
                BtConnectionState.SCANNING     -> Color(0xFFFFC107) to "Scanning…"
                BtConnectionState.CONNECTING   -> Color(0xFFFFC107) to "Connecting…"
                BtConnectionState.CONNECTED    -> Color(0xFF00E676) to "Connected"
            }
            Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
            Spacer(Modifier.width(5.dp))
            Text(label, color = Color.White, fontSize = 11.sp, modifier = Modifier.weight(1f))
            if (connected) {
                TextButton(
                    onClick = onDisconnect,
                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                    modifier = Modifier.height(24.dp)
                ) {
                    Text("Disconnect", fontSize = 9.sp, color = Color(0xFFFF6B6B))
                }
            }
        }

        // Connection actions (non-connected states)
        when (btState) {
            BtConnectionState.DISCONNECTED -> {
                Button(
                    onClick = onScan,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryPurple)
                ) {
                    Text("Search", fontSize = 12.sp)
                }
            }
            BtConnectionState.SCANNING -> {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = Color.White)
                    TextButton(onClick = onStopScan) { Text("Stop", fontSize = 11.sp, color = Color.White) }
                }
            }
            BtConnectionState.CONNECTING -> {
                CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = Color.White)
            }
            BtConnectionState.CONNECTED -> Unit
        }

        // Controls shown only when connected
        if (connected) {
            Divider(color = Color.White.copy(alpha = 0.18f))

            Text("Balloon", color = Color.White.copy(alpha = 0.65f), fontSize = 10.sp)

            // D-pad
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(3.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                DPadButton(DPadDir.UP)    { onBalloonMove(0, -step) }
                Row(horizontalArrangement = Arrangement.spacedBy(3.dp), verticalAlignment = Alignment.CenterVertically) {
                    DPadButton(DPadDir.LEFT)  { onBalloonMove(-step, 0) }
                    Spacer(Modifier.size(44.dp))
                    DPadButton(DPadDir.RIGHT) { onBalloonMove(step, 0) }
                }
                DPadButton(DPadDir.DOWN)  { onBalloonMove(0, step) }
            }

            // Step selector（大きめ）
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Step:", color = Color.White.copy(alpha = 0.65f), fontSize = 11.sp)
                listOf(5, 10).forEach { s ->
                    val sel = step == s
                    Box(
                        modifier = Modifier
                            .background(
                                if (sel) Color.White else Color.White.copy(alpha = 0.15f),
                                RoundedCornerShape(5.dp)
                            )
                            .clickable { onStepChange(s) }
                            .padding(horizontal = 10.dp, vertical = 5.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("$s", color = if (sel) Color.Black else Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            // Size buttons（大きめ）
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Size:", color = Color.White.copy(alpha = 0.65f), fontSize = 11.sp)
                SizeButton("−") { onBalloonSizeChange(-15) }
                SizeButton("+") { onBalloonSizeChange(+15) }
            }

            Divider(color = Color.White.copy(alpha = 0.18f))

            // プリセットボタン ①②③④
            Text("Preset", color = Color.White.copy(alpha = 0.65f), fontSize = 10.sp)
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                listOf("①", "②", "③", "④").forEachIndexed { i, label ->
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color(0xFF37474F))
                            .clickable { onPreset(i + 1) },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(label, color = Color.White, fontSize = 18.sp)
                    }
                }
            }
        }
    }
}

// ── D-pad chevron button ──────────────────────────────────────────────────────

@Composable
private fun DPadButton(dir: DPadDir, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(46.dp)
            .clip(CircleShape)
            .background(Color(0xFF1C1C1C))
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.size(22.dp)) {
            val stroke = Stroke(width = 2.6.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round)
            val cx = size.width / 2f;  val cy = size.height / 2f
            val arm = size.width * 0.33f;  val dep = size.height * 0.22f
            val path = Path()
            when (dir) {
                DPadDir.UP    -> { path.moveTo(cx - arm, cy + dep); path.lineTo(cx, cy - dep); path.lineTo(cx + arm, cy + dep) }
                DPadDir.DOWN  -> { path.moveTo(cx - arm, cy - dep); path.lineTo(cx, cy + dep); path.lineTo(cx + arm, cy - dep) }
                DPadDir.LEFT  -> { path.moveTo(cx + dep, cy - arm); path.lineTo(cx - dep, cy); path.lineTo(cx + dep, cy + arm) }
                DPadDir.RIGHT -> { path.moveTo(cx - dep, cy - arm); path.lineTo(cx + dep, cy); path.lineTo(cx - dep, cy + arm) }
            }
            drawPath(path, Color.White, style = stroke)
        }
    }
}

// ── Balloon size button（少し大きめ）────────────────────────────────────────

@Composable
private fun SizeButton(label: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(32.dp)
            .clip(CircleShape)
            .background(Color(0xFF1C1C1C))
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(label, color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.Bold)
    }
}

// ── Session panel ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SessionPanel(
    modifier: Modifier = Modifier,
    bgColor: Color,
    patientId: String,
    onPatientIdChange: (String) -> Unit,
    selectedEye: String,
    onEyeSelect: (String) -> Unit,
    isAnalysisRunning: Boolean,
    onToggleAnalysis: () -> Unit,
    onFinishSession: () -> Unit,
    onOpenGallery: () -> Unit,
    onOpenSettings: () -> Unit
) {
    Column(
        modifier = modifier
            .background(bgColor)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("Session", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .background(Color.Black.copy(alpha = 0.50f))
                    .clickable { onOpenSettings() }
                    .padding(5.dp)
            ) {
                Icon(
                    imageVector        = Icons.Default.Settings,
                    contentDescription = "Settings",
                    tint               = Color.White,
                    modifier           = Modifier.size(18.dp)
                )
            }
        }

        OutlinedTextField(
            value         = patientId,
            onValueChange = onPatientIdChange,
            label         = { Text("Patient ID") },
            singleLine    = true,
            modifier      = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor     = Color.White,
                unfocusedTextColor   = Color.White,
                focusedBorderColor   = Color.White,
                unfocusedBorderColor = Color.White.copy(alpha = 0.5f),
                focusedLabelColor    = Color.White,
                unfocusedLabelColor  = Color.White.copy(alpha = 0.7f),
                cursorColor          = Color.White
            )
        )

        EyeToggleButtons(selected = selectedEye, onSelect = onEyeSelect)

        Button(
            onClick  = onToggleAnalysis,
            modifier = Modifier.fillMaxWidth(),
            colors   = ButtonDefaults.buttonColors(containerColor = PrimaryPurple)
        ) {
            Text(
                text     = if (isAnalysisRunning) "■ Stop Analysis" else "▶ Start Analysis",
                fontSize = 12.sp,
                color    = Color.White
            )
        }

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            TextButton(
                onClick  = onOpenGallery,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 4.dp),
                colors   = ButtonDefaults.textButtonColors(contentColor = PrimaryPurple)
            ) {
                Text("Gallery", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
            TextButton(
                onClick  = onFinishSession,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 4.dp),
                colors   = ButtonDefaults.textButtonColors(contentColor = PrimaryPurple)
            ) {
                Text("End", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

// ── ISO quick-adjust column ───────────────────────────────────────────────────
//
// Sits just left of the SessionPanel.  Up / Down chevrons bump ISO ±100 with the
// current value shown between them.  Lets the operator retune ISO mid-session
// without opening the full settings drawer.

@Composable
private fun IsoQuickPanel(
    modifier: Modifier = Modifier,
    bgColor: Color,
    isoValue: Int,
    onBumpUp: () -> Unit,
    onBumpDown: () -> Unit
) {
    Column(
        modifier = modifier
            .background(bgColor, RoundedCornerShape(8.dp))
            .padding(vertical = 4.dp, horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        IsoBumpButton("▲", onClick = onBumpUp)
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("ISO", color = Color.White.copy(alpha = 0.65f), fontSize = 9.sp)
            Text("$isoValue", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
        }
        IsoBumpButton("▼", onClick = onBumpDown)
    }
}

@Composable
private fun IsoBumpButton(label: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(width = 48.dp, height = 26.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(PrimaryPurple)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(label, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
    }
}

// ── Eye toggle ────────────────────────────────────────────────────────────────

@Composable
private fun EyeToggleButtons(selected: String, onSelect: (String) -> Unit) {
    Row(
        modifier = Modifier
            .background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(8.dp))
            .padding(3.dp),
        horizontalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        listOf("RIGHT" to "R", "LEFT" to "L").forEach { (eye, label) ->
            val isSelected = selected == eye
            Box(
                modifier = Modifier
                    .background(if (isSelected) Color.White else Color.Transparent, RoundedCornerShape(6.dp))
                    .clickable { onSelect(eye) }
                    .padding(horizontal = 14.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text       = label,
                    color      = if (isSelected) Color.Black else Color.White,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                    fontSize   = 14.sp
                )
            }
        }
    }
}

// ── Bottom shutter bar — Start REC (left) + 3D / Shutter / 10D (center-right) ─

@Composable
private fun BottomShutterBar(
    modifier: Modifier = Modifier,
    isCapturing: Boolean,
    isRemoteRecording: Boolean,
    onCapture: () -> Unit,
    onCapture3D: () -> Unit,
    onCapture10D: () -> Unit,
    onVideoStart: () -> Unit,
    onVideoStop: () -> Unit
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        // Start / Stop REC — シャッターの左、緑 or 赤の円形ボタン
        RecButton(
            isRecording = isRemoteRecording,
            onClick     = { if (isRemoteRecording) onVideoStop() else onVideoStart() }
        )

        Spacer(Modifier.width(28.dp))

        // 3D / メインシャッター / 10D
        FocusShutterButton(label = "3D",  isCapturing = isCapturing, onClick = onCapture3D)
        Spacer(Modifier.width(16.dp))
        ShutterButton(isCapturing = isCapturing, onClick = onCapture)
        Spacer(Modifier.width(16.dp))
        FocusShutterButton(label = "10D", isCapturing = isCapturing, onClick = onCapture10D)
    }
}

// ── REC ボタン（緑/赤の小さめ円形）────────────────────────────────────────────

@Composable
private fun RecButton(isRecording: Boolean, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(52.dp)
            .clip(CircleShape)
            .background(
                if (isRecording) Color(0xFFD32F2F) else Color(0xFF388E3C)
            )
            .border(2.dp, Color.White.copy(alpha = 0.55f), CircleShape)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(
            text     = if (isRecording) "■" else "●",
            color    = Color.White,
            fontSize = 20.sp
        )
    }
}

@Composable
private fun ShutterButton(isCapturing: Boolean, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(64.dp)
            .clip(CircleShape)
            .border(3.dp, Color.White.copy(alpha = 0.75f), CircleShape)
            .clickable(enabled = !isCapturing) { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Box(modifier = Modifier.size(50.dp).clip(CircleShape)
            .background(if (isCapturing) Color(0xFFAAAAAA) else Color.White))
    }
}

@Composable
private fun FocusShutterButton(label: String, isCapturing: Boolean, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(56.dp)
            .clip(CircleShape)
            .border(2.dp, Color.White.copy(alpha = 0.75f), CircleShape)
            .clickable(enabled = !isCapturing) { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier.size(44.dp).clip(CircleShape)
                .background(if (isCapturing) Color(0xFFAAAAAA) else Color.White),
            contentAlignment = Alignment.Center
        ) {
            Text(label, color = Color.Black, fontSize = 11.sp, fontWeight = FontWeight.Bold)
        }
    }
}

// ── Manual settings panel ─────────────────────────────────────────────────────

@Composable
private fun SettingsPanel(
    modifier: Modifier = Modifier,
    settings: CameraSettings,
    presets: List<Preset?>,
    chars: CameraCharacteristics?,
    onSettingsChange: (CameraSettings) -> Unit,
    onReset: () -> Unit,
    onSavePreset: (Int, String) -> Unit,
    onLoadPreset: (Preset) -> Unit,
    onDeletePreset: (Int) -> Unit,
    onClose: () -> Unit
) {
    Column(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.93f))
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("Manual Settings", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = onReset) { Text("Reset") }
            IconButton(onClick = onClose) { Icon(Icons.Default.Close, contentDescription = "Close") }
        }

        val isoRange  = chars?.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val isoMin    = (isoRange?.lower ?: 50).toFloat().coerceAtLeast(1f)
        val isoMax    = (isoRange?.upper ?: 3200).toFloat()
        val logIsoMin = ln(isoMin); val logIsoMax = ln(isoMax)
        val logIsoVal = ln(settings.iso.toFloat().coerceIn(isoMin, isoMax))

        LabeledSlider("ISO: ${settings.iso}", logIsoVal, logIsoMin..logIsoMax, 0, !settings.aeEnabled) { logVal ->
            val raw = exp(logVal)
            val rounded = ((raw / 100f).roundToInt() * 100).coerceIn(isoMin.toInt(), isoMax.toInt())
            onSettingsChange(settings.copy(iso = rounded.coerceAtLeast(1)))
        }

        val expRange     = chars?.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val expMinMs     = ((expRange?.lower ?: 100_000L) / 1_000_000f).coerceAtLeast(0.001f)
        val expMaxMs     = minOf((expRange?.upper ?: 200_000_000L) / 1_000_000f, MAX_SLIDER_EXPOSURE_MS)
        val expCurrentMs = (settings.exposureTimeNs / 1_000_000f).coerceIn(expMinMs, expMaxMs)
        val logExpMin    = ln(expMinMs); val logExpMax = ln(expMaxMs)

        LabeledSlider("Exposure: ${"%.1f".format(expCurrentMs)} ms", ln(expCurrentMs), logExpMin..logExpMax, 0, !settings.aeEnabled) { logVal ->
            onSettingsChange(settings.copy(exposureTimeNs = (exp(logVal) * 1_000_000f).toLong()))
        }

        val maxFocus = chars?.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 10f
        LabeledSlider("Focus: ${"%.2f".format(settings.focusDistance)} diopters", settings.focusDistance, 0f..maxFocus, 0, !settings.afEnabled) {
            onSettingsChange(settings.copy(focusDistance = it))
        }

        val aeRange = chars?.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val aeStep  = chars?.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP)?.toFloat() ?: 1f
        val aeMin   = aeRange?.lower?.toFloat() ?: -3f
        val aeMax   = aeRange?.upper?.toFloat() ?: 3f
        LabeledSlider("EV Comp: ${settings.exposureCompensation}", settings.exposureCompensation.toFloat(), aeMin..aeMax,
            ((aeMax - aeMin) / aeStep).toInt().coerceAtLeast(0), settings.aeEnabled) {
            onSettingsChange(settings.copy(exposureCompensation = it.toInt()))
        }

        Text("Auto Modes", style = MaterialTheme.typography.labelLarge)
        AutoToggleRow("AE (Auto Exposure)", settings.aeEnabled) { onSettingsChange(settings.copy(aeEnabled = it)) }
        AutoToggleRow("AF (Auto Focus)",    settings.afEnabled) { onSettingsChange(settings.copy(afEnabled = it)) }

        Text("Presets", style = MaterialTheme.typography.labelLarge)
        presets.forEachIndexed { index, preset ->
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                if (preset != null) {
                    Button(onClick = { onLoadPreset(preset) }, modifier = Modifier.weight(1f)) {
                        Text(preset.name, maxLines = 1)
                    }
                    Spacer(Modifier.width(4.dp))
                    TextButton(onClick = { onDeletePreset(index) }) { Text("X") }
                } else {
                    Button(onClick = { onSavePreset(index, "Preset ${index + 1}") }, modifier = Modifier.weight(1f)) {
                        Text("Save → P${index + 1}")
                    }
                }
            }
        }
    }
}

// ── Shared primitives ─────────────────────────────────────────────────────────

@Composable
private fun LabeledSlider(
    label: String, value: Float, valueRange: ClosedFloatingPointRange<Float>,
    steps: Int, enabled: Boolean, onValueChange: (Float) -> Unit
) {
    Column {
        Text(label, fontSize = 12.sp)
        Slider(value = value.coerceIn(valueRange), onValueChange = onValueChange,
            valueRange = valueRange, steps = steps, enabled = enabled, modifier = Modifier.fillMaxWidth())
    }
}

@Composable
private fun AutoToggleRow(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Text(label, modifier = Modifier.weight(1f), fontSize = 13.sp)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
