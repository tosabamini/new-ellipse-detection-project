package com.example.smakiart_screen_frontcamera.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.BlurredEdgeTreatment
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smakiart_screen_frontcamera.R
import com.example.smakiart_screen_frontcamera.data.BalloonPreset
import kotlinx.coroutines.delay

// ── 雲霧（フォグ）演出パラメータ ───────────────────────────────────────────────
// ボタン押下でパネル全体へ段階的にブラーをかけ、オートレフの雲霧をシミュレートする。
// シーケンス: 0 →(0.8s)→ 中ブラー →(0.5s保持)→(0.5s)→ 最大ブラー →(1.0s保持)→(0.5s)→ 0
private const val FOG_MID_BLUR_DP   = 9f    // 第1段の中間ブラー半径 (dp)
private const val FOG_MAX_BLUR_DP   = 15f   // 最大ブラー半径 (dp) — 旧22dpの約7割
private const val FOG_RAMP1_MS      = 800   // 0 → 中ブラー
private const val FOG_HOLD1_MS      = 500L  // 中ブラーで保持
private const val FOG_RAMP2_MS      = 500   // 中ブラー → 最大ブラー
private const val FOG_HOLD2_MS      = 1000L // 最大ブラーで保持
private const val FOG_RAMP_OUT_MS   = 500   // 最大ブラー → 0 (解除)

@Composable
fun MainScreen(viewModel: MainViewModel) {
    val balloonOffset      by viewModel.balloonOffset.collectAsState()
    val isRecording        by viewModel.isRecording.collectAsState()
    val isConnected        by viewModel.isConnected.collectAsState()
    val patientId          by viewModel.patientId.collectAsState()
    val advertisingState   by viewModel.advertisingState.collectAsState()
    val advertisingError   by viewModel.advertisingError.collectAsState()
    val isReverseLandscape by viewModel.isReverseLandscape.collectAsState()
    val balloonSizeDp      by viewModel.balloonSizeDp.collectAsState()
    val presets            by viewModel.presets.collectAsState()

    // ── 雲霧（フォグ）アニメーション ────────────────────────────────────────────
    // fogTrigger は ViewModel が保持し、ローカルの「雲霧」ボタンと
    // BLE の FOG コマンドの両方からインクリメントされる。値が変わるたびに
    // パネル全体のブラー半径をオートレフ風に 2段階で変化させる（計 約3.3秒）:
    //   0 →(0.8s)→ 中ブラー →(0.5s保持)→(0.5s)→ 最大ブラー →(1.0s保持)→(0.5s)→ 0
    val fogTrigger by viewModel.fogTrigger.collectAsState()
    val balloonBlur = remember { Animatable(0f) }
    LaunchedEffect(fogTrigger) {
        if (fogTrigger == 0) return@LaunchedEffect   // 初期状態では何もしない
        balloonBlur.snapTo(0f)
        balloonBlur.animateTo(FOG_MID_BLUR_DP, tween(FOG_RAMP1_MS, easing = LinearEasing))
        delay(FOG_HOLD1_MS)
        balloonBlur.animateTo(FOG_MAX_BLUR_DP, tween(FOG_RAMP2_MS, easing = LinearEasing))
        delay(FOG_HOLD2_MS)
        balloonBlur.animateTo(0f, tween(FOG_RAMP_OUT_MS, easing = LinearEasing))
    }
    val balloonBlurDp = balloonBlur.value.dp

    // フロントカメラ側に常に StimulusPanel が来るようにレイアウトを切り替える
    // 通常ランドスケープ (ROTATION_90):  フロントカメラ LEFT → StimulusPanel 先
    // 逆ランドスケープ  (ROTATION_270): フロントカメラ RIGHT → ControlPanel 先
    Row(modifier = Modifier.fillMaxSize()) {
        if (isReverseLandscape) {
            ControlPanel(
                modifier         = Modifier.width(280.dp).fillMaxHeight(),
                isRecording      = isRecording,
                isConnected      = isConnected,
                advertisingState = advertisingState,
                advertisingError = advertisingError,
                patientId        = patientId,
                presets          = presets,
                onPatientIdChange = { viewModel.setPatientId(it) },
                onMove           = { dx, dy -> viewModel.moveBalloon(dx, dy) },
                onReset          = { viewModel.resetBalloon() },
                onSavePreset     = { index -> viewModel.savePreset(index) },
                onApplyPreset    = { index -> viewModel.applyPreset(index) },
                onFog            = { viewModel.triggerFog() },
                onStartRecording = { viewModel.startLocalRecording() },
                onStopRecording  = { viewModel.stopLocalRecording() }
            )
            StimulusPanel(
                modifier       = Modifier.weight(1f).fillMaxHeight(),
                balloonOffsetX = balloonOffset.x.dp,
                balloonOffsetY = balloonOffset.y.dp,
                balloonSizeDp  = balloonSizeDp,
                blurRadius     = balloonBlurDp
            )
        } else {
            StimulusPanel(
                modifier       = Modifier.weight(1f).fillMaxHeight(),
                balloonOffsetX = balloonOffset.x.dp,
                balloonOffsetY = balloonOffset.y.dp,
                balloonSizeDp  = balloonSizeDp,
                blurRadius     = balloonBlurDp
            )
            ControlPanel(
                modifier         = Modifier.width(280.dp).fillMaxHeight(),
                isRecording      = isRecording,
                isConnected      = isConnected,
                advertisingState = advertisingState,
                advertisingError = advertisingError,
                patientId        = patientId,
                presets          = presets,
                onPatientIdChange = { viewModel.setPatientId(it) },
                onMove           = { dx, dy -> viewModel.moveBalloon(dx, dy) },
                onReset          = { viewModel.resetBalloon() },
                onSavePreset     = { index -> viewModel.savePreset(index) },
                onApplyPreset    = { index -> viewModel.applyPreset(index) },
                onFog            = { viewModel.triggerFog() },
                onStartRecording = { viewModel.startLocalRecording() },
                onStopRecording  = { viewModel.stopLocalRecording() }
            )
        }
    }
}

// ── Stimulus panel ────────────────────────────────────────────────────────────

@Composable
private fun StimulusPanel(
    modifier: Modifier = Modifier,
    balloonOffsetX: Dp,
    balloonOffsetY: Dp,
    balloonSizeDp: Int = 110,
    blurRadius: Dp = 0.dp
) {
    // 雲霧演出: 背景の山と気球をまとめてぼかす。
    // blurRadius が 0dp のときは何も起きない (no-op)。
    BoxWithConstraints(
        modifier = modifier.blur(radius = blurRadius, edgeTreatment = BlurredEdgeTreatment.Rectangle)
    ) {
        Image(
            painter            = painterResource(id = R.drawable.landscape_mountain),
            contentDescription = null,
            contentScale       = ContentScale.Crop,
            modifier           = Modifier.fillMaxSize()
        )

        val balloonW = balloonSizeDp.dp
        val balloonH = (balloonSizeDp * 1.18f).dp
        val halfW    = balloonW / 2
        val halfH    = balloonH / 2

        val centerX = maxWidth  / 2
        val centerY = maxHeight * 0.42f

        val maxDx = centerX - halfW
        val maxDy = (centerY - halfH).coerceAtLeast(0.dp)

        val clampedDx = balloonOffsetX.coerceIn(-maxDx, maxDx)
        val clampedDy = balloonOffsetY.coerceIn(-maxDy, (maxHeight - centerY - halfH).coerceAtLeast(0.dp))

        val finalX = centerX + clampedDx - halfW
        val finalY = centerY + clampedDy - halfH

        Image(
            painter            = painterResource(id = R.drawable.hotballon01),
            contentDescription = "Hot air balloon",
            contentScale       = ContentScale.Fit,
            modifier           = Modifier
                .offset(x = finalX, y = finalY)
                .size(balloonW, balloonH)
        )
    }
}

// ── Control panel ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ControlPanel(
    modifier: Modifier = Modifier,
    isRecording: Boolean,
    isConnected: Boolean,
    advertisingState: Boolean?,
    advertisingError: Int,
    patientId: String,
    presets: List<BalloonPreset?>,
    onPatientIdChange: (String) -> Unit,
    onMove: (Int, Int) -> Unit,
    onReset: () -> Unit,
    onSavePreset: (Int) -> Unit,     // 0-indexed
    onApplyPreset: (Int) -> Unit,    // 0-indexed
    onFog: () -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit
) {
    Column(
        modifier = modifier
            .background(Color(0xFF1A1A2E))
            .safeDrawingPadding()
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // BLE アドバタイズ状態
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            val (dotColor, label) = when (advertisingState) {
                true  -> Color(0xFF00E676) to "BLE Advertising: Active"
                false -> Color(0xFFFF5252) to "BLE Advertising: Failed (E$advertisingError)"
                null  -> Color(0xFFFFC107) to "BLE Advertising: Starting…"
            }
            Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(dotColor))
            Spacer(Modifier.width(6.dp))
            Text(label, color = Color.White, fontSize = 11.sp)
        }

        // BLE 接続状態
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier.size(10.dp).clip(CircleShape)
                    .background(if (isConnected) Color(0xFF00E676) else Color(0xFF757575))
            )
            Spacer(Modifier.width(6.dp))
            Text(
                text     = if (isConnected) "Remote: Connected" else "Remote: Not connected",
                color    = Color.White,
                fontSize = 12.sp
            )
        }

        // Patient ID
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

        // 気球移動ボタン
        Text("Balloon Position", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                MoveButton("↑ 5",  Modifier.weight(1f)) { onMove(0, -5) }
                MoveButton("↑ 10", Modifier.weight(1f)) { onMove(0, -10) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                MoveButton("← 5",  Modifier.weight(1f)) { onMove(-5, 0) }
                MoveButton("← 10", Modifier.weight(1f)) { onMove(-10, 0) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                MoveButton("→ 5",  Modifier.weight(1f)) { onMove(5, 0) }
                MoveButton("→ 10", Modifier.weight(1f)) { onMove(10, 0) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                MoveButton("↓ 5",  Modifier.weight(1f)) { onMove(0, 5) }
                MoveButton("↓ 10", Modifier.weight(1f)) { onMove(0, 10) }
            }
        }

        // センター戻しボタン
        TextButton(
            onClick  = onReset,
            modifier = Modifier
                .fillMaxWidth()
                .height(32.dp)
                .background(Color(0xFF263238), RoundedCornerShape(6.dp)),
            colors   = ButtonDefaults.textButtonColors(contentColor = Color.White)
        ) {
            Text("Reset to Center", fontSize = 11.sp)
        }

        // ── 雲霧（フォグ）ボタン ───────────────────────────────────────────────
        // 押すと気球に段階的なブラーがかかり、オートレフの雲霧をシミュレートする。
        Button(
            onClick  = onFog,
            modifier = Modifier.fillMaxWidth(),
            colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF5C6BC0))
        ) {
            Text("☁  雲霧 (Fog)", fontSize = 14.sp)
        }

        // ── プリセット ───────────────────────────────────────────────────────
        Text("Presets", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)

        presets.forEachIndexed { index, preset ->
            Row(
                modifier              = Modifier.fillMaxWidth(),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                // スロット番号 + 保存済み表示
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .background(
                            if (preset != null) Color(0xFF1565C0) else Color(0xFF37474F),
                            RoundedCornerShape(6.dp)
                        )
                        .clickable(enabled = preset != null) { onApplyPreset(index) },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text      = "${index + 1}",
                        color     = Color.White,
                        fontSize  = 13.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                }

                // 保存済み情報 or 空スロット表示
                if (preset != null) {
                    Text(
                        text     = "x:${"%.0f".format(preset.offsetX)} y:${"%.0f".format(preset.offsetY)} s:${preset.sizeDp}",
                        color    = Color.White.copy(alpha = 0.70f),
                        fontSize = 9.sp,
                        modifier = Modifier.weight(1f)
                    )
                } else {
                    Text(
                        text     = "— empty —",
                        color    = Color.White.copy(alpha = 0.35f),
                        fontSize = 10.sp,
                        modifier = Modifier.weight(1f)
                    )
                }

                // Save ボタン
                Box(
                    modifier = Modifier
                        .background(Color(0xFF388E3C), RoundedCornerShape(6.dp))
                        .clickable { onSavePreset(index) }
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text("Save", color = Color.White, fontSize = 10.sp)
                }
            }
        }

        Spacer(Modifier.height(4.dp))

        // 録画ボタン
        Button(
            onClick  = { if (isRecording) onStopRecording() else onStartRecording() },
            modifier = Modifier.fillMaxWidth(),
            colors   = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) Color(0xFFD32F2F) else Color(0xFF388E3C)
            )
        ) {
            Text(if (isRecording) "■  Stop Recording" else "●  Start Recording", fontSize = 14.sp)
        }
    }
}

@Composable
private fun MoveButton(label: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick        = onClick,
        modifier       = modifier.height(36.dp),
        colors         = ButtonDefaults.buttonColors(containerColor = Color(0xFF37474F)),
        shape          = RoundedCornerShape(6.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(0.dp)
    ) {
        Text(label, fontSize = 12.sp, color = Color.White)
    }
}
