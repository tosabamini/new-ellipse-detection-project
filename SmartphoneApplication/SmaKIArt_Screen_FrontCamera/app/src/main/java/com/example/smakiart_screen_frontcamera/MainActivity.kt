package com.example.smakiart_screen_frontcamera

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.view.OrientationEventListener
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.smakiart_screen_frontcamera.ui.MainScreen
import com.example.smakiart_screen_frontcamera.ui.MainViewModel
import com.example.smakiart_screen_frontcamera.ui.theme.SmaKIArt_Screen_FrontCameraTheme

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private var permissionsGranted by mutableStateOf(false)

    // センサーベースの向き検出（onConfigurationChanged より確実）
    // OrientationEventListener は clockwise 度数（自然向き=0° からの回転）を報告する。
    //   angle 225-315° → 通常ランドスケープ (ROTATION_90)  → フロントカメラ LEFT
    //   angle  45-135° → 逆ランドスケープ  (ROTATION_270) → フロントカメラ RIGHT
    private val orientationListener by lazy {
        object : OrientationEventListener(this, SensorManager.SENSOR_DELAY_NORMAL) {
            override fun onOrientationChanged(angle: Int) {
                if (angle == ORIENTATION_UNKNOWN) return
                when (angle) {
                    in 225..315 -> viewModel.setReverseLandscape(false) // 通常ランドスケープ
                    in 45..135  -> viewModel.setReverseLandscape(true)  // 逆ランドスケープ
                }
            }
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        permissionsGranted = results.values.all { it }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        checkOrRequestPermissions()

        setContent {
            SmaKIArt_Screen_FrontCameraTheme {
                androidx.compose.material3.Surface(modifier = Modifier.fillMaxSize()) {
                    if (permissionsGranted) {
                        MainScreen(viewModel = viewModel)
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                "Camera, audio, and Bluetooth permissions are required. Please grant them in Settings.",
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        orientationListener.enable()
    }

    override fun onPause() {
        super.onPause()
        orientationListener.disable()
    }

    // configChanges はアクティビティ再生成を防ぐために引き続き必要だが、
    // 向き検出はセンサーに任せるため updateOrientation() の呼び出しは不要
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
    }

    private fun checkOrRequestPermissions() {
        val required = buildList {
            add(Manifest.permission.CAMERA)
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                add(Manifest.permission.BLUETOOTH_ADVERTISE)
                add(Manifest.permission.BLUETOOTH_CONNECT)
            }
        }
        val missing = required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) permissionsGranted = true
        else permissionLauncher.launch(missing.toTypedArray())
    }
}
