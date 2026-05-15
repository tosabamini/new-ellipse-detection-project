package com.example.smakiartclinical

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
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
import androidx.activity.enableEdgeToEdge
import androidx.core.content.ContextCompat
import com.example.smakiartclinical.camera.OrientationManager
import com.example.smakiartclinical.ui.CameraScreen
import com.example.smakiartclinical.ui.CameraViewModel
import com.example.smakiartclinical.ui.theme.SmaKIArtClinicalTheme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {

    private val viewModel: CameraViewModel by viewModels()
    private lateinit var orientationManager: OrientationManager

    private var permissionsGranted by mutableStateOf(false)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        permissionsGranted = results.values.all { it }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Android 15+(targetSdk=36)でシステムバーの下までコンテンツが延びるよう設定
        enableEdgeToEdge()

        OpenCVLoader.initLocal()

        orientationManager = OrientationManager(this) { orientation ->
            viewModel.onOrientationChanged(orientation)
        }

        @Suppress("DEPRECATION")
        val rotation = windowManager.defaultDisplay.rotation
        viewModel.onOrientationChanged(OrientationManager.rotationToOrientation(rotation))

        checkOrRequestPermissions()

        setContent {
            SmaKIArtClinicalTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    if (permissionsGranted) {
                        CameraScreen(viewModel = viewModel)
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                "Camera permission is required. Please grant it in Settings.",
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
        orientationManager.start()
    }

    override fun onPause() {
        super.onPause()
        orientationManager.stop()
    }

    private fun checkOrRequestPermissions() {
        val required = buildList {
            add(Manifest.permission.CAMERA)
            if (Build.VERSION.SDK_INT <= 28) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
            // BLE スキャン権限
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.R) {
                // Android 11 以下: 位置情報が BLE スキャンに必要
                add(Manifest.permission.ACCESS_FINE_LOCATION)
            } else {
                // Android 12+: BLUETOOTH_SCAN + BLUETOOTH_CONNECT
                add(Manifest.permission.BLUETOOTH_SCAN)
                add(Manifest.permission.BLUETOOTH_CONNECT)
            }
        }
        val missing = required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            permissionsGranted = true
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }
}
