package com.example.smakiartclinical.ui.components

import android.view.TextureView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun CameraPreviewView(
    modifier: Modifier = Modifier,
    onTextureViewReady: (TextureView) -> Unit
) {
    AndroidView(
        factory = { context ->
            TextureView(context).also { onTextureViewReady(it) }
        },
        modifier = modifier
    )
}
