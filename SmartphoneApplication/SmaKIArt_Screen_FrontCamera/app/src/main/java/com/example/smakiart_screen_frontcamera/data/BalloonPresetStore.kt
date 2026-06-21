package com.example.smakiart_screen_frontcamera.data

import android.content.Context

data class BalloonPreset(
    val offsetX: Float,
    val offsetY: Float,
    val sizeDp: Int
)

/**
 * SharedPreferences を使った気球プリセットの永続化。
 * プリセットは 1〜4 の 1-indexed で保存・読み込みする。
 */
class BalloonPresetStore(context: Context) {

    private val prefs = context.getSharedPreferences("balloon_presets", Context.MODE_PRIVATE)

    /** index: 1-indexed (1〜4) */
    fun save(index: Int, preset: BalloonPreset) {
        prefs.edit()
            .putFloat("p${index}_x",    preset.offsetX)
            .putFloat("p${index}_y",    preset.offsetY)
            .putInt  ("p${index}_size", preset.sizeDp)
            .putBoolean("p${index}_ok", true)
            .apply()
    }

    /** index: 1-indexed (1〜4)。未保存なら null を返す。 */
    fun load(index: Int): BalloonPreset? {
        if (!prefs.getBoolean("p${index}_ok", false)) return null
        return BalloonPreset(
            offsetX = prefs.getFloat("p${index}_x",    0f),
            offsetY = prefs.getFloat("p${index}_y",    0f),
            sizeDp  = prefs.getInt  ("p${index}_size", 110)
        )
    }

    /** 全 4 プリセットを 0-indexed List で返す（未保存スロットは null）。 */
    fun loadAll(): List<BalloonPreset?> = List(4) { load(it + 1) }
}
