package com.example.smakiartclinical.data

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.example.smakiartclinical.data.model.CameraSettings
import com.example.smakiartclinical.data.model.Preset
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "presets")

class PresetDataStore(private val context: Context) {

    companion object {
        const val MAX_PRESETS = 4

        private fun keyIso(i: Int) = intPreferencesKey("preset_${i}_iso")
        private fun keyExpTime(i: Int) = longPreferencesKey("preset_${i}_exp_time")
        private fun keyFocus(i: Int) = floatPreferencesKey("preset_${i}_focus")
        private fun keyExpComp(i: Int) = intPreferencesKey("preset_${i}_exp_comp")
        private fun keyName(i: Int) = stringPreferencesKey("preset_${i}_name")
        private fun keyExists(i: Int) = stringPreferencesKey("preset_${i}_exists")
    }

    val presetsFlow: Flow<List<Preset?>> = context.dataStore.data.map { prefs ->
        (0 until MAX_PRESETS).map { i ->
            if (prefs[keyExists(i)] == "1") {
                Preset(
                    index = i,
                    name = prefs[keyName(i)] ?: "Preset ${i + 1}",
                    settings = CameraSettings(
                        iso = prefs[keyIso(i)] ?: 100,
                        exposureTimeNs = prefs[keyExpTime(i)] ?: 33_000_000L,
                        focusDistance = prefs[keyFocus(i)] ?: 0f,
                        exposureCompensation = prefs[keyExpComp(i)] ?: 0
                    )
                )
            } else null
        }
    }

    suspend fun savePreset(index: Int, name: String, settings: CameraSettings) {
        context.dataStore.edit { prefs ->
            prefs[keyExists(index)] = "1"
            prefs[keyName(index)] = name
            prefs[keyIso(index)] = settings.iso
            prefs[keyExpTime(index)] = settings.exposureTimeNs
            prefs[keyFocus(index)] = settings.focusDistance
            prefs[keyExpComp(index)] = settings.exposureCompensation
        }
    }

    suspend fun deletePreset(index: Int) {
        context.dataStore.edit { prefs ->
            prefs.remove(keyExists(index))
            prefs.remove(keyName(index))
            prefs.remove(keyIso(index))
            prefs.remove(keyExpTime(index))
            prefs.remove(keyFocus(index))
            prefs.remove(keyExpComp(index))
        }
    }
}
