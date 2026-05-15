package com.example.smakiartclinical.analysis

/**
 * Calibration constants for the v150526 geometry-only pipeline.
 * Update these when reference data (model eye) is retaken.
 */
object EllipseConstants {

    // ── Scale factor ──────────────────────────────────────────────────────────
    // 暫定: patient image px / model-eye image px ratio
    const val SCALE_FACTOR = 1.3f

    // ── Pupil estimation (area = slope(p)*ratio + intercept(p)) ──────────────
    // slope(p)     = S2*p^2 + S1*p + S0
    // intercept(p) = I2*p^2 + I1*p + I0
    const val S2 =  928.28f;  const val S1 = 1780.95f;  const val S0 = -872.10f
    const val I2 = -462.23f;  const val I1 = 3344.24f;  const val I0 = -4477.24f

    // ── Pupil validity range ──────────────────────────────────────────────────
    const val P_MIN = 2.0f
    const val P_MAX = 9.0f

    // ── D estimation: a(p)*D^2 + b(p)*D + (c(p) - ratio) = 0 ────────────────
    // a(p) = A2*p^2 + A1*p + A0
    const val A2 = -0.000250028726793751f
    const val A1 =  0.004161575675553433f
    const val A0 =  0.007984965068556180f

    // b(p) = B2*p^2 + B1*p + B0
    const val B2 =  0.001737698861861923f
    const val B1 = -0.020599458135334574f
    const val B0 =  0.031627036530218920f

    // c(p) = C2*p^2 + C1*p + C0
    const val C2 =  0.001959184323888975f
    const val C1 = -0.011793575764022856f
    const val C0 =  0.145406184754346270f

    // ── Crop ─────────────────────────────────────────────────────────────────
    const val CROP_RATIO = 0.2f
}
