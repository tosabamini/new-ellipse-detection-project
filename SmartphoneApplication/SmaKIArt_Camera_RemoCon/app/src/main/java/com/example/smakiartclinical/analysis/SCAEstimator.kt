package com.example.smakiartclinical.analysis

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Cos-curve refraction fit (pipeline_v150526).
 *
 *   D(α) = P0 + P1·cos(2α) + P2·sin(2α)
 *   SE = P0
 *   C  = -2·√(P1² + P2²)         (cylinder, minus notation)
 *   S  = SE − C/2
 *   A  = ½·atan2(−P2, −P1)  mod 180°  (cylinder axis)
 */
data class SCAResult(
    val sphere: Float,
    val cylinder: Float,
    val axisDeg: Float,
    val se: Float,
    val r2: Float,
    val n: Int,
    val p0: Float, val p1: Float, val p2: Float,
    val samples: List<Sample>
) {
    data class Sample(val angleDeg: Float, val dEst: Float)
}

object SCAEstimator {

    const val MIN_VALID = 3

    fun fit(samples: List<SCAResult.Sample>): SCAResult? {
        if (samples.size < MIN_VALID) return null

        // Normal-equation accumulators for X = [1, cos(2α), sin(2α)]
        var n00 = samples.size.toDouble(); var n01 = 0.0; var n02 = 0.0
        var n11 = 0.0; var n12 = 0.0; var n22 = 0.0
        var y0 = 0.0; var y1 = 0.0; var y2 = 0.0
        for (s in samples) {
            val a2 = 2.0 * s.angleDeg * PI / 180.0
            val c = cos(a2); val si = sin(a2)
            n01 += c;       n02 += si
            n11 += c * c;   n12 += c * si;  n22 += si * si
            y0  += s.dEst;  y1  += s.dEst * c;  y2  += s.dEst * si
        }

        // Solve 3×3 symmetric system via Cramer's rule
        fun det3(
            a: Double, b: Double, c: Double,
            d: Double, e: Double, f: Double,
            g: Double, h: Double, i: Double
        ) = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

        val det  = det3(n00, n01, n02, n01, n11, n12, n02, n12, n22)
        if (abs(det) < 1e-9) return null
        val dP0  = det3(y0,  n01, n02, y1,  n11, n12, y2,  n12, n22)
        val dP1  = det3(n00, y0,  n02, n01, y1,  n12, n02, y2,  n22)
        val dP2  = det3(n00, n01, y0,  n01, n11, y1,  n02, n12, y2)
        val p0 = dP0 / det
        val p1 = dP1 / det
        val p2 = dP2 / det

        val se = p0
        val cyl = -2.0 * sqrt(p1 * p1 + p2 * p2)
        val sph = se - cyl / 2.0
        val axis = ((0.5 * atan2(-p2, -p1) * 180.0 / PI) + 360.0) % 180.0

        // R² (coefficient of determination)
        val yMean = y0 / samples.size
        var ssTot = 0.0; var ssRes = 0.0
        for (s in samples) {
            val a2 = 2.0 * s.angleDeg * PI / 180.0
            val pred = p0 + p1 * cos(a2) + p2 * sin(a2)
            ssTot += (s.dEst - yMean).pow(2)
            ssRes += (s.dEst - pred).pow(2)
        }
        val r2 = if (ssTot > 1e-9) (1.0 - ssRes / ssTot).coerceIn(-1.0, 1.0) else 0.0

        return SCAResult(
            sphere   = sph.toFloat(),
            cylinder = cyl.toFloat(),
            axisDeg  = axis.toFloat(),
            se       = se.toFloat(),
            r2       = r2.toFloat(),
            n        = samples.size,
            p0       = p0.toFloat(),
            p1       = p1.toFloat(),
            p2       = p2.toFloat(),
            samples  = samples
        )
    }

    /** Sample the fitted curve over [0°, 180°] for plotting. */
    fun curvePoints(result: SCAResult, n: Int = 181): List<Pair<Float, Float>> {
        val pts = ArrayList<Pair<Float, Float>>(n)
        for (i in 0 until n) {
            val angle = i.toFloat() * 180f / (n - 1)
            val a2 = 2.0 * angle * PI / 180.0
            val d = result.p0 + result.p1 * cos(a2).toFloat() + result.p2 * sin(a2).toFloat()
            pts += angle to d.toFloat()
        }
        return pts
    }
}
