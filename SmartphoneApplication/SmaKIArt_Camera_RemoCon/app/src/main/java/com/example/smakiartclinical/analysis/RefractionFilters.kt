package com.example.smakiartclinical.analysis

import com.example.smakiartclinical.analysis.EllipseConstants.D_IQR_K
import com.example.smakiartclinical.analysis.EllipseConstants.IQR_K

/**
 * Pre-SCA outlier filters — Kotlin port of `iqr_filter` / `d_iqr_filter`
 * (src/ellipse/adaptdog.py) as used by run_repeatability_pipeline.
 *
 * Apply order (same as Python):
 *   1. [iqrFilterMajor]  — drop images whose major axis is a low outlier
 *                          (very small / thin red reflex).
 *   2. joint solver → D per surviving image.
 *   3. [dIqrFilter]      — drop D outliers within each angle bin.
 */
object RefractionFilters {

    /** Per-image record carried through the filtering pipeline. */
    data class Item(
        val angleDeg: Float,
        val majorPx: Float,
        var dEst: Float = Float.NaN
    )

    // numpy.percentile with linear interpolation (default method).
    private fun percentile(sortedAsc: DoubleArray, q: Double): Double {
        val n = sortedAsc.size
        if (n == 0) return Double.NaN
        if (n == 1) return sortedAsc[0]
        val rank = q / 100.0 * (n - 1)
        val lo = Math.floor(rank).toInt()
        val hi = Math.ceil(rank).toInt()
        if (lo == hi) return sortedAsc[lo]
        return sortedAsc[lo] + (rank - lo) * (sortedAsc[hi] - sortedAsc[lo])
    }

    /**
     * Keep images whose major axis ≥ Q1 − k·IQR (k = IQR_K = 0.5).
     * Fewer than 4 images → keep all. ratio is NOT filtered.
     */
    fun iqrFilterMajor(items: List<Item>, k: Double = IQR_K.toDouble()): List<Item> {
        if (items.size < 4) return items
        val majors = items.map { it.majorPx.toDouble() }.sorted().toDoubleArray()
        val q1 = percentile(majors, 25.0)
        val q3 = percentile(majors, 75.0)
        val fence = q1 - k * (q3 - q1)
        return items.filter { it.majorPx.toDouble() >= fence }
    }

    /** Angle → bin name (matches Python `_angle_bin`). */
    private fun angleBin(deg: Float): String {
        var a = deg % 180f
        if (a < 0) a += 180f
        return when {
            a in 70f..109.9999f -> "90deg"
            a in 30f..59.9999f  -> "45deg"
            a < 20f || a >= 160f -> "0deg"
            else                 -> "other"
        }
    }

    /**
     * IQR filter on D within each angle bin separately (k = D_IQR_K = 1.5).
     * Bins with < 4 images are kept as-is. Items must already have [Item.dEst] set.
     */
    fun dIqrFilter(items: List<Item>, k: Double = D_IQR_K.toDouble()): List<Item> {
        val bins = items.groupBy { angleBin(it.angleDeg) }
        val drop = HashSet<Item>()
        for ((_, group) in bins) {
            if (group.size < 4) continue
            val vals = group.map { it.dEst.toDouble() }.sorted().toDoubleArray()
            val q1 = percentile(vals, 25.0)
            val q3 = percentile(vals, 75.0)
            val iqr = q3 - q1
            val lo = q1 - k * iqr
            val hi = q3 + k * iqr
            for (it in group) {
                val v = it.dEst.toDouble()
                if (v < lo || v > hi) drop.add(it)
            }
        }
        return items.filter { it !in drop }
    }
}
