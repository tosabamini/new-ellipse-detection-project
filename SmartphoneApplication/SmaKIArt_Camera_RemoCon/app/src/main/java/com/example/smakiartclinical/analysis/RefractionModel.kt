package com.example.smakiartclinical.analysis

import com.example.smakiartclinical.analysis.EllipseConstants.COEF_AREA
import com.example.smakiartclinical.analysis.EllipseConstants.COEF_RATIO
import com.example.smakiartclinical.analysis.EllipseConstants.DEG
import com.example.smakiartclinical.analysis.EllipseConstants.D_MAX
import com.example.smakiartclinical.analysis.EllipseConstants.D_MIN
import com.example.smakiartclinical.analysis.EllipseConstants.P_MAX
import com.example.smakiartclinical.analysis.EllipseConstants.P_MIN
import com.example.smakiartclinical.analysis.EllipseConstants.RATIO_THRESH

/**
 * poly10 joint solver — Kotlin port of `experiments/refraction_from_ratio_area.py`.
 *
 * Each image's observed (ratio, area) is fed to [solveOne], which finds the (D, p)
 * minimising
 *     L(D,p) = [(ratio_real(D,p) − ratio_obs)/ratio_obs]²
 *            + [(area_real (D,p) − area_obs )/area_obs ]²
 * over D ∈ [D_MIN, 0], p ∈ [P_MIN, P_MAX].
 *
 * Near emmetropia (ratio < RATIO_THRESH) the ratio contour is vertical → D unmeasurable;
 * D is fixed to 0 and p is recovered from area alone (bisection). status = UNMEASURABLE.
 *
 * Python uses scipy L-BFGS-B; here a coarse grid (65×29, identical to Python) seeds a
 * bounded Nelder–Mead simplex. For this smooth 2D loss the two agree to ≪0.01 D.
 */
object RefractionModel {

    enum class Status { OK, UNMEASURABLE, FAILED }

    data class Solution(
        val d: Double,        // refraction [diopters], ≤ 0 (myopic side)
        val p: Double,        // pupil diameter [mm]
        val status: Status
    )

    // ── Polynomial evaluation ────────────────────────────────────────────────
    // term order: for i in 0..DEG, for j in 0..(DEG-i): coef · D^i · p^j

    private fun evalPoly(coef: DoubleArray, d: Double, p: Double): Double {
        val dPow = DoubleArray(DEG + 1); dPow[0] = 1.0
        val pPow = DoubleArray(DEG + 1); pPow[0] = 1.0
        for (k in 1..DEG) { dPow[k] = dPow[k - 1] * d; pPow[k] = pPow[k - 1] * p }
        var acc = 0.0
        var idx = 0
        for (i in 0..DEG) {
            val di = dPow[i]
            for (j in 0..(DEG - i)) {
                acc += coef[idx] * di * pPow[j]
                idx++
            }
        }
        return acc
    }

    fun ratioReal(d: Double, p: Double): Double = evalPoly(COEF_RATIO, d, p)
    fun areaReal(d: Double, p: Double): Double = evalPoly(COEF_AREA, d, p)

    private fun loss(d: Double, p: Double, ratioObs: Double, areaObs: Double): Double {
        val r = (ratioReal(d, p) - ratioObs) / ratioObs
        val a = (areaReal(d, p) - areaObs) / areaObs
        return r * r + a * a
    }

    // ── Public entry point ────────────────────────────────────────────────────

    fun solveOne(ratioObs: Double, areaObs: Double): Solution {
        if (ratioObs <= 0.0 || areaObs <= 0.0) return Solution(Double.NaN, Double.NaN, Status.FAILED)

        if (ratioObs < RATIO_THRESH) {
            val p = solvePAtD0(areaObs)
            return if (p.isNaN()) Solution(Double.NaN, Double.NaN, Status.FAILED)
            else Solution(0.0, p, Status.UNMEASURABLE)
        }

        // Coarse grid for a robust initial guess (matches Python: 65 × 29).
        var bestLoss = Double.POSITIVE_INFINITY
        var bd = -1.0; var bp = 4.0
        for (gi in 0 until 65) {
            val d0 = D_MIN + (D_MAX - D_MIN) * gi / 64.0
            for (gj in 0 until 29) {
                val p0 = P_MIN + (P_MAX - P_MIN) * gj / 28.0
                val l = loss(d0, p0, ratioObs, areaObs)
                if (l < bestLoss) { bestLoss = l; bd = d0; bp = p0 }
            }
        }

        val (d, p) = nelderMead(bd, bp, ratioObs, areaObs)
        return Solution(d, p, Status.OK)
    }

    // ── D=0 fixed: recover p from area via bisection (Python brentq) ──────────

    private fun solvePAtD0(areaObs: Double): Double {
        val f = { p: Double -> areaReal(0.0, p) - areaObs }
        var lo = P_MIN; var hi = P_MAX
        var flo = f(lo); var fhi = f(hi)
        if (flo == 0.0) return lo
        if (fhi == 0.0) return hi
        if (flo * fhi > 0.0) return Double.NaN   // no sign change in range
        repeat(80) {
            val mid = 0.5 * (lo + hi)
            val fmid = f(mid)
            if (fmid == 0.0 || (hi - lo) < 1e-7) return mid
            if (flo * fmid < 0.0) { hi = mid; fhi = fmid } else { lo = mid; flo = fmid }
        }
        return 0.5 * (lo + hi)
    }

    // ── Bounded Nelder–Mead simplex (2D) ──────────────────────────────────────

    private fun clampD(x: Double) = x.coerceIn(D_MIN, D_MAX)
    private fun clampP(x: Double) = x.coerceIn(P_MIN, P_MAX)

    private fun nelderMead(
        d0: Double, p0: Double, ratioObs: Double, areaObs: Double
    ): Pair<Double, Double> {
        // Simplex vertices as (D, p); initial step sized to the grid spacing.
        val stepD = (D_MAX - D_MIN) / 64.0
        val stepP = (P_MAX - P_MIN) / 28.0
        val v = arrayOf(
            doubleArrayOf(d0, p0),
            doubleArrayOf(clampD(d0 + stepD), p0),
            doubleArrayOf(d0, clampP(p0 + stepP))
        )
        val fv = DoubleArray(3) { loss(v[it][0], v[it][1], ratioObs, areaObs) }

        val alpha = 1.0; val gamma = 2.0; val rho = 0.5; val sigma = 0.5

        repeat(300) {
            // order: best..worst
            val order = (0..2).sortedBy { fv[it] }
            val b = order[0]; val g = order[1]; val w = order[2]
            if (kotlin.math.abs(fv[w] - fv[b]) < 1e-14) return@repeat

            // centroid of all but worst
            val cd = 0.5 * (v[b][0] + v[g][0])
            val cp = 0.5 * (v[b][1] + v[g][1])

            // reflection
            val rd = clampD(cd + alpha * (cd - v[w][0]))
            val rp = clampP(cp + alpha * (cp - v[w][1]))
            val fr = loss(rd, rp, ratioObs, areaObs)

            if (fr < fv[b]) {
                // expansion
                val ed = clampD(cd + gamma * (rd - cd))
                val ep = clampP(cp + gamma * (rp - cp))
                val fe = loss(ed, ep, ratioObs, areaObs)
                if (fe < fr) { v[w][0] = ed; v[w][1] = ep; fv[w] = fe }
                else { v[w][0] = rd; v[w][1] = rp; fv[w] = fr }
            } else if (fr < fv[g]) {
                v[w][0] = rd; v[w][1] = rp; fv[w] = fr
            } else {
                // contraction
                val ccd = clampD(cd + rho * (v[w][0] - cd))
                val ccp = clampP(cp + rho * (v[w][1] - cp))
                val fc = loss(ccd, ccp, ratioObs, areaObs)
                if (fc < fv[w]) { v[w][0] = ccd; v[w][1] = ccp; fv[w] = fc }
                else {
                    // shrink toward best
                    for (idx in intArrayOf(g, w)) {
                        v[idx][0] = clampD(v[b][0] + sigma * (v[idx][0] - v[b][0]))
                        v[idx][1] = clampP(v[b][1] + sigma * (v[idx][1] - v[b][1]))
                        fv[idx] = loss(v[idx][0], v[idx][1], ratioObs, areaObs)
                    }
                }
            }
        }
        val best = (0..2).minByOrNull { fv[it] }!!
        return clampD(v[best][0]) to clampP(v[best][1])
    }
}
