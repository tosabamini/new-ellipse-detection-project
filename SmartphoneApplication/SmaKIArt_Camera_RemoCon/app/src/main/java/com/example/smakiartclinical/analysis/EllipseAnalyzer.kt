package com.example.smakiartclinical.analysis

import android.graphics.Bitmap
import com.example.smakiartclinical.analysis.EllipseConstants.CROP_RATIO
import com.example.smakiartclinical.analysis.EllipseConstants.REF_LONG_SIDE
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

data class EllipseResult(
    val cxPx: Float,      // ellipse center x in full-bitmap pixel space
    val cyPx: Float,      // ellipse center y in full-bitmap pixel space
    val majorPx: Float,   // major axis length in pixels (ROI scale = bitmap scale)
    val minorPx: Float,   // minor axis length in pixels
    val angleDeg: Float,  // major-axis angle in degrees [0, 180)
    val ratio: Float,     // minor / major
    val dEst: Float?,     // estimated refraction D [diopters] (≤0; 0 = unmeasurable/emmetropia), null = solver failed
    val pEstMm: Float?,   // estimated pupil diameter [mm], or null
    val bitmapW: Int,     // bitmap width used for this analysis (for coordinate mapping)
    val bitmapH: Int      // bitmap height used for this analysis
)

/**
 * Port of the poly10 joint-solver pipeline (run_repeatability_pipeline / refraction_from_ratio_area)
 * to Android/OpenCV.
 *
 *   center_crop(0.2) → red_channel → stretch_to_255
 *     → core fit (DoG → Otsu → central blob → fitEllipse, **no dilation/close**)
 *     → RefractionModel.solveOne(ratio, area_norm) → D
 *
 * Thread-safe: stateless; all intermediate Mats are local.
 */
class EllipseAnalyzer {

    // ── Public entry point ────────────────────────────────────────────────────

    fun analyze(bitmap: Bitmap): EllipseResult? = try {
        val bmpW = bitmap.width
        val bmpH = bitmap.height

        // 1. Bitmap (ARGB_8888) → OpenCV RGBA Mat → BGR
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()

        // 2. Center crop (CROP_RATIO = 0.2)
        val h = bgr.rows(); val w = bgr.cols()
        val cropW = (w * CROP_RATIO).toInt().coerceAtLeast(1)
        val cropH = (h * CROP_RATIO).toInt().coerceAtLeast(1)
        val roiX  = (w - cropW) / 2
        val roiY  = (h - cropH) / 2
        val roi   = Mat(bgr, Rect(roiX, roiY, cropW, cropH)).clone()
        bgr.release()

        // 3. Red channel: R − 0.5G − 0.5B, clipped to [0, 255]
        val red = redChannel(roi)
        roi.release()

        // 4. stretch_to_255
        val redStr = stretchTo255(red)
        red.release()

        // 5. Core ellipse fit (DoG → Otsu → central blob → fitEllipse, no dilation/close)
        val eRaw = fitCore(redStr) ?: run { redStr.release(); return null }
        redStr.release()

        // 6. Map ROI coords → full-bitmap coords
        val cxFull = (roiX + eRaw.cx).toFloat()
        val cyFull = (roiY + eRaw.cy).toFloat()

        // 7. ratio (scale-invariant) and area normalised to the 4000px-long-side
        //    calibration scale.  major/minor are in this bitmap's px; the area
        //    polynomial was fit on 4000px-wide patient images.
        val ratio    = eRaw.minor / eRaw.major
        val areaPx   = eRaw.major.toDouble() * eRaw.minor.toDouble()
        val scale    = REF_LONG_SIDE / max(bmpW, bmpH).toDouble()
        val areaNorm = areaPx * scale * scale

        // 8. poly10 joint solver → D (D≤0; 0 = unmeasurable near emmetropia)
        val sol = RefractionModel.solveOne(ratio.toDouble(), areaNorm)
        val dEst: Float? = when (sol.status) {
            RefractionModel.Status.FAILED -> null
            else -> sol.d.toFloat()
        }
        val pEst: Float? = if (sol.p.isNaN()) null else sol.p.toFloat()

        EllipseResult(
            cxPx      = cxFull,
            cyPx      = cyFull,
            majorPx   = eRaw.major,
            minorPx   = eRaw.minor,
            angleDeg  = eRaw.angle,
            ratio     = ratio,
            dEst      = dEst,
            pEstMm    = pEst,
            bitmapW   = bmpW,
            bitmapH   = bmpH
        )
    } catch (_: Exception) {
        null
    }

    // ── Image helpers ─────────────────────────────────────────────────────────

    private fun redChannel(bgr: Mat): Mat {
        val ch = ArrayList<Mat>()
        Core.split(bgr, ch)
        val bF = Mat(); ch[0].convertTo(bF, CvType.CV_32F)
        val gF = Mat(); ch[1].convertTo(gF, CvType.CV_32F)
        val rF = Mat(); ch[2].convertTo(rF, CvType.CV_32F)

        // temp = −0.5·G − 0.5·B
        val temp = Mat()
        Core.addWeighted(gF, -0.5, bF, -0.5, 0.0, temp)

        // result = R + temp → clip negatives
        val raw = Mat()
        Core.add(rF, temp, raw)
        val clipped = Mat()
        Imgproc.threshold(raw, clipped, 0.0, 255.0, Imgproc.THRESH_TOZERO)

        val out = Mat()
        clipped.convertTo(out, CvType.CV_8U)

        bF.release(); gF.release(); rF.release()
        temp.release(); raw.release(); clipped.release()
        ch.forEach { it.release() }
        return out
    }

    private fun stretchTo255(src: Mat): Mat {
        val mmr = Core.minMaxLoc(src)
        if (mmr.maxVal == mmr.minVal) return Mat.zeros(src.size(), CvType.CV_8U)
        val scale = 255.0 / (mmr.maxVal - mmr.minVal)
        val shift = -mmr.minVal * scale
        val out = Mat()
        src.convertTo(out, CvType.CV_8U, scale, shift)
        return out
    }

    // ── AdaptDoG core ─────────────────────────────────────────────────────────

    private data class EllipseRaw(val cx: Float, val cy: Float,
                                   val major: Float, val minor: Float, val angle: Float)

    /**
     * Current method: fit the ellipse directly on `mask_core` — **no dilation/close**.
     * Mirrors `core_fit()` in run_repeatability_pipeline.py / pickup_mask_core_fit.py.
     * The legacy `run_adaptive_dog` (Final/dilated mask) over-estimated major/minor on
     * thin reflexes and is deprecated.
     */
    private fun fitCore(redStr: Mat): EllipseRaw? {
        val minorEst = estimateMinor(redStr)
        val sigmaL   = maxOf(8.0, minorEst * 0.75)

        // DoG = GaussianBlur(σ=1.5) − GaussianBlur(σ=sigmaL), clip negatives
        val redF = Mat(); redStr.convertTo(redF, CvType.CV_32F)
        val blurS = Mat(); Imgproc.GaussianBlur(redF, blurS, Size(0.0, 0.0), 1.5)
        val blurL = Mat(); Imgproc.GaussianBlur(redF, blurL, Size(0.0, 0.0), sigmaL)
        redF.release()

        val dogRaw = Mat(); Core.subtract(blurS, blurL, dogRaw)
        blurS.release(); blurL.release()

        val dogClipped = Mat()
        Imgproc.threshold(dogRaw, dogClipped, 0.0, 255.0, Imgproc.THRESH_TOZERO)
        dogRaw.release()

        val dogStr = stretchTo255(dogClipped)
        dogClipped.release()

        // Otsu → central blob → fit directly (mask_core)
        val maskRaw = Mat()
        Imgproc.threshold(dogStr, maskRaw, 0.0, 255.0,
            Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        dogStr.release()

        val maskCore = pickCentralBlob(maskRaw)
        maskRaw.release()

        val result = fitEllipseOnMask(maskCore)
        maskCore.release()
        return result
    }

    private fun estimateMinor(redStr: Mat, topPct: Double = 0.005): Float {
        val thresh  = percentile8U(redStr, 100.0 * (1.0 - topPct))
        val coarse  = Mat()
        Imgproc.threshold(redStr, coarse, thresh, 255.0, Imgproc.THRESH_BINARY)
        val blob = pickCentralBlob(coarse, minArea = 10, openK = 3, closeK = 3)
        coarse.release()
        val e = fitEllipseOnMask(blob)
        blob.release()
        return if (e != null) maxOf(6f, e.minor) else 12f
    }

    private fun pickCentralBlob(
        binary: Mat, minArea: Int = 30, openK: Int = 5, closeK: Int = 9
    ): Mat {
        val kOpen  = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, Size(openK.toDouble(), openK.toDouble()))
        val kClose = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, Size(closeK.toDouble(), closeK.toDouble()))
        val m = Mat()
        Imgproc.morphologyEx(binary, m, Imgproc.MORPH_OPEN,  kOpen)
        Imgproc.morphologyEx(m,      m, Imgproc.MORPH_CLOSE, kClose)
        kOpen.release(); kClose.release()

        val src = m.clone()
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(src, contours, Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        src.release(); m.release()

        if (contours.isEmpty()) return Mat.zeros(binary.size(), binary.type())

        val cx0 = binary.cols() / 2.0
        val cy0 = binary.rows() / 2.0
        var best: MatOfPoint? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < minArea) continue
            val mom = Imgproc.moments(c)
            if (mom.m00 == 0.0) continue
            val cx = mom.m10 / mom.m00
            val cy = mom.m01 / mom.m00
            val score = area - 0.5 * ((cx - cx0) * (cx - cx0) + (cy - cy0) * (cy - cy0))
            if (score > bestScore) { bestScore = score; best = c }
        }

        val out = Mat.zeros(binary.size(), binary.type())
        if (best != null) {
            Imgproc.drawContours(out, listOf(best), 0, Scalar(255.0), Imgproc.FILLED)
        }
        return out
    }

    private fun fitEllipseOnMask(mask: Mat): EllipseRaw? {
        val src = mask.clone()
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(src, contours, Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        src.release()

        if (contours.isEmpty()) return null
        val cnt = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        if (cnt.rows() < 5) return null

        val cnt2f = MatOfPoint2f(); cnt2f.fromList(cnt.toList())
        val rect  = Imgproc.fitEllipse(cnt2f)
        cnt2f.release()

        val a1 = rect.size.width.toFloat()
        val a2 = rect.size.height.toFloat()
        var ang = rect.angle.toFloat()
        val major = maxOf(a1, a2)
        val minor = minOf(a1, a2)
        if (a2 > a1) ang += 90f

        return EllipseRaw(
            cx    = rect.center.x.toFloat(),
            cy    = rect.center.y.toFloat(),
            major = major,
            minor = minor,
            angle = ang % 180f
        )
    }

    // ── Percentile helper ─────────────────────────────────────────────────────

    private fun percentile8U(mat: Mat, pct: Double): Double {
        val rows = mat.rows(); val cols = mat.cols()
        val n    = rows * cols
        val vals = IntArray(n)
        var idx  = 0
        val row  = ByteArray(cols)
        for (r in 0 until rows) {
            mat.get(r, 0, row)
            for (b in row) vals[idx++] = b.toInt() and 0xFF
        }
        vals.sort()
        val pIdx = ((pct / 100.0) * (n - 1)).toInt().coerceIn(0, n - 1)
        return vals[pIdx].toDouble()
    }
}
