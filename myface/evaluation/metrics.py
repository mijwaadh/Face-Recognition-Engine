import cv2
import logging
import os
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("myface.evaluation")

class BiometricEvaluator:
    """
    Computes biometric performance metrics and generates visual evaluation reports.
    
    Mathematical Principles:
    1. Error Rates:
       - FAR (False Acceptance Rate) = FP / (FP + TN) = Imposters accepted / Total imposters
       - FRR (False Rejection Rate) = FN / (TP + FN) = Genuines rejected / Total genuines
       - Accuracy = (TP + TN) / (P + N)
       - Precision = TP / (TP + FP)
       - Recall (TPR) = TP / P
       - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    2. EER (Equal Error Rate):
       Sweep thresholds to find the point minimizing Abs(FAR - FRR).
    3. AUC (Area Under Curve):
       Integrates the ROC curve using the Trapezoidal Rule:
         AUC = Sum_i ( (TPR_i + TPR_{i-1}) / 2 * (FPR_i - FPR_{i-1}) )
    4. Custom Graphical Plotter:
       Draws a grid, axes, labels, chance line, and the empirical ROC line on a 512x512 canvas.
    """
    def __init__(self):
        pass

    def compute_metrics(
        self,
        genuine_scores: List[float],
        imposter_scores: List[float],
        threshold: float
    ) -> Dict[str, float]:
        """
        Calculates biometric metrics for a given decision threshold.
        """
        try:
            gen_arr = np.array(genuine_scores) if genuine_scores else np.array([])
            imp_arr = np.array(imposter_scores) if imposter_scores else np.array([])

            total_genuine = len(gen_arr)
            total_imposter = len(imp_arr)

            tp = int(np.sum(gen_arr >= threshold)) if total_genuine > 0 else 0
            fn = total_genuine - tp
            fp = int(np.sum(imp_arr >= threshold)) if total_imposter > 0 else 0
            tn = total_imposter - fp

            # Handle edge cases
            denom_genuine = total_genuine if total_genuine > 0 else 1
            denom_imposter = total_imposter if total_imposter > 0 else 1

            far = float(fp / denom_imposter)
            frr = float(fn / denom_genuine)
            
            accuracy = float((tp + tn) / (denom_genuine + denom_imposter))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / denom_genuine)
            
            f1 = float(2.0 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

            return {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far": far,
                "frr": frr
            }
        except Exception as e:
            logger.error(f"Failed to calculate biometric metrics: {e}")
            raise e

    def find_eer(
        self,
        genuine_scores: List[float],
        imposter_scores: List[float],
        steps: int = 200
    ) -> Tuple[float, float]:
        """
        Calculates the Equal Error Rate (EER) and the optimal threshold.
        """
        try:
            thresholds = np.linspace(0.0, 1.0, steps)
            min_diff = float("inf")
            eer = 0.5
            opt_thresh = 0.5

            for t in thresholds:
                metrics = self.compute_metrics(genuine_scores, imposter_scores, t)
                diff = abs(metrics["far"] - metrics["frr"])
                if diff < min_diff:
                    min_diff = diff
                    eer = float((metrics["far"] + metrics["frr"]) / 2.0)
                    opt_thresh = float(t)
            
            logger.info(f"Calculated offline EER: {eer:.4f} at threshold: {opt_thresh:.4f}")
            return eer, opt_thresh
        except Exception as e:
            logger.error(f"EER calculations failed: {e}")
            raise e

    def compute_roc_and_auc(
        self,
        genuine_scores: List[float],
        imposter_scores: List[float],
        steps: int = 200
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Calculates ROC points (FPR, TPR) and computes the Area Under the Curve (AUC).
        """
        try:
            # Sweep thresholds from 1.0 down to 0.0 to traverse ROC from left to right
            thresholds = np.linspace(1.0, 0.0, steps)
            roc_points = []

            for t in thresholds:
                m = self.compute_metrics(genuine_scores, imposter_scores, t)
                fpr = m["far"]  # False Positive Rate
                tpr = m["recall"]  # True Positive Rate
                roc_points.append((fpr, tpr))

            # Compute AUC using Trapezoidal Rule integration
            auc = 0.0
            for i in range(1, len(roc_points)):
                fpr_prev, tpr_prev = roc_points[i - 1]
                fpr_curr, tpr_curr = roc_points[i]
                
                # Area of trapezoid: (tpr_curr + tpr_prev) / 2 * delta_fpr
                delta_fpr = fpr_curr - fpr_prev
                auc += (tpr_curr + tpr_prev) * 0.5 * delta_fpr

            auc = float(np.clip(auc, 0.0, 1.0))
            return roc_points, auc
        except Exception as e:
            logger.error(f"ROC/AUC calculation failed: {e}")
            raise e

    def generate_roc_curve_image(
        self,
        genuine_scores: List[float],
        imposter_scores: List[float],
        output_path: str
    ) -> None:
        """
        Renders a graphical ROC curve plot and saves it to disk as a PNG image.
        """
        try:
            # 1. Calculate stats
            roc_points, auc = self.compute_roc_and_auc(genuine_scores, imposter_scores)
            eer, opt_t = self.find_eer(genuine_scores, imposter_scores)

            # 2. Initialize clean white canvas
            img_size = 512
            margin = 60
            plot_size = img_size - 2 * margin
            
            canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 245 # Light grey background

            # Draw white plotting box
            cv2.rectangle(
                canvas, 
                (margin, margin), 
                (img_size - margin, img_size - margin), 
                (255, 255, 255), 
                -1
            )

            # 3. Draw grid lines (every 20%)
            for i in range(1, 5):
                val = i * 0.2
                pos = int(margin + val * plot_size)
                # Vertical grid lines
                cv2.line(canvas, (pos, margin), (pos, img_size - margin), (220, 220, 220), 1)
                # Horizontal grid lines
                cv2.line(canvas, (margin, pos), (img_size - margin, pos), (220, 220, 220), 1)
                
                # Text labels on axes
                cv2.putText(canvas, f"{val:.1f}", (pos - 15, img_size - margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
                cv2.putText(canvas, f"{1.0 - val:.1f}", (margin - 35, pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            # Origin/Endpoint labels
            cv2.putText(canvas, "0.0", (margin - 10, img_size - margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
            cv2.putText(canvas, "1.0", (img_size - margin - 10, img_size - margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
            cv2.putText(canvas, "1.0", (margin - 35, margin + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            # Draw plotting boundary box
            cv2.rectangle(
                canvas, 
                (margin, margin), 
                (img_size - margin, img_size - margin), 
                (100, 100, 100), 
                1
            )

            # 4. Draw chance diagonal line (FPR = TPR)
            cv2.line(
                canvas, 
                (margin, img_size - margin), 
                (img_size - margin, margin), 
                (150, 150, 150), 
                1, 
                lineType=cv2.LINE_AA
            )

            # 5. Draw ROC curve line points
            pts = []
            for fpr, tpr in roc_points:
                x = int(margin + fpr * plot_size)
                y = int(img_size - margin - tpr * plot_size)
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, (220, 80, 40), 2, lineType=cv2.LINE_AA) # Red-orange curve

            # 6. Title and labels
            cv2.putText(canvas, "ROC Curve Profile", (margin, margin - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"AUC: {auc:.4f} | EER: {eer:.4f}", (margin, margin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
            
            # X label: False Positive Rate (FAR)
            cv2.putText(canvas, "False Positive Rate (FPR)", (img_size // 2 - 80, img_size - margin + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)
            # Y label: True Positive Rate (Recall)
            # Draw rotated text by copying/pasting rotated region or just draw vertically
            cv2.putText(canvas, "True Positive Rate (TPR)", (15, margin + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, canvas)
            logger.info(f"Saved ROC curve image plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate visual ROC report image: {e}")
            raise e

    def generate_threshold_report(
        self,
        genuine_scores: List[float],
        imposter_scores: List[float],
        report_path: str
    ) -> str:
        """
        Generates a threshold recommendation report and saves it to disk as markdown.
        """
        try:
            eer, opt_t = self.find_eer(genuine_scores, imposter_scores)
            _, auc = self.compute_roc_and_auc(genuine_scores, imposter_scores)

            # Sweeps standard thresholds
            target_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
            rows = []
            
            for t in target_thresholds:
                m = self.compute_metrics(genuine_scores, imposter_scores, t)
                rows.append(
                    f"| {t:.2f} | {m['far']*100:.2f}% | {m['frr']*100:.2f}% | {m['accuracy']*100:.2f}% | {m['f1']:.4f} |"
                )

            # Build recommendations
            # High security: threshold where FAR <= 0.1% (or closest minimum)
            high_sec_thresh = 0.85
            for t in np.linspace(0.5, 0.95, 46):
                m = self.compute_metrics(genuine_scores, imposter_scores, t)
                if m["far"] <= 0.001:
                    high_sec_thresh = t
                    break

            # Convenience: threshold where FRR <= 1% (or closest minimum)
            convenience_thresh = 0.60
            for t in np.linspace(0.5, 0.95, 46):
                m = self.compute_metrics(genuine_scores, imposter_scores, t)
                if m["frr"] <= 0.01:
                    convenience_thresh = t
                    # Don't break immediately, find highest threshold satisfying FRR <= 1%
                    
            report_content = f"""# Biometric Threshold Recommendation Report

This report evaluates system similarity scores and calibrates decision thresholds to meet different security tolerances.

## Biometric System Summary
- **EER (Equal Error Rate):** {eer * 100:.2f}% at threshold **{opt_t:.4f}**
- **AUC (Area Under ROC Curve):** {auc:.4f}
- **Total Genuine Trials:** {len(genuine_scores)}
- **Total Imposter Trials:** {len(imposter_scores)}

## Threshold Sweep Analysis Table

| Threshold | FAR (False Accept %) | FRR (False Reject %) | Accuracy % | F1-Score |
|-----------|----------------------|----------------------|------------|----------|
{os.linesep.join(rows)}

## Recommended Settings Profiles

### 🔒 High Security (e.g., Financial Transactions)
- **Recommended Threshold:** `{high_sec_thresh:.3f}`
- **Target FAR:** `<= 0.10%`
- **Use Case:** Systems demanding absolute defense against imposters, accepting higher user re-tries.

### ⚖️ Balanced (General Purpose)
- **Recommended Threshold:** `{opt_t:.3f}` (EER Optimal)
- **Target FAR / FRR:** `{eer * 100:.2f}%`
- **Use Case:** Default configuration balancing ease of access with robust spoof rejection.

### ⚡ Maximum Convenience (e.g., Gaming/Personal Device Unlock)
- **Recommended Threshold:** `{convenience_thresh:.3f}`
- **Target FRR:** `<= 1.00%`
- **Use Case:** Personal devices prioritizing fast verification, accepting higher hypothetical spoof risk.
"""
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            logger.info(f"Saved threshold recommendation report to {report_path}")
            return report_content
        except Exception as e:
            logger.error(f"Failed to generate threshold report: {e}")
            raise e
