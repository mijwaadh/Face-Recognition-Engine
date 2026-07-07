import pytest
import numpy as np
import os
from myface.evaluation.metrics import BiometricEvaluator

def test_metrics_computations():
    """Verifies baseline FAR, FRR, Accuracy, Precision, Recall, and F1 calculations."""
    evaluator = BiometricEvaluator()
    
    genuine_scores = [0.85, 0.90, 0.95, 0.60] # 3 pass, 1 fail at thresh=0.80
    imposter_scores = [0.10, 0.20, 0.30, 0.85] # 1 pass, 3 fail at thresh=0.80
    
    m = evaluator.compute_metrics(genuine_scores, imposter_scores, threshold=0.80)
    
    # Genuine count = 4, Imposter count = 4
    # TP = 3 (0.85, 0.90, 0.95)
    # FN = 1 (0.60)
    # FP = 1 (0.85)
    # TN = 3 (0.10, 0.20, 0.30)
    
    assert m["far"] == 0.25  # 1/4
    assert m["frr"] == 0.25  # 1/4
    assert m["accuracy"] == 0.75 # 6/8
    assert m["precision"] == 0.75 # 3/4
    assert m["recall"] == 0.75 # 3/4
    assert abs(m["f1"] - 0.75) < 1e-5

def test_find_eer():
    """Verifies EER sweep finds the optimal boundary matching FAR and FRR."""
    evaluator = BiometricEvaluator()
    
    genuine_scores = [0.80, 0.85, 0.90, 0.95]
    imposter_scores = [0.10, 0.20, 0.30, 0.40]
    
    eer, opt_t = evaluator.find_eer(genuine_scores, imposter_scores, steps=100)
    # The EER should be exactly 0.0 because there is no overlap in distributions
    assert eer == 0.0
    # The optimal threshold should be in the separation gap (0.40 to 0.80)
    assert 0.40 < opt_t < 0.80

def test_roc_and_auc():
    """Verifies ROC curve coordinates and trapezoidal area integration."""
    evaluator = BiometricEvaluator()
    
    genuine_scores = [0.80, 0.85, 0.90, 0.95]
    imposter_scores = [0.10, 0.20, 0.30, 0.40]
    
    pts, auc = evaluator.compute_roc_and_auc(genuine_scores, imposter_scores, steps=50)
    assert len(pts) == 50
    # No overlap means perfect separation: AUC must be 1.0
    assert abs(auc - 1.0) < 1e-4

def test_visual_and_markdown_reports(tmp_path):
    """Verifies PNG graphic rendering and markdown report generation on disk."""
    evaluator = BiometricEvaluator()
    
    genuine_scores = [0.75, 0.80, 0.85, 0.90, 0.95]
    imposter_scores = [0.10, 0.15, 0.20, 0.25, 0.80]
    
    img_path = str(tmp_path / "roc_curve.png")
    report_path = str(tmp_path / "recommendation_report.md")
    
    # 1. Generate PNG graphical plot
    evaluator.generate_roc_curve_image(genuine_scores, imposter_scores, img_path)
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 0
    
    # 2. Generate Markdown recommendation report
    report = evaluator.generate_threshold_report(genuine_scores, imposter_scores, report_path)
    assert os.path.exists(report_path)
    assert "Biometric Threshold Recommendation Report" in report
    assert "EER" in report
    assert "AUC" in report
