import cv2
import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("myface.feature_extraction")

class FeatureExtractor:
    """
    Extracts LBP and HOG face descriptors and projects them onto PCA Eigenspaces.
    
    Mathematical Principles:
    1. Local Binary Pattern (LBP):
       For each center pixel c and its 8 circular neighbors p:
         b_p = 1 if I(p) >= I(c) else 0
       The binary LBP code is compiled as:
         LBP(c) = Sum_{p=0}^7 ( b_p * 2^p )
       Histograms of LBP codes are calculated over cells and L2 normalized.
    2. Histogram of Oriented Gradients (HOG):
       Gradients Gx and Gy are calculated. Angle orientation is grouped into unsigned bins [0, 180].
       Votes are accumulated weighted by gradient magnitude. Blocks of 2x2 cells are normalized.
    3. Feature Fusion and Normalization:
       LBP and HOG descriptors are concatenated, then normalized to unit length:
         V_norm = V / (||V||_2 + epsilon)
    4. PCA SVD training:
       Given centered data matrix X, Singular Value Decomposition (SVD) solves:
         X = U * S * V^T
       The eigenvectors are the right singular vectors (rows of Vt).
    5. PCA Projection:
       Projects raw combined vector V onto Eigenspace:
         V_projected = Vt * (V - mean)
    """
    def __init__(self, lbp_grid: Tuple[int, int] = (8, 8), hog_bins: int = 9):
        self.lbp_grid = lbp_grid
        self.hog_bins = hog_bins

    def extract_lbp_histogram(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Computes spatial grid-based Local Binary Patterns (LBP) histograms.
        
        Args:
            gray_face: Grayscale input face crop.
            
        Returns:
            np.ndarray: 1D LBP feature vector.
        """
        try:
            h, w = gray_face.shape
            lbp_img = np.zeros((h - 2, w - 2), dtype=np.uint8)
            
            # Vectorized LBP check across 8-neighborhood
            center = gray_face[1:-1, 1:-1]
            neighbors = [
                gray_face[0:-2, 0:-2],  # Top-Left
                gray_face[0:-2, 1:-1],  # Top-Center
                gray_face[0:-2, 2:],    # Top-Right
                gray_face[1:-1, 2:],    # Right-Center
                gray_face[2:, 2:],      # Bottom-Right
                gray_face[2:, 1:-1],    # Bottom-Center
                gray_face[2:, 0:-2],    # Bottom-Left
                gray_face[1:-1, 0:-2]   # Left-Center
            ]
            
            for p, neighbor in enumerate(neighbors):
                # Set bit flags
                lbp_img += ((neighbor >= center).astype(np.uint8) << p)
            
            # Divide into grid cells
            grid_rows, grid_cols = self.lbp_grid
            cell_h = (h - 2) // grid_rows
            cell_w = (w - 2) // grid_cols
            
            lbp_trimmed = lbp_img[:cell_h * grid_rows, :cell_w * grid_cols]
            
            histograms = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    cell = lbp_trimmed[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                    # Compute 256-bin histogram
                    hist, _ = np.histogram(cell, bins=range(257))
                    # L2 Normalization
                    norm = np.sqrt(np.sum(hist**2) + 1e-6)
                    histograms.append(hist.astype(np.float32) / norm)
                    
            return np.concatenate(histograms)
        except Exception as e:
            logger.error(f"LBP histogram extraction failed: {e}")
            raise e

    def extract_hog_descriptor(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Computes spatial grid Histogram of Oriented Gradients (HOG) descriptor.
        
        Args:
            gray_face: Grayscale input face crop.
            
        Returns:
            np.ndarray: 1D HOG feature vector.
        """
        try:
            # 1. Compute Sobel gradients
            gx = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx) * (180.0 / np.pi)
            angle = np.where(angle < 0, angle + 180.0, angle)
            
            h, w = gray_face.shape
            cell_size = 16
            num_cells_y = h // cell_size
            num_cells_x = w // cell_size
            num_bins = self.hog_bins
            
            histograms = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
            bin_width = 180.0 / num_bins
            
            # 2. Accumulate histogram votes for each cell
            for cy in range(num_cells_y):
                for cx in range(num_cells_x):
                    cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    cell_ang = angle[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    
                    bins_float = cell_ang / bin_width
                    bin_idx1 = np.floor(bins_float).astype(np.int32) % num_bins
                    bin_idx2 = (bin_idx1 + 1) % num_bins
                    
                    weight2 = bins_float - np.floor(bins_float)
                    weight1 = 1.0 - weight2
                    
                    np.add.at(histograms[cy, cx], bin_idx1, cell_mag * weight1)
                    np.add.at(histograms[cy, cx], bin_idx2, cell_mag * weight2)
            
            # 3. Block normalization (2x2 cells block, overlap of 1 cell)
            block_feats = []
            epsilon = 1e-6
            for by in range(num_cells_y - 1):
                for bx in range(num_cells_x - 1):
                    block = histograms[by:by+2, bx:bx+2].flatten()
                    norm = np.sqrt(np.sum(block**2) + epsilon**2)
                    block_feats.append(block / norm)
                    
            return np.concatenate(block_feats)
        except Exception as e:
            logger.error(f"HOG descriptor extraction failed: {e}")
            raise e

    def train_pca(self, feature_matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains a personalized Eigenspace projection using Singular Value Decomposition.
        
        Args:
            feature_matrix: Matrix of shape (N, D) representing N training descriptors.
            k: Target projection dimension.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (eigenvectors, mean) projection coordinates.
        """
        try:
            # 1. Compute mean
            mean_vector = np.mean(feature_matrix, axis=0)
            
            # 2. Subtract mean
            X_centered = feature_matrix - mean_vector
            
            # 3. Run Singular Value Decomposition
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # 4. Extract top k components
            eigenvectors = Vt[:k, :]
            
            logger.info(f"Trained PCA eigenspace. Eigenvectors shape: {eigenvectors.shape}")
            return eigenvectors, mean_vector
        except Exception as e:
            logger.error(f"PCA training failed: {e}")
            raise e

    def project_pca(self, feature_vector: np.ndarray, eigenvectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """
        Projects a high-dimensional feature vector onto the user's PCA eigenspace.
        
        Args:
            feature_vector: Combined normalized raw vector.
            eigenvectors: Enrolled PCA projection matrix of shape (k, d).
            mean: Average feature vector calculated during SVD training.
            
        Returns:
            np.ndarray: Lower dimensional coordinate projection vector of shape (k,).
        """
        try:
            centered = feature_vector - mean
            projection = np.dot(eigenvectors, centered)
            return projection
        except Exception as e:
            logger.error(f"PCA Eigenspace projection failed: {e}")
            raise e

    def extract_features(
        self,
        gray_face: np.ndarray,
        eigenvectors: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extracts LBP+HOG features, normalizes them, and optionally projects onto PCA Eigenspace.
        
        Args:
            gray_face: Aligned grayscale crop of shape (128, 128).
            eigenvectors: Enrolled PCA matrix.
            mean: Enrolled training mean vector.
            
        Returns:
            np.ndarray: Final feature descriptor vector.
        """
        try:
            logger.debug("Extracting raw spatial LBP features...")
            lbp = self.extract_lbp_histogram(gray_face)
            
            logger.debug("Extracting raw spatial HOG features...")
            hog = self.extract_hog_descriptor(gray_face)
            
            # Combine raw descriptors (Feature Fusion)
            combined_raw = np.concatenate([lbp, hog])
            
            # Normalize vector to unit length (Feature Normalization)
            norm = np.linalg.norm(combined_raw) + 1e-6
            normalized_raw = combined_raw / norm
            
            if eigenvectors is not None and mean is not None:
                logger.debug("Projecting features onto PCA Eigenspace...")
                return self.project_pca(normalized_raw, eigenvectors, mean)
                
            return normalized_raw
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise e
