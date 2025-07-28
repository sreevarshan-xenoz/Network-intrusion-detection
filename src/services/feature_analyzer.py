"""
Multi-method feature analysis system for model interpretability.
"""
import os
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature analysis will be limited.")

# LIME imports
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Will use SHAP as fallback.")

from ..models.interfaces import MLModel
from ..utils.config import config
from ..utils.logging import get_logger


@dataclass
class FeatureImportanceResult:
    """Result structure for feature importance analysis."""
    method: str
    feature_names: List[str]
    importance_scores: List[float]
    global_importance: Dict[str, float]
    local_explanations: Optional[List[Dict[str, Any]]] = None
    visualization_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelInterpretabilityReport:
    """Comprehensive model interpretability report."""
    model_name: str
    model_type: str
    feature_count: int
    sample_count: int
    analysis_methods: List[str]
    feature_importance_results: List[FeatureImportanceResult]
    top_features: Dict[str, List[Tuple[str, float]]]  # method -> [(feature, importance)]
    feature_interactions: Optional[Dict[str, Any]] = None
    summary_statistics: Optional[Dict[str, Any]] = None


class FeatureAnalyzer:
    """
    Multi-method feature analysis system using SHAP values with LIME fallback.
    Generates feature importance visualizations and model interpretability reports.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the feature analyzer."""
        self.config = config_dict or config.get('feature_analysis', {})
        self.logger = get_logger(__name__)
        
        # Analysis settings
        self.max_samples_for_analysis = self.config.get('max_samples', 1000)
        self.top_features_count = self.config.get('top_features_count', 15)
        self.visualization_dir = self.config.get('visualization_directory', 'visualizations')
        
        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        
        self.logger.info("FeatureAnalyzer initialized with methods: %s", 
                        self._get_available_methods())
    
    def _get_available_methods(self) -> List[str]:
        """Get list of available analysis methods."""
        methods = []
        if SHAP_AVAILABLE:
            methods.append('SHAP')
        if LIME_AVAILABLE:
            methods.append('LIME')
        methods.append('Intrinsic')  # Always available for tree-based models
        return methods
    
    def analyze_model_interpretability(self, model: Union[MLModel, BaseEstimator],
                                     X_train: np.ndarray,
                                     X_test: np.ndarray,
                                     feature_names: List[str],
                                     model_name: str = "Unknown",
                                     sample_explanations: bool = True) -> ModelInterpretabilityReport:
        """
        Perform comprehensive model interpretability analysis.
        
        Args:
            model: Trained model to analyze
            X_train: Training data for background/reference
            X_test: Test data for analysis
            feature_names: Names of features
            model_name: Name of the model
            sample_explanations: Whether to generate sample-level explanations
            
        Returns:
            ModelInterpretabilityReport with comprehensive analysis
        """
        self.logger.info("Starting interpretability analysis for model: %s", model_name)
        
        # Limit sample size for performance
        if len(X_test) > self.max_samples_for_analysis:
            indices = np.random.choice(len(X_test), self.max_samples_for_analysis, replace=False)
            X_test_sample = X_test[indices]
            self.logger.info("Sampled %d instances for analysis", self.max_samples_for_analysis)
        else:
            X_test_sample = X_test
        
        # Determine model type
        model_type = self._determine_model_type(model)
        
        # Perform different types of analysis
        results = []
        available_methods = self._get_available_methods()
        
        # 1. Intrinsic feature importance (for tree-based models)
        if self._supports_intrinsic_importance(model):
            try:
                intrinsic_result = self._analyze_intrinsic_importance(
                    model, feature_names, model_name
                )
                results.append(intrinsic_result)
                self.logger.info("Intrinsic feature importance analysis completed")
            except Exception as e:
                self.logger.warning("Intrinsic analysis failed: %s", str(e))
        
        # 2. SHAP analysis (preferred method)
        if SHAP_AVAILABLE:
            try:
                shap_result = self._analyze_shap_importance(
                    model, X_train, X_test_sample, feature_names, model_name,
                    sample_explanations
                )
                results.append(shap_result)
                self.logger.info("SHAP analysis completed")
            except Exception as e:
                self.logger.warning("SHAP analysis failed: %s", str(e))
        
        # 3. LIME analysis (fallback method)
        if LIME_AVAILABLE and (not SHAP_AVAILABLE or len(results) == 0):
            try:
                lime_result = self._analyze_lime_importance(
                    model, X_train, X_test_sample, feature_names, model_name,
                    sample_explanations
                )
                results.append(lime_result)
                self.logger.info("LIME analysis completed")
            except Exception as e:
                self.logger.warning("LIME analysis failed: %s", str(e))
        
        if not results:
            raise ValueError("No feature analysis methods succeeded")
        
        # Generate top features summary
        top_features = {}
        for result in results:
            sorted_features = sorted(
                result.global_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:self.top_features_count]
            top_features[result.method] = sorted_features
        
        # Create comprehensive report
        report = ModelInterpretabilityReport(
            model_name=model_name,
            model_type=model_type,
            feature_count=len(feature_names),
            sample_count=len(X_test_sample),
            analysis_methods=[r.method for r in results],
            feature_importance_results=results,
            top_features=top_features,
            summary_statistics=self._calculate_summary_statistics(results)
        )
        
        self.logger.info("Model interpretability analysis completed for %s", model_name)
        return report
    
    def _determine_model_type(self, model: Union[MLModel, BaseEstimator]) -> str:
        """Determine the type of model for appropriate analysis method selection."""
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__
            
            if 'RandomForest' in class_name:
                return 'RandomForest'
            elif 'XGBoost' in class_name or 'XGB' in class_name:
                return 'XGBoost'
            elif 'DecisionTree' in class_name:
                return 'DecisionTree'
            elif 'LogisticRegression' in class_name:
                return 'LogisticRegression'
            elif 'SVM' in class_name or 'SVC' in class_name:
                return 'SVM'
            elif 'Neural' in class_name or 'MLP' in class_name:
                return 'NeuralNetwork'
        
        return 'Unknown'
    
    def _supports_intrinsic_importance(self, model: Union[MLModel, BaseEstimator]) -> bool:
        """Check if model supports intrinsic feature importance."""
        return hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
    
    def _analyze_intrinsic_importance(self, model: Union[MLModel, BaseEstimator],
                                    feature_names: List[str],
                                    model_name: str) -> FeatureImportanceResult:
        """Analyze intrinsic feature importance from the model."""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                # Multi-class: use mean absolute coefficients
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not support intrinsic feature importance")
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        global_importance = dict(zip(feature_names, importances))
        
        # Generate visualization
        viz_path = self._create_feature_importance_plot(
            feature_names, importances, f"Intrinsic_{model_name}", "Intrinsic Feature Importance"
        )
        
        return FeatureImportanceResult(
            method="Intrinsic",
            feature_names=feature_names,
            importance_scores=importances.tolist(),
            global_importance=global_importance,
            visualization_path=viz_path,
            metadata={
                'model_type': self._determine_model_type(model),
                'normalization': 'sum_to_one'
            }
        )
    
    def _analyze_shap_importance(self, model: Union[MLModel, BaseEstimator],
                               X_train: np.ndarray,
                               X_test: np.ndarray,
                               feature_names: List[str],
                               model_name: str,
                               sample_explanations: bool = True) -> FeatureImportanceResult:
        """Analyze feature importance using SHAP values."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available")
        
        # Select appropriate explainer based on model type
        model_type = self._determine_model_type(model)
        
        try:
            if model_type in ['RandomForest', 'XGBoost', 'DecisionTree']:
                # Tree-based explainer
                explainer = shap.TreeExplainer(model)
                # Limit test samples to avoid memory issues
                test_sample = X_test[:min(50, len(X_test))]
                shap_values = explainer.shap_values(test_sample)
            else:
                # Use KernelExplainer as fallback (slower but more general)
                # Use a subset of training data as background
                background_size = min(50, len(X_train))
                background = X_train[np.random.choice(len(X_train), background_size, replace=False)]
                
                explainer = shap.KernelExplainer(model.predict_proba, background)
                test_sample = X_test[:min(20, len(X_test))]  # Even smaller for kernel explainer
                shap_values = explainer.shap_values(test_sample)
        
        except Exception as e:
            self.logger.warning("Tree/Kernel explainer failed, trying Permutation explainer: %s", str(e))
            try:
                # Fallback to permutation explainer
                explainer = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                test_sample = X_test[:min(20, len(X_test))]
                shap_values = explainer.shap_values(test_sample)
            except Exception as e2:
                self.logger.error("All SHAP explainers failed: %s", str(e2))
                raise RuntimeError(f"SHAP analysis failed: {str(e2)}")
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: use mean absolute SHAP values across classes
            shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            # Binary classification or regression
            shap_values_combined = np.abs(shap_values)
        
        # Calculate global importance (mean absolute SHAP values)
        global_importance_scores = np.mean(shap_values_combined, axis=0)
        global_importance = dict(zip(feature_names, global_importance_scores))
        
        # Generate sample explanations if requested
        local_explanations = None
        if sample_explanations:
            local_explanations = []
            actual_samples = len(shap_values) if not isinstance(shap_values, list) else len(shap_values[0])
            for i in range(min(5, actual_samples)):  # Explain first 5 samples
                if isinstance(shap_values, list):
                    sample_shap = {f"class_{j}": dict(zip(feature_names, shap_values[j][i])) 
                                 for j in range(len(shap_values))}
                else:
                    sample_shap = dict(zip(feature_names, shap_values[i]))
                
                local_explanations.append({
                    'sample_index': i,
                    'shap_values': sample_shap
                })
        
        # Generate visualizations
        viz_path = self._create_shap_visualizations(
            shap_values, feature_names, model_name, explainer
        )
        
        return FeatureImportanceResult(
            method="SHAP",
            feature_names=feature_names,
            importance_scores=global_importance_scores.tolist(),
            global_importance=global_importance,
            local_explanations=local_explanations,
            visualization_path=viz_path,
            metadata={
                'explainer_type': type(explainer).__name__,
                'samples_analyzed': actual_samples,
                'model_type': model_type
            }
        )
    
    def _analyze_lime_importance(self, model: Union[MLModel, BaseEstimator],
                               X_train: np.ndarray,
                               X_test: np.ndarray,
                               feature_names: List[str],
                               model_name: str,
                               sample_explanations: bool = True) -> FeatureImportanceResult:
        """Analyze feature importance using LIME."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['Normal', 'Malicious'],  # Assuming binary classification
            mode='classification',
            discretize_continuous=True
        )
        
        # Analyze samples
        feature_importance_scores = np.zeros(len(feature_names))
        local_explanations = []
        
        # Analyze subset of test samples
        n_samples = min(50, len(X_test))  # LIME is slow, use fewer samples
        
        for i in range(n_samples):
            try:
                # Get explanation for this sample
                explanation = explainer.explain_instance(
                    X_test[i],
                    model.predict_proba,
                    num_features=len(feature_names)
                )
                
                # Extract feature importance for this sample
                sample_importance = dict(explanation.as_list())
                
                # Accumulate global importance
                for feature_name, importance in sample_importance.items():
                    if feature_name in feature_names:
                        feature_idx = feature_names.index(feature_name)
                        feature_importance_scores[feature_idx] += abs(importance)
                
                # Store local explanation if requested
                if sample_explanations and i < 5:  # Store first 5 explanations
                    local_explanations.append({
                        'sample_index': i,
                        'lime_explanation': sample_importance,
                        'prediction_proba': model.predict_proba([X_test[i]])[0].tolist()
                    })
                    
            except Exception as e:
                self.logger.warning("LIME explanation failed for sample %d: %s", i, str(e))
                continue
        
        # Normalize global importance
        if np.sum(feature_importance_scores) > 0:
            feature_importance_scores = feature_importance_scores / np.sum(feature_importance_scores)
        
        global_importance = dict(zip(feature_names, feature_importance_scores))
        
        # Generate visualization
        viz_path = self._create_feature_importance_plot(
            feature_names, feature_importance_scores, f"LIME_{model_name}", "LIME Feature Importance"
        )
        
        return FeatureImportanceResult(
            method="LIME",
            feature_names=feature_names,
            importance_scores=feature_importance_scores.tolist(),
            global_importance=global_importance,
            local_explanations=local_explanations if sample_explanations else None,
            visualization_path=viz_path,
            metadata={
                'samples_analyzed': n_samples,
                'explainer_type': 'LimeTabularExplainer'
            }
        )
    
    def _create_feature_importance_plot(self, feature_names: List[str],
                                      importance_scores: np.ndarray,
                                      filename_prefix: str,
                                      title: str) -> str:
        """Create feature importance visualization."""
        # Get top features for plotting
        top_indices = np.argsort(np.abs(importance_scores))[-self.top_features_count:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)
        y_pos = np.arange(len(top_features))
        
        bars = ax.barh(y_pos, top_scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        # Color bars based on positive/negative importance
        for bar, score in zip(bars, top_scores):
            if score >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        # Save plot
        viz_path = os.path.join(self.visualization_dir, f"{filename_prefix}_importance.png")
        plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_shap_visualizations(self, shap_values: Union[np.ndarray, List[np.ndarray]],
                                  feature_names: List[str],
                                  model_name: str,
                                  explainer) -> str:
        """Create SHAP-specific visualizations."""
        viz_dir = os.path.join(self.visualization_dir, f"SHAP_{model_name}")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Summary plot
            if isinstance(shap_values, list):
                # Multi-class case
                shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            else:
                # Binary case
                shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            
            summary_path = os.path.join(viz_dir, "summary_plot.png")
            plt.savefig(summary_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            if isinstance(shap_values, list):
                combined_shap = np.concatenate([np.abs(sv) for sv in shap_values], axis=1)
                shap.summary_plot(combined_shap, feature_names=feature_names, 
                                plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, feature_names=feature_names, 
                                plot_type="bar", show=False)
            
            bar_path = os.path.join(viz_dir, "feature_importance_bar.png")
            plt.savefig(bar_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return viz_dir
            
        except Exception as e:
            self.logger.warning("SHAP visualization failed: %s", str(e))
            return viz_dir
    
    def _calculate_summary_statistics(self, results: List[FeatureImportanceResult]) -> Dict[str, Any]:
        """Calculate summary statistics across different analysis methods."""
        if not results:
            return {}
        
        # Feature consistency across methods
        all_features = set()
        method_features = {}
        
        for result in results:
            features = set(result.global_importance.keys())
            all_features.update(features)
            method_features[result.method] = features
        
        # Calculate feature ranking consistency
        feature_rankings = {}
        for result in results:
            sorted_features = sorted(
                result.global_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            feature_rankings[result.method] = [f[0] for f in sorted_features]
        
        # Top feature overlap
        top_feature_overlap = {}
        if len(results) > 1:
            for i, result1 in enumerate(results):
                for j, result2 in enumerate(results[i+1:], i+1):
                    top_features_1 = set(feature_rankings[result1.method][:10])
                    top_features_2 = set(feature_rankings[result2.method][:10])
                    overlap = len(top_features_1.intersection(top_features_2))
                    top_feature_overlap[f"{result1.method}_vs_{result2.method}"] = overlap / 10.0
        
        return {
            'total_features': len(all_features),
            'methods_used': [r.method for r in results],
            'top_feature_overlap': top_feature_overlap,
            'feature_rankings': feature_rankings
        }
    
    def generate_interpretability_report(self, report: ModelInterpretabilityReport,
                                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive interpretability report.
        
        Args:
            report: ModelInterpretabilityReport to generate text report from
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append(f"# Model Interpretability Report: {report.model_name}")
        report_lines.append(f"Model Type: {report.model_type}")
        report_lines.append(f"Features: {report.feature_count}")
        report_lines.append(f"Samples Analyzed: {report.sample_count}")
        report_lines.append(f"Analysis Methods: {', '.join(report.analysis_methods)}")
        report_lines.append("")
        
        # Top features by method
        report_lines.append("## Top Features by Analysis Method")
        for method, features in report.top_features.items():
            report_lines.append(f"### {method}")
            for i, (feature, importance) in enumerate(features[:10], 1):
                report_lines.append(f"{i:2d}. {feature}: {importance:.4f}")
            report_lines.append("")
        
        # Method comparison
        if len(report.analysis_methods) > 1:
            report_lines.append("## Method Comparison")
            if report.summary_statistics and 'top_feature_overlap' in report.summary_statistics:
                for comparison, overlap in report.summary_statistics['top_feature_overlap'].items():
                    report_lines.append(f"- {comparison}: {overlap:.2%} overlap in top 10 features")
            report_lines.append("")
        
        # Feature analysis details
        report_lines.append("## Detailed Analysis Results")
        for result in report.feature_importance_results:
            report_lines.append(f"### {result.method} Analysis")
            if result.metadata:
                for key, value in result.metadata.items():
                    report_lines.append(f"- {key}: {value}")
            
            if result.visualization_path:
                report_lines.append(f"- Visualization: {result.visualization_path}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        recommendations = self._generate_interpretability_recommendations(report)
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(report_text)
                self.logger.info("Interpretability report saved to: %s", save_path)
            except Exception as e:
                self.logger.error("Failed to save report: %s", str(e))
        
        return report_text
    
    def _generate_interpretability_recommendations(self, report: ModelInterpretabilityReport) -> List[str]:
        """Generate recommendations based on interpretability analysis."""
        recommendations = []
        
        # Check for method consistency
        if len(report.analysis_methods) > 1:
            if report.summary_statistics and 'top_feature_overlap' in report.summary_statistics:
                avg_overlap = np.mean(list(report.summary_statistics['top_feature_overlap'].values()))
                if avg_overlap < 0.5:
                    recommendations.append(
                        "Low agreement between analysis methods. Consider investigating feature stability."
                    )
                elif avg_overlap > 0.8:
                    recommendations.append(
                        "High agreement between analysis methods indicates robust feature importance."
                    )
        
        # Check for feature concentration
        for method, features in report.top_features.items():
            if len(features) > 0:
                top_5_importance = sum(abs(imp) for _, imp in features[:5])
                total_importance = sum(abs(imp) for _, imp in features)
                
                if total_importance > 0:
                    concentration = top_5_importance / total_importance
                    if concentration > 0.8:
                        recommendations.append(
                            f"{method}: High feature concentration - top 5 features account for {concentration:.1%} of importance."
                        )
        
        # Model-specific recommendations
        if report.model_type in ['RandomForest', 'XGBoost']:
            recommendations.append(
                "Tree-based model: Consider feature selection based on importance scores to reduce complexity."
            )
        elif report.model_type == 'LogisticRegression':
            recommendations.append(
                "Linear model: Feature coefficients directly indicate feature impact direction."
            )
        elif report.model_type == 'NeuralNetwork':
            recommendations.append(
                "Neural network: SHAP/LIME analysis is crucial for understanding non-linear feature interactions."
            )
        
        return recommendations
    
    def compare_feature_importance_methods(self, results: List[FeatureImportanceResult]) -> Dict[str, Any]:
        """
        Compare feature importance across different analysis methods.
        
        Args:
            results: List of FeatureImportanceResult from different methods
            
        Returns:
            Comparison analysis
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 methods for comparison")
        
        comparison = {
            'methods': [r.method for r in results],
            'feature_correlations': {},
            'ranking_correlations': {},
            'top_feature_stability': {},
            'method_agreements': {}
        }
        
        # Calculate pairwise correlations
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                method1, method2 = result1.method, result2.method
                
                # Get common features
                common_features = set(result1.global_importance.keys()) & set(result2.global_importance.keys())
                
                if len(common_features) > 1:
                    # Feature importance correlation
                    imp1 = [result1.global_importance[f] for f in common_features]
                    imp2 = [result2.global_importance[f] for f in common_features]
                    
                    correlation = np.corrcoef(imp1, imp2)[0, 1]
                    comparison['feature_correlations'][f"{method1}_vs_{method2}"] = correlation
                    
                    # Ranking correlation (Spearman)
                    from scipy.stats import spearmanr
                    rank_corr, _ = spearmanr(imp1, imp2)
                    comparison['ranking_correlations'][f"{method1}_vs_{method2}"] = rank_corr
        
        return comparison