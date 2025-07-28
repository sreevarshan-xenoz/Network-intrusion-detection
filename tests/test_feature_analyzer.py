"""
Unit tests for the FeatureAnalyzer class.
"""
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

from src.services.feature_analyzer import (
    FeatureAnalyzer, FeatureImportanceResult, ModelInterpretabilityReport
)


class TestFeatureAnalyzer:
    """Test cases for FeatureAnalyzer class."""
    
    @pytest.fixture
    def temp_viz_dir(self):
        """Create temporary directory for visualizations."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_analyzer(self, temp_viz_dir):
        """Create FeatureAnalyzer instance with temporary directory."""
        config = {
            'visualization_directory': temp_viz_dir,
            'max_samples': 100,
            'top_features_count': 10
        }
        return FeatureAnalyzer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split into train/test
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_names
    
    @pytest.fixture
    def trained_rf_model(self, sample_data):
        """Create trained Random Forest model."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    @pytest.fixture
    def trained_lr_model(self, sample_data):
        """Create trained Logistic Regression model."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def test_initialization(self, temp_viz_dir):
        """Test FeatureAnalyzer initialization."""
        config = {
            'visualization_directory': temp_viz_dir,
            'max_samples': 500,
            'top_features_count': 15
        }
        analyzer = FeatureAnalyzer(config)
        
        assert analyzer.max_samples_for_analysis == 500
        assert analyzer.top_features_count == 15
        assert analyzer.visualization_dir == temp_viz_dir
        assert os.path.exists(temp_viz_dir)
    
    def test_get_available_methods(self, feature_analyzer):
        """Test getting available analysis methods."""
        methods = feature_analyzer._get_available_methods()
        
        assert isinstance(methods, list)
        assert 'Intrinsic' in methods
        # SHAP and LIME availability depends on installation
    
    def test_determine_model_type(self, feature_analyzer, trained_rf_model, trained_lr_model):
        """Test model type determination."""
        rf_type = feature_analyzer._determine_model_type(trained_rf_model)
        lr_type = feature_analyzer._determine_model_type(trained_lr_model)
        
        assert rf_type == 'RandomForest'
        assert lr_type == 'LogisticRegression'
    
    def test_supports_intrinsic_importance(self, feature_analyzer, trained_rf_model, trained_lr_model):
        """Test checking for intrinsic importance support."""
        assert feature_analyzer._supports_intrinsic_importance(trained_rf_model) == True
        assert feature_analyzer._supports_intrinsic_importance(trained_lr_model) == True
        
        # Mock model without intrinsic importance
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        assert feature_analyzer._supports_intrinsic_importance(mock_model) == False
    
    def test_analyze_intrinsic_importance_rf(self, feature_analyzer, trained_rf_model, sample_data):
        """Test intrinsic importance analysis for Random Forest."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        result = feature_analyzer._analyze_intrinsic_importance(
            trained_rf_model, feature_names, "TestRF"
        )
        
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "Intrinsic"
        assert len(result.feature_names) == len(feature_names)
        assert len(result.importance_scores) == len(feature_names)
        assert len(result.global_importance) == len(feature_names)
        assert result.visualization_path is not None
        assert os.path.exists(result.visualization_path)
        
        # Check that importances sum to 1 (normalized)
        assert abs(sum(result.importance_scores) - 1.0) < 1e-6
    
    def test_analyze_intrinsic_importance_lr(self, feature_analyzer, trained_lr_model, sample_data):
        """Test intrinsic importance analysis for Logistic Regression."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        result = feature_analyzer._analyze_intrinsic_importance(
            trained_lr_model, feature_names, "TestLR"
        )
        
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "Intrinsic"
        assert len(result.importance_scores) == len(feature_names)
        assert result.visualization_path is not None
        
        # For logistic regression, should use absolute coefficients
        assert all(score >= 0 for score in result.importance_scores)
    
    def test_analyze_intrinsic_importance_no_support(self, feature_analyzer):
        """Test intrinsic importance analysis with unsupported model."""
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        with pytest.raises(ValueError, match="does not support intrinsic feature importance"):
            feature_analyzer._analyze_intrinsic_importance(
                mock_model, ['f1', 'f2'], "TestModel"
            )
    
    @patch('src.services.feature_analyzer.SHAP_AVAILABLE', True)
    @patch('shap.TreeExplainer')
    def test_analyze_shap_importance_tree_model(self, mock_tree_explainer, feature_analyzer, 
                                              trained_rf_model, sample_data):
        """Test SHAP analysis for tree-based model."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        # Mock SHAP explainer and values
        mock_explainer_instance = Mock()
        mock_shap_values = np.random.rand(len(X_test), len(feature_names))
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance
        
        with patch('shap.summary_plot'), patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            result = feature_analyzer._analyze_shap_importance(
                trained_rf_model, X_train, X_test, feature_names, "TestRF"
            )
        
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "SHAP"
        assert len(result.importance_scores) == len(feature_names)
        assert result.local_explanations is not None
        assert len(result.local_explanations) <= 5  # Should explain first 5 samples
        
        mock_tree_explainer.assert_called_once_with(trained_rf_model)
        mock_explainer_instance.shap_values.assert_called_once()
    
    @patch('src.services.feature_analyzer.SHAP_AVAILABLE', False)
    def test_analyze_shap_importance_not_available(self, feature_analyzer, trained_rf_model, sample_data):
        """Test SHAP analysis when SHAP is not available."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        with pytest.raises(ImportError, match="SHAP is not available"):
            feature_analyzer._analyze_shap_importance(
                trained_rf_model, X_train, X_test, feature_names, "TestRF"
            )
    
    @patch('src.services.feature_analyzer.LIME_AVAILABLE', True)
    @patch('lime.lime_tabular.LimeTabularExplainer')
    def test_analyze_lime_importance(self, mock_lime_explainer, feature_analyzer, 
                                   trained_rf_model, sample_data):
        """Test LIME analysis."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        # Mock LIME explainer and explanation
        mock_explainer_instance = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [(f'feature_{i}', np.random.rand()) for i in range(5)]
        mock_explainer_instance.explain_instance.return_value = mock_explanation
        mock_lime_explainer.return_value = mock_explainer_instance
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            result = feature_analyzer._analyze_lime_importance(
                trained_rf_model, X_train, X_test[:10], feature_names, "TestRF"
            )
        
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "LIME"
        assert len(result.importance_scores) == len(feature_names)
        assert result.local_explanations is not None
        
        mock_lime_explainer.assert_called_once()
    
    @patch('src.services.feature_analyzer.LIME_AVAILABLE', False)
    def test_analyze_lime_importance_not_available(self, feature_analyzer, trained_rf_model, sample_data):
        """Test LIME analysis when LIME is not available."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        with pytest.raises(ImportError, match="LIME is not available"):
            feature_analyzer._analyze_lime_importance(
                trained_rf_model, X_train, X_test, feature_names, "TestRF"
            )
    
    def test_create_feature_importance_plot(self, feature_analyzer, sample_data):
        """Test feature importance plot creation."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        importance_scores = np.random.rand(len(feature_names))
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            viz_path = feature_analyzer._create_feature_importance_plot(
                feature_names, importance_scores, "test_model", "Test Importance"
            )
        
        assert viz_path is not None
        assert "test_model_importance.png" in viz_path
    
    def test_analyze_model_interpretability_rf(self, feature_analyzer, trained_rf_model, sample_data):
        """Test comprehensive model interpretability analysis for Random Forest."""
        X_train, X_test, y_train, y_test, feature_names = sample_data
        
        # Mock SHAP to avoid fatal exceptions
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('src.services.feature_analyzer.SHAP_AVAILABLE', False), \
             patch('src.services.feature_analyzer.LIME_AVAILABLE', False):
            
            report = feature_analyzer.analyze_model_interpretability(
                trained_rf_model, X_train, X_test, feature_names, "TestRF"
            )
        
        assert isinstance(report, ModelInterpretabilityReport)
        assert report.model_name == "TestRF"
        assert report.model_type == "RandomForest"
        assert report.feature_count == len(feature_names)
        assert len(report.feature_importance_results) >= 1  # At least intrinsic
        assert len(report.analysis_methods) >= 1
        assert "Intrinsic" in report.analysis_methods
        assert "Intrinsic" in report.top_features
    
    def test_analyze_model_interpretability_large_dataset(self, feature_analyzer):
        """Test analysis with large dataset (should be sampled)."""
        # Create large dataset with correct number of features
        X_large = np.random.rand(2000, 20)  # Match the original training data features
        X_train = np.random.rand(1000, 20)
        feature_names = [f'feature_{i}' for i in range(20)]
        
        # Create and train a new model with the correct feature count
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        y_train = np.random.randint(0, 2, 1000)
        model.fit(X_train, y_train)
        
        # Mock SHAP to avoid fatal exceptions
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('src.services.feature_analyzer.SHAP_AVAILABLE', False), \
             patch('src.services.feature_analyzer.LIME_AVAILABLE', False):
            
            report = feature_analyzer.analyze_model_interpretability(
                model, X_train, X_large, feature_names, "TestRF"
            )
        
        # Should sample down to max_samples_for_analysis
        assert report.sample_count <= feature_analyzer.max_samples_for_analysis
    
    def test_calculate_summary_statistics(self, feature_analyzer):
        """Test summary statistics calculation."""
        # Create mock results
        result1 = FeatureImportanceResult(
            method="Method1",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.5, 0.3, 0.2],
            global_importance={'f1': 0.5, 'f2': 0.3, 'f3': 0.2}
        )
        
        result2 = FeatureImportanceResult(
            method="Method2",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.4, 0.4, 0.2],
            global_importance={'f1': 0.4, 'f2': 0.4, 'f3': 0.2}
        )
        
        stats = feature_analyzer._calculate_summary_statistics([result1, result2])
        
        assert 'total_features' in stats
        assert 'methods_used' in stats
        assert 'top_feature_overlap' in stats
        assert 'feature_rankings' in stats
        
        assert stats['total_features'] == 3
        assert stats['methods_used'] == ['Method1', 'Method2']
    
    def test_generate_interpretability_report(self, feature_analyzer, temp_viz_dir):
        """Test interpretability report generation."""
        # Create mock report
        result = FeatureImportanceResult(
            method="Intrinsic",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.5, 0.3, 0.2],
            global_importance={'f1': 0.5, 'f2': 0.3, 'f3': 0.2},
            metadata={'model_type': 'RandomForest'}
        )
        
        report = ModelInterpretabilityReport(
            model_name="TestModel",
            model_type="RandomForest",
            feature_count=3,
            sample_count=100,
            analysis_methods=["Intrinsic"],
            feature_importance_results=[result],
            top_features={"Intrinsic": [('f1', 0.5), ('f2', 0.3), ('f3', 0.2)]}
        )
        
        report_text = feature_analyzer.generate_interpretability_report(report)
        
        assert "TestModel" in report_text
        assert "RandomForest" in report_text
        assert "Intrinsic" in report_text
        assert "f1" in report_text
        
        # Test saving to file
        save_path = os.path.join(temp_viz_dir, "test_report.md")
        report_text = feature_analyzer.generate_interpretability_report(report, save_path)
        
        assert os.path.exists(save_path)
        with open(save_path, 'r') as f:
            saved_content = f.read()
        assert saved_content == report_text
    
    def test_generate_interpretability_recommendations(self, feature_analyzer):
        """Test recommendation generation."""
        # Create mock report with multiple methods
        result1 = FeatureImportanceResult(
            method="Method1",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.8, 0.1, 0.1],
            global_importance={'f1': 0.8, 'f2': 0.1, 'f3': 0.1}
        )
        
        result2 = FeatureImportanceResult(
            method="Method2",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.4, 0.4, 0.2],
            global_importance={'f1': 0.4, 'f2': 0.4, 'f3': 0.2}
        )
        
        report = ModelInterpretabilityReport(
            model_name="TestModel",
            model_type="RandomForest",
            feature_count=3,
            sample_count=100,
            analysis_methods=["Method1", "Method2"],
            feature_importance_results=[result1, result2],
            top_features={
                "Method1": [('f1', 0.8), ('f2', 0.1), ('f3', 0.1)],
                "Method2": [('f1', 0.4), ('f2', 0.4), ('f3', 0.2)]
            },
            summary_statistics={'top_feature_overlap': {'Method1_vs_Method2': 0.3}}
        )
        
        recommendations = feature_analyzer._generate_interpretability_recommendations(report)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations about feature concentration and tree-based model
        rec_text = ' '.join(recommendations)
        assert 'concentration' in rec_text or 'Tree-based' in rec_text
    
    def test_compare_feature_importance_methods(self, feature_analyzer):
        """Test comparison of feature importance methods."""
        result1 = FeatureImportanceResult(
            method="Method1",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.5, 0.3, 0.2],
            global_importance={'f1': 0.5, 'f2': 0.3, 'f3': 0.2}
        )
        
        result2 = FeatureImportanceResult(
            method="Method2",
            feature_names=['f1', 'f2', 'f3'],
            importance_scores=[0.4, 0.4, 0.2],
            global_importance={'f1': 0.4, 'f2': 0.4, 'f3': 0.2}
        )
        
        comparison = feature_analyzer.compare_feature_importance_methods([result1, result2])
        
        assert 'methods' in comparison
        assert 'feature_correlations' in comparison
        assert 'ranking_correlations' in comparison
        
        assert comparison['methods'] == ['Method1', 'Method2']
        assert 'Method1_vs_Method2' in comparison['feature_correlations']
    
    def test_compare_feature_importance_methods_insufficient_methods(self, feature_analyzer):
        """Test comparison with insufficient methods."""
        result1 = FeatureImportanceResult(
            method="Method1",
            feature_names=['f1', 'f2'],
            importance_scores=[0.6, 0.4],
            global_importance={'f1': 0.6, 'f2': 0.4}
        )
        
        with pytest.raises(ValueError, match="Need at least 2 methods"):
            feature_analyzer.compare_feature_importance_methods([result1])
    
    def test_analyze_model_interpretability_no_methods_succeed(self, feature_analyzer):
        """Test when no analysis methods succeed."""
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        X_train = np.random.rand(50, 5)
        X_test = np.random.rand(20, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # Mock all methods to fail
        with patch.object(feature_analyzer, '_supports_intrinsic_importance', return_value=False), \
             patch('src.services.feature_analyzer.SHAP_AVAILABLE', False), \
             patch('src.services.feature_analyzer.LIME_AVAILABLE', False):
            
            with pytest.raises(ValueError, match="No feature analysis methods succeeded"):
                feature_analyzer.analyze_model_interpretability(
                    mock_model, X_train, X_test, feature_names, "FailModel"
                )


if __name__ == '__main__':
    pytest.main([__file__])