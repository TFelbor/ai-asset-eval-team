"""
Tests for the chart service.
"""
import pytest
import json
from unittest.mock import patch

from app.services.chart_service import ChartService


class TestChartService:
    """Tests for the ChartService class."""

    def test_create_stock_price_vs_dcf_chart(self):
        """Test create_stock_price_vs_dcf_chart method."""
        chart_json = ChartService.create_stock_price_vs_dcf_chart('AAPL', 150.0, 180.0)
        
        # Check that the chart JSON has the expected structure
        assert 'data' in chart_json
        assert 'layout' in chart_json
        
        # Check that there are two traces (current price and DCF value)
        assert len(chart_json['data']) == 2
        
        # Check that the first trace is for current price
        assert chart_json['data'][0]['name'] == 'Current Price'
        assert chart_json['data'][0]['y'][0] == 150.0
        
        # Check that the second trace is for DCF value
        assert chart_json['data'][1]['name'] == 'DCF Value'
        assert chart_json['data'][1]['y'][0] == 180.0
        
        # Check that the layout has the expected title
        assert 'AAPL' in chart_json['layout']['title']

    def test_create_stock_metrics_chart(self):
        """Test create_stock_metrics_chart method."""
        metrics = {
            'P/E Ratio': 0.7,
            'P/B Ratio': 0.8,
            'Dividend Yield': 0.5,
            'Beta': 0.9,
            'Confidence': 0.75
        }
        
        chart_json = ChartService.create_stock_metrics_chart('AAPL', metrics)
        
        # Check that the chart JSON has the expected structure
        assert 'data' in chart_json
        assert 'layout' in chart_json
        
        # Check that there is one trace
        assert len(chart_json['data']) == 1
        
        # Check that the trace is a scatterpolar
        assert chart_json['data'][0]['type'] == 'scatterpolar'
        
        # Check that the values match the input metrics
        assert chart_json['data'][0]['r'] == list(metrics.values())
        assert chart_json['data'][0]['theta'] == list(metrics.keys())
        
        # Check that the layout has the expected title
        assert 'AAPL' in chart_json['layout']['title']

    def test_create_score_gauge_chart(self):
        """Test create_score_gauge_chart method."""
        chart_json = ChartService.create_score_gauge_chart(75.0, 'Buy')
        
        # Check that the chart JSON has the expected structure
        assert 'data' in chart_json
        assert 'layout' in chart_json
        
        # Check that there is one trace
        assert len(chart_json['data']) == 1
        
        # Check that the trace is an indicator
        assert chart_json['data'][0]['mode'] == 'gauge+number'
        
        # Check that the value matches the input score
        assert chart_json['data'][0]['value'] == 75.0
        
        # Check that the title contains the recommendation
        assert 'Buy' in chart_json['data'][0]['title']['text']

    def test_create_crypto_price_changes_chart(self):
        """Test create_crypto_price_changes_chart method."""
        changes = {
            '24h': 5.0,
            '7d': 10.0,
            '30d': -2.0,
            '1y': 150.0
        }
        
        chart_json = ChartService.create_crypto_price_changes_chart('BTC', changes)
        
        # Check that the chart JSON has the expected structure
        assert 'data' in chart_json
        assert 'layout' in chart_json
        
        # Check that there is one trace
        assert len(chart_json['data']) == 1
        
        # Check that the trace is a bar chart
        assert chart_json['data'][0]['type'] == 'bar'
        
        # Check that the x values match the input keys
        assert chart_json['data'][0]['x'] == list(changes.keys())
        
        # Check that the y values match the input values
        assert chart_json['data'][0]['y'] == list(changes.values())
        
        # Check that the layout has the expected title
        assert 'BTC' in chart_json['layout']['title']

    def test_create_comparison_chart(self):
        """Test create_comparison_chart method."""
        names = ['AAPL', 'MSFT', 'GOOG']
        scores = [80.0, 85.0, 75.0]
        
        chart_json = ChartService.create_comparison_chart(names, scores)
        
        # Check that the chart JSON has the expected structure
        assert 'data' in chart_json
        assert 'layout' in chart_json
        
        # Check that there is one trace
        assert len(chart_json['data']) == 1
        
        # Check that the trace is a bar chart
        assert chart_json['data'][0]['type'] == 'bar'
        
        # Check that the x values match the input names
        assert chart_json['data'][0]['x'] == names
        
        # Check that the y values match the input scores
        assert chart_json['data'][0]['y'] == scores
        
        # Check that the layout has the expected title
        assert 'Comparison' in chart_json['layout']['title']
