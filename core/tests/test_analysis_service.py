"""
Tests for the analysis service.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.services.analysis_service import AnalysisService


@pytest.fixture
def mock_stock_team():
    """Mock StockAnalysisTeam."""
    with patch('teams.StockAnalysisTeam') as mock:
        instance = mock.return_value
        instance.analyze.return_value = {
            'stock': {
                'name': 'Test Stock',
                'ticker': 'TEST',
                'sector': 'Technology',
                'industry': 'Software',
                'current_price': 100.0,
                'upside_potential': '10.0%',
                'pe': 20.0,
                'sector_pe': 25.0,
                'pb': 5.0,
                'market_cap': '$1B',
                'dividend_yield': '2.0%',
                'beta': 1.2
            },
            'macro': {
                'sentiment': 70,
                'inflation_risk': 'Medium'
            },
            'recommendation': 'Buy',
            'overall_score': 80.0
        }
        yield instance


@pytest.fixture
def mock_crypto_team():
    """Mock CryptoAnalysisTeam."""
    with patch('teams.CryptoAnalysisTeam') as mock:
        instance = mock.return_value
        instance.analyze.return_value = {
            'crypto': {
                'name': 'Test Coin',
                'symbol': 'TCOIN',
                'current_price': 1000.0,
                'mcap': '$10B',
                'market_cap_rank': 5,
                'market_dominance': 2.5,
                'price_change_24h': '5.0%',
                'price_change_7d': '10.0%',
                'volatility': 'High',
                'volume_24h': '$1B',
                'all_time_high': 2000.0,
                'all_time_high_change': -50.0,
                'circulating_supply': 10000000,
                'max_supply': 21000000,
                'supply_percentage': 47.6
            },
            'macro': {
                'gdp_outlook': 'Positive',
                'inflation_risk': 'Medium'
            },
            'recommendation': 'Buy',
            'overall_score': 75.0
        }
        yield instance


class TestAnalysisService:
    """Tests for the AnalysisService class."""

    def test_analyze_stock(self, mock_stock_team):
        """Test analyze_stock method."""
        result = AnalysisService.analyze_stock('TEST')

        # Check that the team was used correctly
        mock_stock_team.analyze.assert_called_once_with('TEST')

        # Check that the result has the expected structure
        assert 'report' in result
        assert 'insights' in result
        assert 'charts' in result
        assert 'news' in result
        assert 'advanced' in result

        # Check that the insights were generated
        assert len(result['insights']) > 0

        # Check that the chart links were created
        assert len(result['charts']) == 4
        assert result['charts'][0]['type'] == 'price'
        assert '/analyze/stock/chart/TEST' in result['charts'][0]['url']

    def test_analyze_crypto(self, mock_crypto_team):
        """Test analyze_crypto method."""
        result = AnalysisService.analyze_crypto('tcoin')

        # Check that the team was used correctly
        mock_crypto_team.analyze.assert_called_once_with('tcoin')

        # Check that the result has the expected structure
        assert 'report' in result
        assert 'insights' in result
        assert 'charts' in result
        assert 'news' in result
        assert 'advanced' in result

        # Check that the insights were generated
        assert len(result['insights']) > 0

        # Check that the chart links were created
        assert len(result['charts']) == 3
        assert result['charts'][0]['type'] == 'price'
        assert '/analyze/crypto/chart/tcoin' in result['charts'][0]['url']

    def test_generate_stock_insights(self, mock_stock_team):
        """Test _generate_stock_insights method."""
        report = mock_stock_team.analyze.return_value
        insights = AnalysisService._generate_stock_insights(report)

        # Check that insights were generated
        assert len(insights) > 0

        # Check that insights contain key information
        assert any('TEST' in insight for insight in insights)
        assert any('Technology' in insight for insight in insights)
        assert any('$100.0' in insight or '$100' in insight for insight in insights)

    def test_generate_crypto_insights(self, mock_crypto_team):
        """Test _generate_crypto_insights method."""
        report = mock_crypto_team.analyze.return_value
        insights = AnalysisService._generate_crypto_insights(report)

        # Check that insights were generated
        assert len(insights) > 0

        # Check that insights contain key information
        assert any('$1,000' in insight for insight in insights)
        assert any('5.0%' in insight for insight in insights)
        assert any('$10B' in insight for insight in insights)
