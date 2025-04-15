// Financial Analysis Dashboard JavaScript

// Check if Plotly is loaded
function checkPlotlyLoaded() {
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded! Charts will not display correctly.');
        // Try to reload Plotly
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
        script.async = true;
        document.head.appendChild(script);
        return false;
    }
    console.log('Plotly is loaded successfully, version:', Plotly.version);
    return true;
}

document.addEventListener('DOMContentLoaded', function() {
    // Check if Plotly is loaded
    checkPlotlyLoaded();

    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const contentId = tab.getAttribute('data-tab');
            document.getElementById(contentId).classList.add('active');
        });
    });

    // Form submission handlers
    const stockForm = document.getElementById('stock-form');
    if (stockForm) {
        stockForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const ticker = document.getElementById('stock-ticker').value.trim().toUpperCase();
            if (!ticker) return;

            try {
                showLoading('stock-results');
                const response = await fetch(`/analyze/stock/${ticker}`);
                const data = await response.json();
                displayStockResults(data, ticker);
            } catch (error) {
                showError('stock-results', `Error analyzing stock ${ticker}: ${error.message}`);
            }
        });
    }

    const cryptoForm = document.getElementById('crypto-form');
    if (cryptoForm) {
        cryptoForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const coinId = document.getElementById('crypto-id').value.trim().toLowerCase();
            if (!coinId) return;

            try {
                showLoading('crypto-results');
                const response = await fetch(`/analyze/crypto/${coinId}`);
                const data = await response.json();
                displayCryptoResults(data, coinId);
            } catch (error) {
                showError('crypto-results', `Error analyzing cryptocurrency ${coinId}: ${error.message}`);
            }
        });
    }

    const reitForm = document.getElementById('reit-form');
    if (reitForm) {
        reitForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const ticker = document.getElementById('reit-ticker').value.trim().toUpperCase();
            if (!ticker) return;

            try {
                showLoading('reit-results');
                const response = await fetch(`/analyze/reit/${ticker}`);
                const data = await response.json();
                displayReitResults(data, ticker);
            } catch (error) {
                showError('reit-results', `Error analyzing REIT ${ticker}: ${error.message}`);
            }
        });
    }

    const etfForm = document.getElementById('etf-form');
    if (etfForm) {
        etfForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const ticker = document.getElementById('etf-ticker').value.trim().toUpperCase();
            if (!ticker) return;

            try {
                showLoading('etf-results');
                const response = await fetch(`/analyze/etf/${ticker}`);
                const data = await response.json();
                displayEtfResults(data, ticker);
            } catch (error) {
                showError('etf-results', `Error analyzing ETF ${ticker}: ${error.message}`);
            }
        });
    }

    const compareForm = document.getElementById('compare-form');
    if (compareForm) {
        compareForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const stock = document.getElementById('compare-stock').value.trim().toUpperCase();
            const crypto = document.getElementById('compare-crypto').value.trim().toLowerCase();
            const reit = document.getElementById('compare-reit').value.trim().toUpperCase();
            const etf = document.getElementById('compare-etf').value.trim().toUpperCase();

            if (!stock && !crypto && !reit && !etf) {
                showError('compare-results', 'Please enter at least one security to compare');
                return;
            }

            try {
                showLoading('compare-results');

                // Build the URL for the comparison view
                let params = [];
                if (stock) params.push(`stock=${stock}`);
                if (crypto) params.push(`crypto=${crypto}`);
                if (reit) params.push(`reit=${reit}`);
                if (etf) params.push(`etf=${etf}`);

                let url = '/compare/view?' + params.join('&');

                // Redirect to the comparison view page
                window.location.href = url;
            } catch (error) {
                showError('compare-results', `Error comparing securities: ${error.message}`);
            }
        });
    }

    // Initialize chart buttons
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('chart-button') ||
            event.target.closest('.chart-button')) {

            const button = event.target.classList.contains('chart-button') ?
                event.target : event.target.closest('.chart-button');

            const assetType = button.dataset.assetType;
            const symbol = button.dataset.symbol;
            const chartType = button.dataset.chartType;
            const url = button.dataset.url;

            // Get all chart links for this asset
            const chartButtons = document.querySelectorAll(`.chart-button[data-asset-type="${assetType}"][data-symbol="${symbol}"]`);
            const chartLinks = Array.from(chartButtons).map(btn => {
                return {
                    type: btn.dataset.chartType,
                    url: btn.dataset.url,
                    title: btn.textContent
                };
            });

            // Initialize chart tabs
            new ChartTabs('chart-tabs-container', chartLinks, assetType, symbol);

            // Prevent default link behavior
            event.preventDefault();
        }
    });
});

// Helper functions
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading">Loading analysis...</div>';
    }
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="error-message">${message}</div>`;
    }
}

function displayStockResults(data, ticker) {
    const container = document.getElementById('stock-results');
    if (!container) return;

    // Create a unique ID for the results container
    const resultsId = `stock-analysis-${ticker}`;

    console.log('Stock analysis data:', data); // Debug log

    const report = data.report;
    const insights = data.insights;
    const charts = data.charts || [];

    let html = `
        <div class="card" id="${resultsId}">
            <div class="card-header">
                <h2>${ticker} - ${report.stock.name || ticker}</h2>
                <span class="recommendation ${getRecommendationClass(report.recommendation)}">${report.recommendation}</span>
            </div>

            <!-- Removed real-time price container -->

            <div class="chart-container">
                <div id="financial-health-chart" class="chart"></div>
            </div>

            <div class="chart-container">
                <div id="pe-ratio-chart" class="chart"></div>
            </div>

            <div class="chart-container">
                <div id="score-chart" class="chart"></div>
            </div>

            ${charts.length > 0 ? `
            <div class="chart-links">
                <h3>Interactive Charts</h3>
                <div class="button-group">
                    ${charts.map(chart => `<button class="button chart-button" data-asset-type="stock" data-symbol="${ticker}" data-chart-type="${chart.type}" data-url="${chart.url}">${chart.title}</button>`).join('')}
                </div>
            </div>` : ''}

            <h3>Key Insights</h3>
            <ul class="insights-list">
                ${insights.map(insight => `<li>${insight}</li>`).join('')}
            </ul>

            <div id="social-share-${ticker}" class="social-sharing-container"></div>
        </div>
    `;

    container.innerHTML = html;

    // Initialize social sharing
    if (window.socialSharing) {
        window.socialSharing.createSharingWidget(`social-share-${ticker}`, {
            url: window.location.href,
            title: `${ticker} Stock Analysis: ${report.recommendation}`,
            description: `Check out my ${ticker} stock analysis with a recommendation of ${report.recommendation}. Current price: ${report.stock.current_price}, Target: ${report.stock.target_price || 'N/A'}`,
            platforms: ['twitter', 'facebook', 'linkedin', 'email'],
            showCopy: true,
            showDownload: true,
            downloadElementId: resultsId,
            downloadFilename: `${ticker}-stock-analysis.png`
        });
    }

    // Real-time price tracking removed

    // Make sure Plotly is properly initialized
    if (checkPlotlyLoaded()) {
        try {
            // Create financial health dashboard
            // We'll use a radar chart to show multiple financial health metrics
            const financialMetrics = {
                'P/E Ratio': report.stock.pe || 0,
                'P/B Ratio': report.stock.pb || 0,
                'Debt/Equity': report.stock.debt_to_equity || 0,
                'Current Ratio': report.stock.current_ratio || 0,
                'Profit Margin': report.stock.profit_margin || 0,
                'ROE': report.stock.roe || 0
            };

            const financialHealthData = [{
                type: 'scatterpolar',
                r: Object.values(financialMetrics),
                theta: Object.keys(financialMetrics),
                fill: 'toself',
                name: ticker
            }];

            Plotly.newPlot('financial-health-chart', financialHealthData, {
                title: `${ticker} Financial Health Dashboard`,
                polar: {
                    radialaxis: {
                        visible: true,
                        range: [0, Math.max(...Object.values(financialMetrics)) * 1.2]
                    }
                },
                margin: {l: 50, r: 50, t: 80, b: 50},
                autosize: true,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#f9fafb'}
            });

            // Create PE ratio chart
            const peRatioData = [{
                x: ['Stock P/E', 'Sector Avg P/E'],
                y: [report.stock.pe, report.stock.sector_pe],
                type: 'bar'
            }];

            Plotly.newPlot('pe-ratio-chart', peRatioData, {
                title: `${ticker} P/E Ratio Comparison`,
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            // Create score gauge chart
            const scoreData = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: report.overall_score,
                title: {text: `Overall Score: ${report.recommendation}`},
                gauge: {
                    axis: {range: [0, 100]},
                    bar: {color: 'darkblue'},
                    steps: [
                        {range: [0, 30], color: 'red'},
                        {range: [30, 45], color: 'orange'},
                        {range: [45, 65], color: 'yellow'},
                        {range: [65, 80], color: 'lightgreen'},
                        {range: [80, 100], color: 'green'}
                    ]
                }
            }];

            Plotly.newPlot('score-chart', scoreData, {
                margin: {l: 30, r: 30, t: 50, b: 30}
            });

            console.log('Charts created successfully');
        } catch (error) {
            console.error('Error creating charts:', error);
        }
    }

    // Real-time data subscription removed
}

// Real-time price update function removed

function displayCryptoResults(data, coinId) {
    const container = document.getElementById('crypto-results');
    if (!container) return;

    // Create a unique ID for the results container
    const resultsId = `crypto-analysis-${coinId}`;

    console.log('Crypto analysis data:', data); // Debug log

    const report = data.report;
    const insights = data.insights;

    // Ensure we have proper display names
    const cryptoName = report.crypto.name || report.crypto.coin || coinId;
    const cryptoSymbol = report.crypto.symbol || coinId.toUpperCase();

    let html = `
        <div class="card" id="${resultsId}">
            <div class="card-header">
                <h2>${cryptoName} (${cryptoSymbol})</h2>
                <span class="recommendation ${getRecommendationClass(report.recommendation)}">${report.recommendation}</span>
            </div>

            <!-- Removed real-time price container -->
            <div class="chart-container" id="price-changes-chart"></div>
            <div class="chart-container" id="market-metrics-chart"></div>
            <div class="chart-container" id="crypto-score-chart"></div>

            ${data.charts && data.charts.length > 0 ? `
            <div class="chart-links">
                <h3>Interactive Charts</h3>
                <div class="button-group">
                    ${data.charts.map(chart => `<button class="button chart-button" data-asset-type="crypto" data-symbol="${coinId}" data-chart-type="${chart.type}" data-url="${chart.url}">${chart.title}</button>`).join('')}
                </div>
            </div>` : ''}
            <h3>Key Insights</h3>
            <ul class="insights-list">
                ${insights.map(insight => `<li>${insight}</li>`).join('')}
            </ul>

            <div id="social-share-${coinId}" class="social-sharing-container"></div>
        </div>
    `;

    container.innerHTML = html;

    // Initialize social sharing
    if (window.socialSharing) {
        window.socialSharing.createSharingWidget(`social-share-${coinId}`, {
            url: window.location.href,
            title: `${cryptoName} (${cryptoSymbol}) Crypto Analysis: ${report.recommendation}`,
            description: `Check out my ${cryptoName} cryptocurrency analysis with a recommendation of ${report.recommendation}. Current price: ${report.crypto.current_price || 'N/A'}, Market Cap: ${report.crypto.mcap || 'N/A'}`,
            platforms: ['twitter', 'facebook', 'linkedin', 'email'],
            showCopy: true,
            showDownload: true,
            downloadElementId: resultsId,
            downloadFilename: `${coinId}-crypto-analysis.png`
        });
    }

    // Real-time price tracking removed

    // Make sure Plotly is properly initialized
    if (checkPlotlyLoaded()) {
        try {
            // Create price changes chart
            const priceChangesData = [{
                x: ['Price Change 24h', 'Price Change 7d', 'Price Change 30d'],
                y: [
                    report.crypto.price_change_24h,
                    report.crypto.price_change_7d,
                    report.crypto.price_change_30d
                ],
                type: 'bar'
            }];

            Plotly.newPlot('price-changes-chart', priceChangesData, {
                title: `${cryptoSymbol} Price Changes`,
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            // Create market metrics chart
            const marketMetricsData = [{
                x: ['Market Cap Rank', 'Market Dominance', 'Volatility'],
                y: [
                    report.crypto.market_cap_rank,
                    report.crypto.market_dominance,
                    report.crypto.volatility_value
                ],
                type: 'bar'
            }];

            Plotly.newPlot('market-metrics-chart', marketMetricsData, {
                title: `${cryptoSymbol} Market Metrics`,
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            // Create score gauge chart
            const scoreData = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: report.overall_score,
                title: {text: `Overall Score: ${report.recommendation}`},
                gauge: {
                    axis: {range: [0, 100]},
                    bar: {color: 'darkblue'},
                    steps: [
                        {range: [0, 30], color: 'red'},
                        {range: [30, 45], color: 'orange'},
                        {range: [45, 65], color: 'yellow'},
                        {range: [65, 80], color: 'lightgreen'},
                        {range: [80, 100], color: 'green'}
                    ]
                }
            }];

            Plotly.newPlot('crypto-score-chart', scoreData, {
                margin: {l: 30, r: 30, t: 50, b: 30}
            });

            console.log('Crypto charts created successfully');
        } catch (error) {
            console.error('Error creating crypto charts:', error);
        }
    }

    // Real-time data subscription removed
}

function displayReitResults(data, ticker) {
    const container = document.getElementById('reit-results');
    if (!container) return;

    console.log('REIT analysis data:', data); // Debug log

    const report = data.report;
    const insights = data.insights;

    let html = `
        <div class="card">
            <div class="card-header">
                <h2>${ticker} - ${report.reit.name || 'REIT Analysis'}</h2>
                <span class="recommendation ${getRecommendationClass(report.recommendation)}">${report.recommendation}</span>
            </div>
            <div class="chart-container" id="reit-metrics-chart"></div>
            <div class="chart-container" id="reit-score-chart"></div>
            <h3>Key Insights</h3>
            <ul class="insights-list">
                ${insights.map(insight => `<li>${insight}</li>`).join('')}
            </ul>
        </div>
    `;

    container.innerHTML = html;

    // Make sure Plotly is properly initialized
    if (checkPlotlyLoaded()) {
        try {
            // Create REIT metrics chart
            const metricsData = [{
                x: ['Dividend Yield', 'Price to FFO', 'Debt to Equity'],
                y: [
                    report.reit.dividend_yield_value,
                    report.reit.price_to_ffo,
                    report.reit.debt_to_equity
                ],
                type: 'bar'
            }];

            Plotly.newPlot('reit-metrics-chart', metricsData, {
                title: `${ticker} REIT Metrics`,
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            // Create score gauge chart
            const scoreData = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: report.overall_score,
                title: {text: `Overall Score: ${report.recommendation}`},
                gauge: {
                    axis: {range: [0, 100]},
                    bar: {color: 'darkblue'},
                    steps: [
                        {range: [0, 30], color: 'red'},
                        {range: [30, 45], color: 'orange'},
                        {range: [45, 65], color: 'yellow'},
                        {range: [65, 80], color: 'lightgreen'},
                        {range: [80, 100], color: 'green'}
                    ]
                }
            }];

            Plotly.newPlot('reit-score-chart', scoreData, {
                margin: {l: 30, r: 30, t: 50, b: 30}
            });

            console.log('REIT charts created successfully');
        } catch (error) {
            console.error('Error creating REIT charts:', error);
        }
    }
}

function displayEtfResults(data, ticker) {
    const container = document.getElementById('etf-results');
    if (!container) return;

    console.log('ETF analysis data:', data); // Debug log

    const report = data.report;
    const insights = data.insights;

    let html = `
        <div class="card">
            <div class="card-header">
                <h2>${ticker} - ${report.etf.name || 'ETF Analysis'}</h2>
                <span class="recommendation ${getRecommendationClass(report.recommendation)}">${report.recommendation}</span>
            </div>
            <div class="chart-container" id="etf-metrics-chart"></div>
            <div class="chart-container" id="etf-score-chart"></div>
            <h3>Key Insights</h3>
            <ul class="insights-list">
                ${insights.map(insight => `<li>${insight}</li>`).join('')}
            </ul>
        </div>
    `;

    container.innerHTML = html;

    // Make sure Plotly is properly initialized
    if (checkPlotlyLoaded()) {
        try {
            // Create ETF metrics chart
            const metricsData = [{
                x: ['Expense Ratio', 'Yield', 'YTD Return', '3-Year Return'],
                y: [
                    report.etf.expense_ratio_value * 100,  // Convert to percentage
                    report.etf.yield_value,
                    report.etf.ytd_return_value,
                    report.etf.three_year_return_value
                ],
                type: 'bar'
            }];

            Plotly.newPlot('etf-metrics-chart', metricsData, {
                title: `${ticker} ETF Metrics`,
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            // Create score gauge chart
            const scoreData = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: report.overall_score,
                title: {text: `Overall Score: ${report.recommendation}`},
                gauge: {
                    axis: {range: [0, 100]},
                    bar: {color: 'darkblue'},
                    steps: [
                        {range: [0, 30], color: 'red'},
                        {range: [30, 45], color: 'orange'},
                        {range: [45, 65], color: 'yellow'},
                        {range: [65, 80], color: 'lightgreen'},
                        {range: [80, 100], color: 'green'}
                    ]
                }
            }];

            Plotly.newPlot('etf-score-chart', scoreData, {
                margin: {l: 30, r: 30, t: 50, b: 30}
            });

            console.log('ETF charts created successfully');
        } catch (error) {
            console.error('Error creating ETF charts:', error);
        }
    }
}

function displayCompareResults(data) {
    const container = document.getElementById('compare-results');
    if (!container) return;

    console.log('Compare analysis data:', data); // Debug log

    const reports = data.reports;
    const summary = data.summary;

    let html = `
        <div class="card">
            <div class="card-header">
                <h2>Securities Comparison</h2>
            </div>
            <div class="chart-container" id="compare-chart"></div>
            <h3>Summary</h3>
            <p>${summary}</p>
            <h3>Recommendations</h3>
            <ul class="insights-list">
    `;

    // Add recommendations for each security
    if (reports.stock) {
        html += `<li>Stock: ${reports.stock.recommendation} (Score: ${reports.stock.overall_score})</li>`;
    }
    if (reports.crypto) {
        html += `<li>Crypto: ${reports.crypto.recommendation} (Score: ${reports.crypto.overall_score})</li>`;
    }
    if (reports.reit) {
        html += `<li>REIT: ${reports.reit.recommendation} (Score: ${reports.reit.overall_score})</li>`;
    }
    if (reports.etf) {
        html += `<li>ETF: ${reports.etf.recommendation} (Score: ${reports.etf.overall_score})</li>`;
    }

    html += `
            </ul>
        </div>
    `;

    container.innerHTML = html;

    // Make sure Plotly is properly initialized
    if (checkPlotlyLoaded()) {
        try {
            // Create comparison chart
            const names = [];
            const scores = [];

            if (reports.stock) {
                names.push(`Stock: ${reports.stock.stock.ticker}`);
                scores.push(reports.stock.overall_score);
            }
            if (reports.crypto) {
                names.push(`Crypto: ${reports.crypto.crypto.symbol}`);
                scores.push(reports.crypto.overall_score);
            }
            if (reports.reit) {
                names.push(`REIT: ${reports.reit.reit.ticker}`);
                scores.push(reports.reit.overall_score);
            }
            if (reports.etf) {
                names.push(`ETF: ${reports.etf.etf.ticker}`);
                scores.push(reports.etf.overall_score);
            }

            const compareData = [{
                x: names,
                y: scores,
                type: 'bar',
                marker: {
                    color: scores,
                    colorscale: 'RdYlGn'
                }
            }];

            Plotly.newPlot('compare-chart', compareData, {
                title: 'Security Comparison',
                margin: {l: 50, r: 50, t: 80, b: 50}
            });

            console.log('Comparison chart created successfully');
        } catch (error) {
            console.error('Error creating comparison chart:', error);
        }
    }
}

function getRecommendationClass(recommendation) {
    switch (recommendation) {
        case 'Strong Buy': return 'strong-buy';
        case 'Buy': return 'buy';
        case 'Hold': return 'hold';
        case 'Sell': return 'sell';
        case 'Strong Sell': return 'strong-sell';
        default: return '';
    }
}
