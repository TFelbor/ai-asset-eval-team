/**
 * Chart tabs component for the Financial Analysis Dashboard
 * Displays charts in tabs on the same page instead of opening new links
 */

class ChartTabs {
    /**
     * Initialize chart tabs
     * @param {string} containerId - ID of the container element
     * @param {Array} chartLinks - Array of chart link objects
     * @param {string} assetType - Type of asset (stock, crypto, etc.)
     * @param {string} symbol - Symbol or ID of the asset
     */
    constructor(containerId, chartLinks, assetType, symbol) {
        this.container = document.getElementById(containerId);
        this.chartLinks = chartLinks;
        this.assetType = assetType;
        this.symbol = symbol;
        this.activeTab = null;
        this.charts = {};

        if (!this.container) {
            console.error(`Container element with ID ${containerId} not found`);
            return;
        }

        this.render();
    }

    /**
     * Render the chart tabs component
     */
    render() {
        // Create the tabs container
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'chart-tabs';

        // Create the tabs header
        const tabsHeader = document.createElement('div');
        tabsHeader.className = 'chart-tabs-header';

        // Create the tabs content
        const tabsContent = document.createElement('div');
        tabsContent.className = 'chart-tabs-content';

        // Add tabs and content for each chart
        this.chartLinks.forEach((chart, index) => {
            // Create tab button
            const tabButton = document.createElement('button');
            tabButton.className = 'chart-tab-button';
            tabButton.textContent = chart.title;
            tabButton.dataset.chartType = chart.type;

            // Create tab content
            const tabContent = document.createElement('div');
            tabContent.className = 'chart-tab-content';
            tabContent.id = `chart-${this.assetType}-${this.symbol}-${chart.type}`;

            // Add loading indicator
            tabContent.innerHTML = '<div class="loading">Loading chart...</div>';

            // Add event listener to tab button
            tabButton.addEventListener('click', () => {
                this.activateTab(chart.type);
            });

            // Add tab button and content to containers
            tabsHeader.appendChild(tabButton);
            tabsContent.appendChild(tabContent);

            // Set first tab as active by default
            if (index === 0) {
                tabButton.classList.add('active');
                tabContent.classList.add('active');
                this.activeTab = chart.type;
            }
        });

        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'chart-tabs-close';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', () => {
            // Smooth transition when closing
            this.container.style.transition = 'opacity 0.3s ease, transform 0.3s ease, visibility 0.3s';
            this.container.style.opacity = '0';
            this.container.style.transform = 'translate(-50%, -50%) scale(0.95)';

            // Remove active class after transition completes
            setTimeout(() => {
                this.container.classList.remove('active');
                // Reset styles to allow CSS transitions to work next time
                setTimeout(() => {
                    this.container.style.transition = '';
                    this.container.style.opacity = '';
                    this.container.style.transform = '';
                }, 50);
            }, 300);
        });

        // Add minimize/maximize button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'chart-tabs-toggle';
        toggleButton.innerHTML = '&#8722;'; // Minus sign
        toggleButton.addEventListener('click', () => {
            tabsContent.classList.toggle('minimized');
            toggleButton.innerHTML = tabsContent.classList.contains('minimized') ? '&#43;' : '&#8722;'; // Plus or minus sign
        });

        // Add buttons to header
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'chart-tabs-buttons';
        buttonsContainer.appendChild(toggleButton);
        buttonsContainer.appendChild(closeButton);
        tabsHeader.appendChild(buttonsContainer);

        // Assemble the component
        tabsContainer.appendChild(tabsHeader);
        tabsContainer.appendChild(tabsContent);

        // Add to container
        this.container.innerHTML = '';
        this.container.appendChild(tabsContainer);

        // Use a small timeout to ensure smooth animation
        setTimeout(() => {
            this.container.classList.add('active');

            // Load the first chart after the container animation completes
            setTimeout(() => {
                this.loadChart(this.activeTab);
            }, 300);
        }, 50);
    }

    /**
     * Activate a tab
     * @param {string} chartType - Type of chart to activate
     */
    activateTab(chartType) {
        // Check if we're already on this tab
        if (this.activeTab === chartType) {
            return;
        }

        // Check if we're rate limited
        if (this.charts[chartType] === 'rate-limited') {
            const contentElement = document.getElementById(`chart-${this.assetType}-${this.symbol}-${chartType}`);
            if (contentElement) {
                contentElement.innerHTML = `
                    <div class="error-message">
                        <h3>Rate Limit Exceeded</h3>
                        <p>The data provider's rate limit has been reached. Please wait a moment before trying again.</p>
                        <button class="retry-button" onclick="new ChartTabs('${this.container.id}', ${JSON.stringify(this.chartLinks)}, '${this.assetType}', '${this.symbol}').loadChart('${chartType}')">Retry</button>
                    </div>
                `;
            }
            return;
        }

        // Update tab buttons
        const buttons = this.container.querySelectorAll('.chart-tab-button');
        buttons.forEach(button => {
            if (button.dataset.chartType === chartType) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Get the current active content and the new content to activate
        const currentContent = this.container.querySelector('.chart-tab-content.active');
        const newContent = document.getElementById(`chart-${this.assetType}-${this.symbol}-${chartType}`);

        if (!newContent) {
            console.error(`Content element for chart ${chartType} not found`);
            return;
        }

        // Apply a smooth transition
        if (currentContent) {
            // Fade out current content
            currentContent.style.transition = 'opacity 0.2s ease';
            currentContent.style.opacity = '0';

            // After fade out, switch tabs and fade in new content
            setTimeout(() => {
                // Hide current content
                currentContent.classList.remove('active');
                currentContent.style.display = 'none';

                // Prepare new content
                newContent.style.opacity = '0';
                newContent.style.display = 'flex';
                newContent.classList.add('active');

                // Force a reflow to ensure the transition works
                void newContent.offsetWidth;

                // Fade in new content
                newContent.style.transition = 'opacity 0.3s ease';
                newContent.style.opacity = '1';

                // Set active tab
                this.activeTab = chartType;

                // Load chart if not already loaded
                this.loadChart(chartType);
            }, 200);
        } else {
            // No current active content, just show the new one
            newContent.classList.add('active');

            // Set active tab
            this.activeTab = chartType;

            // Load chart if not already loaded
            this.loadChart(chartType);
        }
    }

    /**
     * Load a chart
     * @param {string} chartType - Type of chart to load
     */
    loadChart(chartType) {
        // Check if chart is already loaded
        if (this.charts[chartType]) {
            return;
        }

        // Find the chart link
        const chartLink = this.chartLinks.find(link => link.type === chartType);
        if (!chartLink) {
            console.error(`Chart link for type ${chartType} not found`);
            return;
        }

        // Get the content element
        const contentElement = document.getElementById(`chart-${this.assetType}-${this.symbol}-${chartType}`);
        if (!contentElement) {
            console.error(`Content element for chart ${chartType} not found`);
            return;
        }

        // Create a loading container with a smoother appearance
        const loadingContainer = document.createElement('div');
        loadingContainer.className = 'chart-loading';
        loadingContainer.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading ${chartType} chart...</div>
        `;

        // Clear content and add loading indicator
        contentElement.innerHTML = '';
        contentElement.appendChild(loadingContainer);

        // Fetch the chart data
        // Ensure URL is using the new format
        let fetchUrl = chartLink.url;
        if (fetchUrl.startsWith('/chart/')) {
            // Convert old URL format to new format
            const parts = fetchUrl.split('?');
            const path = parts[0].replace('/chart/', '/analyze/');
            const query = parts.length > 1 ? parts[1] : '';

            // Extract asset type and symbol from the path
            const pathParts = path.split('/');
            if (pathParts.length >= 3) {
                const assetType = pathParts[2]; // stock, crypto, etc.
                const symbol = pathParts[3]; // AAPL, BTC, etc.
                fetchUrl = `/analyze/${assetType}/chart/${symbol}?${query}`;
            }
        }

        console.log('Fetching chart from:', fetchUrl);

        // Add a small delay before fetching to ensure smooth animation
        setTimeout(() => {
            fetch(fetchUrl)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Remove loading indicator with a fade effect
                    loadingContainer.style.transition = 'opacity 0.3s ease';
                    loadingContainer.style.opacity = '0';

                    setTimeout(() => {
                        // Remove loading container
                        if (loadingContainer.parentNode) {
                            loadingContainer.parentNode.removeChild(loadingContainer);
                        }

                        // Create chart container
                        const chartContainer = document.createElement('div');
                        chartContainer.className = 'chart';
                        contentElement.appendChild(chartContainer);

                        // Create insights container
                        const insightsContainer = document.createElement('div');
                        insightsContainer.className = 'chart-insights';

                        // Add insights
                        if (data.insights && data.insights.length > 0) {
                            const insightsList = document.createElement('ul');
                            insightsList.className = 'insights-list';

                            data.insights.forEach(insight => {
                                const insightItem = document.createElement('li');
                                insightItem.textContent = insight;
                                insightsList.appendChild(insightItem);
                            });

                            insightsContainer.appendChild(insightsList);
                        }

                        contentElement.appendChild(insightsContainer);

                        // Render the chart using ChartUtils
                        if (data.chart_json) {
                            // Use ChartUtils to initialize the chart
                            ChartUtils.initChart(chartContainer, data.chart_json, (clickData) => {
                                // Custom click handler for this chart type
                                const point = clickData.points[0];
                                console.log(`Clicked point in ${chartType} chart:`, point);

                                // You can add custom behavior here based on the chart type
                                // For example, show different tooltips for different chart types
                            }).then(() => {
                                console.log(`Chart ${chartType} initialized successfully`);
                                // Mark chart as loaded
                                this.charts[chartType] = true;
                            }).catch(error => {
                                console.error(`Error initializing ${chartType} chart:`, error);
                                contentElement.innerHTML = `<div class="error-message">Error initializing chart: ${error.message}</div>`;
                            });
                        } else {
                            contentElement.innerHTML = '<div class="error-message">Error loading chart data</div>';
                        }
                    }, 300);
                })
                .catch(error => {
                    console.error(`Error loading chart ${chartType}:`, error);

                    // Remove loading indicator
                    if (loadingContainer.parentNode) {
                        loadingContainer.parentNode.removeChild(loadingContainer);
                    }

                    // Check if it's a rate limit error (HTTP 429)
                    if (error.message && error.message.includes('429')) {
                        // Display a more user-friendly message for rate limiting
                        contentElement.innerHTML = `
                            <div class="error-message">
                                <h3>Rate Limit Exceeded</h3>
                                <p>The data provider's rate limit has been reached. This typically happens when making too many requests in a short period.</p>
                                <p>Please try again in a minute or try a different chart type.</p>
                                <button class="retry-button" onclick="new ChartTabs('${this.container.id}', ${JSON.stringify(this.chartLinks)}, '${this.assetType}', '${this.symbol}').loadChart('${chartType}')">Retry</button>
                            </div>
                        `;

                        // Cache a placeholder to prevent immediate retries
                        this.charts[chartType] = 'rate-limited';

                        // Set a timeout to allow retry after 60 seconds
                        setTimeout(() => {
                            this.charts[chartType] = null;
                        }, 60000);
                    } else {
                        // For other errors, display a generic error message
                        contentElement.innerHTML = `<div class="error-message">Error loading chart: ${error.message}</div>`;
                    }
                });
        }, 50);
    }
}

// Make the class available globally
window.ChartTabs = ChartTabs;
