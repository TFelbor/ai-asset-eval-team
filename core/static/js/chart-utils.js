/**
 * Chart utilities for the Financial Analysis Dashboard
 * This file contains common functions for chart initialization and event handling
 */

const ChartUtils = {
    /**
     * Debounce function to limit how often a function can be called
     * @param {Function} func - The function to debounce
     * @param {number} wait - The time to wait in milliseconds
     * @returns {Function} - Debounced function
     */
    debounce: function(func, wait) {
        let timeout;
        return function() {
            const context = this;
            const args = arguments;
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                func.apply(context, args);
            }, wait);
        };
    },

    /**
     * Initialize a Plotly chart with standard configuration
     * @param {string} elementId - ID of the chart container element
     * @param {Object} chartData - Chart data object with data and layout properties
     * @param {Function} clickCallback - Optional callback function for click events
     * @returns {Promise} - Promise that resolves when the chart is initialized
     */
    initChart: function(elementId, chartData, clickCallback) {
        return new Promise((resolve, reject) => {
            try {
                // Get the chart container
                const container = document.getElementById(elementId);
                if (!container) {
                    reject(new Error(`Chart container with ID ${elementId} not found`));
                    return;
                }

                // Add loading indicator
                const loadingId = `loading-${Math.random().toString(36).substr(2, 9)}`;
                const loadingDiv = document.createElement('div');
                loadingDiv.id = loadingId;
                loadingDiv.className = 'chart-loading';
                loadingDiv.innerHTML = '<div class="loading-spinner"></div><div class="loading-text">Rendering chart...</div>';

                // Apply styles to ensure it covers the chart area
                loadingDiv.style.position = 'absolute';
                loadingDiv.style.top = '0';
                loadingDiv.style.left = '0';
                loadingDiv.style.width = '100%';
                loadingDiv.style.height = '100%';
                loadingDiv.style.display = 'flex';
                loadingDiv.style.flexDirection = 'column';
                loadingDiv.style.alignItems = 'center';
                loadingDiv.style.justifyContent = 'center';
                loadingDiv.style.backgroundColor = 'rgba(17, 24, 39, 0.7)';
                loadingDiv.style.zIndex = '10';
                loadingDiv.style.borderRadius = 'inherit';

                // Set position relative on container if not already
                const computedStyle = window.getComputedStyle(container);
                if (computedStyle.position === 'static') {
                    container.style.position = 'relative';
                }

                container.appendChild(loadingDiv);

                // Use a small timeout to ensure the DOM has updated before rendering
                // This helps prevent layout shifts during rendering
                setTimeout(() => {
                    // Render the chart with responsive configuration
                    Plotly.newPlot(container, chartData.data, chartData.layout, {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                        modeBarButtonsToRemove: ['lasso2d'],
                        displaylogo: false
                    }).then(() => {
                        // Remove loading indicator with a fade out effect
                        loadingDiv.style.transition = 'opacity 0.3s ease';
                        loadingDiv.style.opacity = '0';

                        setTimeout(() => {
                            if (loadingDiv.parentNode) {
                                loadingDiv.parentNode.removeChild(loadingDiv);
                            }
                        }, 300);

                        // Add resize event handler with debounce
                        // Use a unique name for this resize handler to avoid duplicates
                        const resizeHandlerId = `resize-handler-${elementId}`;
                        if (window[resizeHandlerId]) {
                            window.removeEventListener('resize', window[resizeHandlerId]);
                        }

                        window[resizeHandlerId] = this.debounce(function() {
                            const chartContainer = document.querySelector(`#${elementId}`).closest('.chart-container');
                            if (chartContainer) {
                                Plotly.relayout(elementId, {
                                    'width': chartContainer.clientWidth * 0.95,
                                    'height': chartContainer.clientHeight * 0.85
                                });
                            }
                        }, 250);

                        window.addEventListener('resize', window[resizeHandlerId]);

                        // Add double-click event to reset axes
                        container.on('dblclick', function() {
                            Plotly.relayout(elementId, {
                                'xaxis.autorange': true,
                                'yaxis.autorange': true
                            });
                        });

                        // Add click event to handle chart interactions
                        if (typeof clickCallback === 'function') {
                            container.on('plotly_click', clickCallback);
                        } else {
                            container.on('plotly_click', function(data) {
                                // Get the point that was clicked
                                const point = data.points[0];
                                console.log('Clicked point:', point);
                            });
                        }

                        resolve(container);
                    }).catch(error => {
                        // Remove loading indicator if there's an error
                        if (loadingDiv.parentNode) {
                            loadingDiv.parentNode.removeChild(loadingDiv);
                        }
                        reject(error);
                    });
                }, 50);
            } catch (error) {
                reject(error);
            }
        });
    },

    /**
     * Create a standard layout configuration for charts
     * @param {string} title - Chart title
     * @param {string} xAxisTitle - X-axis title
     * @param {string} yAxisTitle - Y-axis title
     * @param {Object} additionalConfig - Additional configuration options
     * @returns {Object} - Layout configuration object
     */
    createStandardLayout: function(title, xAxisTitle, yAxisTitle, additionalConfig = {}) {
        return {
            title: title,
            xaxis: {
                title: xAxisTitle,
                ...additionalConfig.xaxis
            },
            yaxis: {
                title: yAxisTitle,
                ...additionalConfig.yaxis
            },
            height: 600,
            template: "plotly_dark",
            hovermode: "x unified",
            margin: {l: 50, r: 50, t: 80, b: 50},
            legend: {
                orientation: "h",
                yanchor: "bottom",
                y: 1.02,
                xanchor: "right",
                x: 1
            },
            ...additionalConfig
        };
    }
};

// Make the utilities available globally
window.ChartUtils = ChartUtils;
