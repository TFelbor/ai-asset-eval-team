/**
 * Simulated real-time data handling
 */

class RealtimeDataManager {
    constructor() {
        this.isConnected = false;
        this.subscriptions = {};
        this.callbacks = {};
        this.simulateConnection();
    }

    /**
     * Simulate a connection to a real-time data source
     */
    simulateConnection() {
        console.log('Simulating real-time data connection...');

        // Simulate connection after a short delay
        setTimeout(() => {
            this.isConnected = true;
            this.updateConnectionStatus();
            console.log('Simulated connection established');

            // Start periodic updates
            this.startPeriodicUpdates();

            // Dispatch connection event
            document.dispatchEvent(new CustomEvent('realtime:connected'));
        }, 1500);
    }

    /**
     * Start sending periodic price updates
     */
    startPeriodicUpdates() {
        // Update prices every 10 seconds
        setInterval(() => {
            Object.keys(this.subscriptions).forEach(key => {
                const [type, symbol] = key.split(':');
                this.simulatePriceUpdate(type, symbol);
            });
        }, 10000);
    }

    /**
     * Generate a simulated price update
     */
    simulatePriceUpdate(type, symbol) {
        // Generate random price changes
        const basePrice = this.subscriptions[`${type}:${symbol}`].basePrice;
        const randomChange = (Math.random() - 0.5) * 2; // Random value between -1 and 1
        const change = parseFloat((randomChange * (basePrice * 0.01)).toFixed(2)); // Up to 1% change
        const newPrice = parseFloat((basePrice + change).toFixed(2));
        const changePercent = parseFloat(((change / basePrice) * 100).toFixed(2));
        const timestamp = new Date().toISOString();

        // Create update data
        const data = {
            price: newPrice,
            change: change,
            change_percent: changePercent,
            timestamp: timestamp
        };

        // Update the display
        this.updatePriceDisplay(type, symbol, data);

        // Update the base price for next time
        this.subscriptions[`${type}:${symbol}`].basePrice = newPrice;

        // Call registered callbacks
        const key = `${type}:${symbol}`;
        if (this.callbacks[key]) {
            this.callbacks[key].forEach(callback => {
                try {
                    callback(data);
                } catch (err) {
                    console.error(`Error in callback for ${key}:`, err);
                }
            });
        }

        // Dispatch a custom event
        document.dispatchEvent(new CustomEvent('realtime:update', {
            detail: {
                symbol,
                assetType: type,
                data
            }
        }));
    }

    /**
     * Update UI to show connection status
     */
    updateConnectionStatus() {
        // Update UI elements to show connection status
        document.querySelectorAll('.price-updated').forEach(el => {
            if (this.isConnected) {
                el.textContent = 'Price data is being updated';
                el.classList.remove('disconnected');
                el.classList.add('connected');
            } else {
                el.textContent = 'Initializing price data...';
                el.classList.remove('connected');
                el.classList.add('disconnected');
            }
        });
    }

    /**
     * Subscribe to updates for a symbol
     */
    subscribe(symbol, type = 'stock', initialPrice = null) {
        if (!symbol) return;

        // Determine initial price if not provided
        if (initialPrice === null) {
            // Default prices by asset type
            if (type === 'stock') {
                initialPrice = 100 + Math.random() * 900; // $100-$1000
            } else if (type === 'crypto') {
                initialPrice = 1000 + Math.random() * 49000; // $1000-$50000
            } else {
                initialPrice = 50 + Math.random() * 150; // $50-$200
            }
        }

        const key = `${type}:${symbol}`;
        this.subscriptions[key] = {
            basePrice: initialPrice
        };

        console.log(`Subscribed to ${type}:${symbol} with initial price $${initialPrice.toFixed(2)}`);

        // Immediately simulate a price update
        if (this.isConnected) {
            setTimeout(() => this.simulatePriceUpdate(type, symbol), 500);
        }
    }

    /**
     * Unsubscribe from updates for a symbol
     */
    unsubscribe(symbol, type = 'stock') {
        if (!symbol) return;

        const key = `${type}:${symbol}`;
        delete this.subscriptions[key];
        console.log(`Unsubscribed from ${type}:${symbol}`);
    }

    /**
     * Register a callback for a specific symbol
     */
    onUpdate(symbol, type, callback) {
        if (!symbol || typeof callback !== 'function') return;

        const key = `${type}:${symbol}`;

        if (!this.callbacks[key]) {
            this.callbacks[key] = [];
        }

        this.callbacks[key].push(callback);

        // Subscribe if not already subscribed
        if (!this.subscriptions[key]) {
            this.subscribe(symbol, type);
        }
    }

    /**
     * Update the price display for a symbol
     */
    updatePriceDisplay(type, symbol, data) {
        const container = document.getElementById(`realtime-price-${symbol}`);
        if (!container) return;

        const priceValue = container.querySelector('.price-value');
        const priceChange = container.querySelector('.price-change');
        const priceUpdated = container.querySelector('.price-updated');

        if (priceValue) {
            priceValue.textContent = `$${data.price.toFixed(2)}`;
        }

        if (priceChange) {
            const isPositive = data.change >= 0;
            priceChange.textContent = `${isPositive ? '+' : ''}${data.change.toFixed(2)} (${isPositive ? '+' : ''}${data.change_percent.toFixed(2)}%)`;
            priceChange.className = `price-change ${isPositive ? 'positive' : 'negative'}`;
        }

        if (priceUpdated && this.isConnected) {
            const date = new Date(data.timestamp);
            priceUpdated.textContent = `Last updated: ${date.toLocaleTimeString()}`;
        }

        // Flash effect
        container.classList.add('flash');
        setTimeout(() => {
            container.classList.remove('flash');
        }, 1000);
    }
}

// Create a global instance
const realtimeData = new RealtimeDataManager();
