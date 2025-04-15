/**
 * Social sharing functionality for the Financial Analysis Dashboard
 */

class SocialSharing {
    /**
     * Initialize social sharing functionality
     */
    constructor() {
        this.initShareButtons();
    }

    /**
     * Initialize share buttons
     */
    initShareButtons() {
        document.addEventListener('DOMContentLoaded', () => {
            // Find all share buttons
            const shareButtons = document.querySelectorAll('.share-button');
            
            // Add click event listeners
            shareButtons.forEach(button => {
                button.addEventListener('click', (event) => {
                    event.preventDefault();
                    const platform = button.getAttribute('data-platform');
                    const url = button.getAttribute('data-url') || window.location.href;
                    const title = button.getAttribute('data-title') || document.title;
                    const description = button.getAttribute('data-description') || '';
                    
                    this.shareContent(platform, url, title, description);
                });
            });
            
            // Initialize copy link buttons
            const copyButtons = document.querySelectorAll('.copy-link-button');
            copyButtons.forEach(button => {
                button.addEventListener('click', (event) => {
                    event.preventDefault();
                    const url = button.getAttribute('data-url') || window.location.href;
                    this.copyToClipboard(url, button);
                });
            });
        });
    }

    /**
     * Share content on a specific platform
     * 
     * @param {string} platform - The platform to share on (twitter, facebook, linkedin, etc.)
     * @param {string} url - The URL to share
     * @param {string} title - The title of the content
     * @param {string} description - The description of the content
     */
    shareContent(platform, url, title, description) {
        let shareUrl = '';
        
        switch (platform) {
            case 'twitter':
                shareUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title)}`;
                break;
            case 'facebook':
                shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
                break;
            case 'linkedin':
                shareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`;
                break;
            case 'reddit':
                shareUrl = `https://www.reddit.com/submit?url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`;
                break;
            case 'email':
                shareUrl = `mailto:?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(description + '\n\n' + url)}`;
                break;
            default:
                console.error(`Unsupported platform: ${platform}`);
                return;
        }
        
        // Open share dialog
        window.open(shareUrl, '_blank', 'width=600,height=400');
    }

    /**
     * Copy a URL to the clipboard
     * 
     * @param {string} text - The text to copy
     * @param {HTMLElement} button - The button element that was clicked
     */
    copyToClipboard(text, button) {
        // Create a temporary input element
        const input = document.createElement('input');
        input.style.position = 'fixed';
        input.style.opacity = 0;
        input.value = text;
        document.body.appendChild(input);
        
        // Select and copy the text
        input.select();
        document.execCommand('copy');
        
        // Remove the temporary input
        document.body.removeChild(input);
        
        // Show feedback
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.classList.add('copied');
        
        // Reset button text after a delay
        setTimeout(() => {
            button.textContent = originalText;
            button.classList.remove('copied');
        }, 2000);
    }

    /**
     * Create a shareable image from an element
     * 
     * @param {string} elementId - The ID of the element to capture
     * @returns {Promise<string>} - A promise that resolves to a data URL of the image
     */
    async createShareableImage(elementId) {
        try {
            // Check if html2canvas is available
            if (typeof html2canvas === 'undefined') {
                console.error('html2canvas is not loaded');
                return null;
            }
            
            const element = document.getElementById(elementId);
            if (!element) {
                console.error(`Element with ID ${elementId} not found`);
                return null;
            }
            
            // Create canvas from element
            const canvas = await html2canvas(element, {
                backgroundColor: '#1e1e1e',
                scale: 2,
                logging: false,
                useCORS: true
            });
            
            // Convert canvas to data URL
            return canvas.toDataURL('image/png');
        } catch (error) {
            console.error('Error creating shareable image:', error);
            return null;
        }
    }

    /**
     * Download a shareable image
     * 
     * @param {string} elementId - The ID of the element to capture
     * @param {string} filename - The filename for the downloaded image
     */
    async downloadShareableImage(elementId, filename = 'financial-analysis.png') {
        const dataUrl = await this.createShareableImage(elementId);
        if (!dataUrl) return;
        
        // Create a download link
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        
        // Trigger download
        link.click();
        
        // Clean up
        document.body.removeChild(link);
    }

    /**
     * Create a social sharing widget
     * 
     * @param {string} containerId - The ID of the container element
     * @param {Object} options - Configuration options
     */
    createSharingWidget(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const defaultOptions = {
            url: window.location.href,
            title: document.title,
            description: '',
            platforms: ['twitter', 'facebook', 'linkedin', 'reddit', 'email'],
            showCopy: true,
            showDownload: false,
            downloadElementId: '',
            downloadFilename: 'financial-analysis.png'
        };
        
        const config = { ...defaultOptions, ...options };
        
        // Create widget HTML
        let html = '<div class="social-sharing-widget">';
        html += '<h3>Share This Analysis</h3>';
        html += '<div class="sharing-buttons">';
        
        // Add platform buttons
        config.platforms.forEach(platform => {
            html += `
                <button class="share-button" data-platform="${platform}" data-url="${config.url}" data-title="${config.title}" data-description="${config.description}">
                    <span class="icon icon-${platform}"></span>
                    <span class="label">${platform.charAt(0).toUpperCase() + platform.slice(1)}</span>
                </button>
            `;
        });
        
        // Add copy link button
        if (config.showCopy) {
            html += `
                <button class="copy-link-button" data-url="${config.url}">
                    <span class="icon icon-link"></span>
                    <span class="label">Copy Link</span>
                </button>
            `;
        }
        
        // Add download button
        if (config.showDownload && config.downloadElementId) {
            html += `
                <button class="download-image-button" data-element-id="${config.downloadElementId}" data-filename="${config.downloadFilename}">
                    <span class="icon icon-download"></span>
                    <span class="label">Download Image</span>
                </button>
            `;
        }
        
        html += '</div></div>';
        
        // Add widget to container
        container.innerHTML = html;
        
        // Initialize download button
        if (config.showDownload && config.downloadElementId) {
            const downloadButton = container.querySelector('.download-image-button');
            if (downloadButton) {
                downloadButton.addEventListener('click', (event) => {
                    event.preventDefault();
                    const elementId = downloadButton.getAttribute('data-element-id');
                    const filename = downloadButton.getAttribute('data-filename');
                    this.downloadShareableImage(elementId, filename);
                });
            }
        }
    }
}

// Initialize social sharing
window.socialSharing = new SocialSharing();
