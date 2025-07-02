// Content script for Truthed extension
console.log('Truthed content script loaded on:', window.location.href);

// Function to highlight suspicious content
function highlightSuspiciousContent() {
    // This is where we'll add content analysis logic
    console.log('Highlighting suspicious content...');

    // For now, just add a simple indicator
    const body = document.body;
    if (body) {
        const indicator = document.createElement('div');
        indicator.id = 'truthed-indicator';
        indicator.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: #007cba;
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      z-index: 10000;
      font-family: Arial, sans-serif;
    `;
        indicator.textContent = 'Truthed: Ready';
        body.appendChild(indicator);
    }
}

// Run when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', highlightSuspiciousContent);
} else {
    highlightSuspiciousContent();
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getPageContent') {
        const content = {
            title: document.title,
            url: window.location.href,
            text: document.body.innerText.substring(0, 1000) // First 1000 chars
        };
        sendResponse(content);
    }
});