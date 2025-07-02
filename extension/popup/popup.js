// Popup script for Truthed extension
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const status = document.getElementById('status');
    const results = document.getElementById('results');

    analyzeBtn.addEventListener('click', async () => {
        status.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;

        try {
            // Get current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Get page content from content script
            const response = await chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' });

            // For now, just show basic info
            results.innerHTML = `
        <h3>Page Analysis</h3>
        <p><strong>Title:</strong> ${response.title}</p>
        <p><strong>URL:</strong> ${response.url}</p>
        <p><strong>Preview:</strong> ${response.text.substring(0, 100)}...</p>
      `;

            status.textContent = 'Analysis complete';
        } catch (error) {
            console.error('Error analyzing page:', error);
            status.textContent = 'Error analyzing page';
            results.textContent = 'Could not analyze this page';
        }

        analyzeBtn.disabled = false;
    });
});