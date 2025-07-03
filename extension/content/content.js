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

function extractArticleContent() {
    // Clone document to avoid modifying original page
    const documentClone = document.cloneNode(true);
    console.log('Cloned document. Creating reader now...');

    // Extract article content
    const reader = new Readability(documentClone);
    const article = reader.parse();

    return article;
}

// Run when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', highlightSuspiciousContent);
} else {
    highlightSuspiciousContent();
    const articleData = extractArticleContent();
    if (articleData) {
        console.log("Title:", articleData.title);
        console.log("Text length:", articleData.length);
        console.log("Author:", articleData.byline);
        // This is what you'll send to your credibility analysis
        console.log("Content:", articleData.textContent);
    } else {
        console.log("extractArticleContent() was not called.")
    }
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