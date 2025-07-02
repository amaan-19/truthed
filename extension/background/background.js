// Background script for Truthed extension
console.log('Truthed background script loaded');

// Listen for extension installation
chrome.runtime.onInstalled.addListener(() => {
    console.log('Truthed extension installed');

    // Set default settings
    chrome.storage.sync.set({
        autoAnalyze: false,
        sensitivity: 'medium'
    });
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzePage') {
        // This is where we'll add the AI analysis logic later
        console.log('Analyzing page:', request.url);
        sendResponse({ status: 'Analysis started' });
    }
});