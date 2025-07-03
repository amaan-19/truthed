// Enhanced background script for Truthed extension
console.log('Truthed background script loaded');

// Listen for extension installation
chrome.runtime.onInstalled.addListener(async () => {
    console.log('Truthed extension installed');

    // Set default settings
    await chrome.storage.sync.set({
        autoAnalyze: false,
        sensitivity: 'medium',
        lastUpdated: Date.now()
    });

    // Show welcome notification
    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'Truthed Installed',
        message: 'Click the extension icon to start analyzing web content for credibility!'
    });
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzePage') {
        console.log('Analyzing page:', request.url);

        // Here you could add API calls to external services
        // For now, we'll just acknowledge the request
        sendResponse({
            status: 'Analysis started',
            timestamp: Date.now()
        });

    } else if (request.action === 'getSettings') {
        // Provide settings to content scripts
        chrome.storage.sync.get(['autoAnalyze', 'sensitivity']).then(settings => {
            sendResponse(settings);
        });
        return true; // Keep message channel open

    } else if (request.action === 'reportIssue') {
        console.log('Issue reported:', request.issue);
        // Handle user feedback/reporting
        sendResponse({ status: 'Issue recorded' });
    }
});

// Handle tab updates for auto-analysis
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    // Only process when page is completely loaded
    if (changeInfo.status === 'complete' && tab.url) {
        // Check if auto-analyze is enabled
        const settings = await chrome.storage.sync.get(['autoAnalyze']);

        if (settings.autoAnalyze && isAnalyzableUrl(tab.url)) {
            // Inject content script if needed and trigger analysis
            try {
                await chrome.tabs.sendMessage(tabId, {
                    action: 'autoAnalyze',
                    url: tab.url
                });
            } catch (error) {
                // Content script not ready or page not accessible
                console.log('Could not auto-analyze:', tab.url);
            }
        }
    }
});

// Badge management for credibility scores
function updateBadge(tabId, score) {
    if (score < 60) {
        chrome.action.setBadgeBackgroundColor({ color: '#dc3545', tabId });
        chrome.action.setBadgeText({ text: '!', tabId });
    } else if (score < 80) {
        chrome.action.setBadgeBackgroundColor({ color: '#ffc107', tabId });
        chrome.action.setBadgeText({ text: '?', tabId });
    } else {
        chrome.action.setBadgeBackgroundColor({ color: '#28a745', tabId });
        chrome.action.setBadgeText({ text: '✓', tabId });
    }
}

// Clear badge when tab changes
chrome.tabs.onActivated.addListener((activeInfo) => {
    chrome.action.setBadgeText({ text: '', tabId: activeInfo.tabId });
});

// Utility function to check if URL can be analyzed
function isAnalyzableUrl(url) {
    return url && (url.startsWith('http://') || url.startsWith('https://'))
        && !url.includes('chrome://')
        && !url.includes('chrome-extension://');
}

// Context menu for right-click analysis
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'analyzePage',
        title: 'Analyze with Truthed',
        contexts: ['page']
    });

    chrome.contextMenus.create({
        id: 'analyzeSelection',
        title: 'Analyze selected text',
        contexts: ['selection']
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === 'analyzePage') {
        // Trigger page analysis
        try {
            await chrome.tabs.sendMessage(tab.id, { action: 'analyzeContent' });
        } catch (error) {
            console.error('Could not analyze page:', error);
        }
    } else if (info.menuItemId === 'analyzeSelection') {
        // Analyze selected text
        try {
            await chrome.tabs.sendMessage(tab.id, {
                action: 'analyzeText',
                text: info.selectionText
            });
        } catch (error) {
            console.error('Could not analyze selection:', error);
        }
    }
});

// Periodic cleanup of old data
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'cleanup') {
        // Clean up old analysis data if we start storing it
        console.log('Running cleanup...');
    }
});

// Set up periodic cleanup
chrome.alarms.create('cleanup', { periodInMinutes: 60 });