// Enhanced popup script for Truthed extension
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const status = document.getElementById('status');
    const results = document.getElementById('results');

    // Load settings
    loadSettings();

    analyzeBtn.addEventListener('click', async () => {
        status.textContent = 'Analyzing page...';
        status.className = 'status analyzing';
        analyzeBtn.disabled = true;
        results.innerHTML = '';

        try {
            // Get current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Check if we can access this tab
            if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://')) {
                throw new Error('Cannot analyze this page type');
            }

            // Get analysis from content script
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'analyzeContent'
            });

            if (response && response.analysis) {
                displayAnalysisResults(response.article, response.analysis);
                status.textContent = 'Analysis complete';
                status.className = 'status complete';
            } else {
                throw new Error('No analysis data received');
            }

        } catch (error) {
            console.error('Error analyzing page:', error);
            status.textContent = 'Error analyzing page';
            status.className = 'status error';

            if (error.message.includes('Cannot analyze')) {
                results.innerHTML = `
                    <div class="error">
                        <p>Cannot analyze this page type. Truthed works on web pages (http/https).</p>
                    </div>
                `;
            } else {
                results.innerHTML = `
                    <div class="error">
                        <p>Could not analyze this page. Make sure the page is fully loaded and try again.</p>
                    </div>
                `;
            }
        }

        analyzeBtn.disabled = false;
    });

    // Add settings button handler
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => {
            chrome.runtime.openOptionsPage();
        });
    }
});

function displayAnalysisResults(article, analysis) {
    const results = document.getElementById('results');

    // Determine credibility level
    let credibilityClass = 'high';
    let credibilityText = 'High';
    if (analysis.score < 60) {
        credibilityClass = 'low';
        credibilityText = 'Low';
    } else if (analysis.score < 80) {
        credibilityClass = 'medium';
        credibilityText = 'Medium';
    }

    results.innerHTML = `
        <div class="analysis-result">
            <h3>Credibility Analysis</h3>
            
            <div class="score-container">
                <div class="score ${credibilityClass}">
                    ${analysis.score}/100
                </div>
                <div class="score-label">${credibilityText} Credibility</div>
            </div>

            <div class="article-info">
                <h4>Article Information</h4>
                <p><strong>Title:</strong> ${truncateText(article.title, 60)}</p>
                <p><strong>Domain:</strong> ${article.domain}</p>
                ${article.byline ? `<p><strong>Author:</strong> ${article.byline}</p>` : ''}
                ${article.publishDate ? `<p><strong>Published:</strong> ${article.publishDate}</p>` : ''}
                <p><strong>Content Length:</strong> ${article.length} characters</p>
            </div>

            ${generateFlagsSection(analysis.flags)}
            
            <div class="analysis-text">
                <h4>Analysis Summary</h4>
                <p>${analysis.analysis}</p>
            </div>

            <div class="learn-more">
                <h4>Learn More</h4>
                <p>This analysis is based on content patterns, source information, and basic credibility indicators. Always verify important information through multiple sources.</p>
            </div>
        </div>
    `;
}

function generateFlagsSection(flags) {
    const flagsArray = Object.entries(flags).filter(([key, value]) => value);

    if (flagsArray.length === 0) {
        return `
            <div class="flags-section">
                <h4>Quality Indicators</h4>
                <div class="flag positive">✓ No major red flags detected</div>
            </div>
        `;
    }

    const flagsHtml = flagsArray.map(([key, value]) => {
        const flagText = {
            clickbait: 'Clickbait language detected',
            excessiveCaps: 'Excessive capitalization',
            shortContent: 'Very short content',
            missingAuthor: 'No author information',
            missingDate: 'No publication date'
        };

        return `<div class="flag negative">⚠ ${flagText[key]}</div>`;
    }).join('');

    return `
        <div class="flags-section">
            <h4>Issues Detected</h4>
            ${flagsHtml}
        </div>
    `;
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

async function loadSettings() {
    try {
        const settings = await chrome.storage.sync.get(['autoAnalyze', 'sensitivity']);
        // Use settings if needed for popup display
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}