// Content script for Truthed extension
console.log('Truthed content script loaded on:', window.location.href);

// Simple text readability extraction function (replaces missing readability.js)
function extractArticleContent() {
    // Try to find main content areas
    const contentSelectors = [
        'article',
        '[role="main"]',
        '.post-content',
        '.article-content',
        '.entry-content',
        '.content',
        'main'
    ];

    let mainContent = null;
    for (const selector of contentSelectors) {
        mainContent = document.querySelector(selector);
        if (mainContent) break;
    }

    // Fallback to body if no main content found
    if (!mainContent) {
        mainContent = document.body;
    }

    // Extract text and basic metadata
    const article = {
        title: document.title,
        url: window.location.href,
        textContent: mainContent ? mainContent.innerText : '',
        length: mainContent ? mainContent.innerText.length : 0,
        byline: extractByline(),
        publishDate: extractPublishDate(),
        domain: window.location.hostname
    };

    return article;
}

function extractByline() {
    const bylineSelectors = [
        '[rel="author"]',
        '.author',
        '.byline',
        '[class*="author"]',
        '.writer'
    ];

    for (const selector of bylineSelectors) {
        const element = document.querySelector(selector);
        if (element) return element.textContent.trim();
    }
    return null;
}

function extractPublishDate() {
    // Look for common date patterns
    const dateSelectors = [
        'time[datetime]',
        '.publish-date',
        '.date',
        '[class*="date"]'
    ];

    for (const selector of dateSelectors) {
        const element = document.querySelector(selector);
        if (element) {
            return element.getAttribute('datetime') || element.textContent.trim();
        }
    }
    return null;
}

// Function to analyze content for potential misinformation
async function analyzeContent(article) {
    // Basic heuristics for suspicious content
    const suspiciousPatterns = [
        /you won't believe/gi,
        /doctors hate/gi,
        /this one trick/gi,
        /breaking:?\s*[^.]*!/gi,
        /urgent:?\s*[^.]*!/gi,
        /must share/gi,
        /before it's too late/gi
    ];

    const clickbaitScore = suspiciousPatterns.reduce((score, pattern) => {
        return score + (pattern.test(article.title) ? 1 : 0);
    }, 0);

    // Check for excessive capitalization
    const capsRatio = (article.title.match(/[A-Z]/g) || []).length / article.title.length;

    // Basic credibility scoring
    let credibilityScore = 100;

    if (clickbaitScore > 0) credibilityScore -= clickbaitScore * 15;
    if (capsRatio > 0.3) credibilityScore -= 20;
    if (article.textContent.length < 200) credibilityScore -= 10;
    if (!article.byline) credibilityScore -= 10;
    if (!article.publishDate) credibilityScore -= 5;

    credibilityScore = Math.max(0, Math.min(100, credibilityScore));

    return {
        score: credibilityScore,
        flags: {
            clickbait: clickbaitScore > 0,
            excessiveCaps: capsRatio > 0.3,
            shortContent: article.textContent.length < 200,
            missingAuthor: !article.byline,
            missingDate: !article.publishDate
        },
        analysis: generateAnalysisText(credibilityScore, clickbaitScore, capsRatio)
    };
}

function generateAnalysisText(score, clickbait, capsRatio) {
    const issues = [];
    if (clickbait > 0) issues.push('clickbait language detected');
    if (capsRatio > 0.3) issues.push('excessive capitalization');

    if (score >= 80) return 'Content appears credible';
    if (score >= 60) return `Some concerns: ${issues.join(', ')}`;
    return `Multiple red flags: ${issues.join(', ')}`;
}

// Function to display credibility indicator
function displayCredibilityIndicator(analysis) {
    // Remove existing indicator
    const existing = document.getElementById('truthed-indicator');
    if (existing) existing.remove();

    const indicator = document.createElement('div');
    indicator.id = 'truthed-indicator';

    // Determine color based on score
    let color = '#28a745'; // Green
    if (analysis.score < 60) color = '#dc3545'; // Red
    else if (analysis.score < 80) color = '#ffc107'; // Yellow

    indicator.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: ${color};
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 14px;
        font-family: Arial, sans-serif;
        z-index: 10000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        cursor: pointer;
        min-width: 200px;
    `;

    indicator.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 5px;">
            Truthed: ${analysis.score}/100
        </div>
        <div style="font-size: 12px;">
            ${analysis.analysis}
        </div>
    `;

    // Add click handler for detailed view
    indicator.addEventListener('click', () => {
        showDetailedAnalysis(analysis);
    });

    document.body.appendChild(indicator);
}

function showDetailedAnalysis(analysis) {
    const modal = document.createElement('div');
    modal.id = 'truthed-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 10001;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    const content = document.createElement('div');
    content.style.cssText = `
        background: white;
        padding: 20px;
        border-radius: 8px;
        max-width: 500px;
        width: 90%;
        font-family: Arial, sans-serif;
    `;

    content.innerHTML = `
        <h3 style="margin-top: 0;">Credibility Analysis</h3>
        <p><strong>Overall Score:</strong> ${analysis.score}/100</p>
        <h4>Detected Issues:</h4>
        <ul>
            ${analysis.flags.clickbait ? '<li>Clickbait language detected</li>' : ''}
            ${analysis.flags.excessiveCaps ? '<li>Excessive capitalization</li>' : ''}
            ${analysis.flags.shortContent ? '<li>Very short content</li>' : ''}
            ${analysis.flags.missingAuthor ? '<li>No author information</li>' : ''}
            ${analysis.flags.missingDate ? '<li>No publication date</li>' : ''}
        </ul>
        <button id="truthed-close" style="
            background: #007cba;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        ">Close</button>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);

    // Close modal handlers
    document.getElementById('truthed-close').addEventListener('click', () => {
        modal.remove();
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// Auto-analyze content when page loads
async function autoAnalyze() {
    // Wait a bit for page to fully load
    setTimeout(async () => {
        const article = extractArticleContent();
        if (article && article.textContent.length > 100) {
            console.log('Extracted article:', article);
            const analysis = await analyzeContent(article);
            console.log('Analysis result:', analysis);
            displayCredibilityIndicator(analysis);
        }
    }, 2000);
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getPageContent') {
        const article = extractArticleContent();
        sendResponse(article);
    } else if (request.action === 'analyzeContent') {
        const article = extractArticleContent();
        analyzeContent(article).then(analysis => {
            sendResponse({ article, analysis });
        });
        return true; // Keep message channel open for async response
    }
});

// Run analysis when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoAnalyze);
} else {
    autoAnalyze();
}