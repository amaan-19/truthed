# Truthed

> Privacy-first browser extension for identifying potentially misleading content

## Status: Early Development

This is a solo project in active development. The extension works but has significant limitations.

## What It Does

Truthed analyzes web content as you browse and provides credibility scores with explanations. It aims to educate users about potential misinformation rather than block or censor content.

**Key Features:**
- Real-time content analysis using pattern detection
- Credibility scoring (0-100) with detailed explanations  
- Privacy-first: all processing happens locally in your browser
- Non-intrusive interface that doesn't block content

## Installation (Development)

```bash
git clone https://github.com/yourusername/truthed.git
cd truthed/extension
npm install
npm run build

# Load in Chrome:
# chrome://extensions/ → Developer mode → Load unpacked → select dist folder
```

## Current Limitations

**Detection Accuracy:**
- Uses basic pattern matching, not sophisticated AI
- May flag legitimate content (false positives likely)
- Works better on news articles than social media
- Misses advanced misinformation techniques

**Technical:**
- Chrome only (no Firefox/Safari yet)
- Simple heuristics only
- Limited source credibility database
- Basic content extraction

**Development:**
- Solo developer with limited time (10-20 hours/week)
- Learning project - some decisions may need refactoring
- Slow but steady progress expected

## How It Works

1. **Content Analysis**: Scans for clickbait patterns, missing author info, suspicious language
2. **Credibility Scoring**: Assigns 0-100 score based on detected issues
3. **User Education**: Shows WHY content might be questionable
4. **Privacy Protection**: No data leaves your browser

## Privacy Promise

- No data collection or tracking
- All analysis happens on your device
- No external servers for content processing
- Open source and auditable

## Roadmap

**Phase 1 (Current):** Basic extension with pattern detection  
**Phase 2 (3-6 months):** AI integration, image analysis, source database  
**Phase 3 (6-12 months):** Multi-browser support, premium features

## Why This Project Exists

I believe in helping people become better informed without compromising their privacy or freedom to read what they want. This is a passion project focused on education and media literacy, not a venture-scale startup.

## Contact

Report issues via GitHub Issues or email amaanakhan523@gmail.com for other inquiries.

## License

MIT
