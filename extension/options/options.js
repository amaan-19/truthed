// Options page script for Truthed extension
document.addEventListener('DOMContentLoaded', () => {
    const autoAnalyzeCheckbox = document.getElementById('auto-analyze');
    const sensitivitySelect = document.getElementById('sensitivity');
    const saveBtn = document.getElementById('save-btn');
    const status = document.getElementById('status');

    // Load saved settings
    loadSettings();

    // Save settings when button is clicked
    saveBtn.addEventListener('click', saveSettings);

    // Auto-save when settings change
    autoAnalyzeCheckbox.addEventListener('change', saveSettings);
    sensitivitySelect.addEventListener('change', saveSettings);

    async function loadSettings() {
        try {
            const settings = await chrome.storage.sync.get({
                autoAnalyze: false,
                sensitivity: 'medium'
            });

            autoAnalyzeCheckbox.checked = settings.autoAnalyze;
            sensitivitySelect.value = settings.sensitivity;
        } catch (error) {
            console.error('Error loading settings:', error);
            showStatus('Error loading settings', 'error');
        }
    }

    async function saveSettings() {
        try {
            const settings = {
                autoAnalyze: autoAnalyzeCheckbox.checked,
                sensitivity: sensitivitySelect.value
            };

            await chrome.storage.sync.set(settings);
            showStatus('Settings saved!', 'success');

            // Notify content scripts of settings change
            const tabs = await chrome.tabs.query({});
            tabs.forEach(tab => {
                if (tab.url.startsWith('http')) {
                    chrome.tabs.sendMessage(tab.id, {
                        action: 'settingsUpdated',
                        settings: settings
                    }).catch(() => {
                        // Ignore errors for tabs without content script
                    });
                }
            });
        } catch (error) {
            console.error('Error saving settings:', error);
            showStatus('Error saving settings', 'error');
        }
    }

    function showStatus(message, type) {
        if (!status) return;

        status.textContent = message;
        status.className = `status ${type}`;
        status.style.display = 'block';

        // Hide status after 3 seconds
        setTimeout(() => {
            status.style.display = 'none';
        }, 3000);
    }

    // Add privacy information
    const privacyInfo = document.getElementById('privacy-info');
    if (privacyInfo) {
        privacyInfo.innerHTML = `
            <h3>Privacy Promise</h3>
            <ul>
                <li>All analysis happens locally in your browser</li>
                <li>No browsing data is sent to external servers</li>
                <li>No tracking or analytics</li>
                <li>Your data stays private</li>
            </ul>
        `;
    }
});