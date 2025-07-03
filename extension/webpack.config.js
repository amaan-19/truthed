const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
    entry: {
        content: './content/content.js',
        popup: './popup/popup.js',
        background: './background/background.js',
        options: './options/options.js'
    },
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: '[name]/[name].js',
    },
    mode: 'production',
    plugins: [
        new CopyPlugin({
            patterns: [
                // Copy manifest
                { from: 'manifest.json', to: 'manifest.json' },

                // Copy HTML files
                { from: 'popup/popup.html', to: 'popup/popup.html' },
                { from: 'options/options.html', to: 'options/options.html' },

                // Copy CSS files
                { from: 'popup/popup.css', to: 'popup/popup.css' },
                { from: 'options/options.css', to: 'options/options.css' },
                { from: 'content/content.css', to: 'content/content.css' },

                // Copy icons
                { from: 'icons/', to: 'icons/' },

                // Copy any other static assets
                { from: 'data/', to: 'data/', noErrorOnMissing: true }
            ],
        }),
    ],
    optimization: {
        minimize: false // Keep readable for debugging
    }
};