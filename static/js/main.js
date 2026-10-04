/**
 * VeritasAI - Frontend JavaScript Logic
 * Handles API communication, UI updates, sample loader, and session history.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const articleTitleInput = document.getElementById('articleTitle');
    const articleTextInput = document.getElementById('articleText');
    const textCounter = document.getElementById('textCounter');
    const btnClear = document.getElementById('btnClear');
    const analyzeForm = document.getElementById('analyzeForm');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const sampleChipsContainer = document.getElementById('sampleChipsContainer');

    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsContent = document.getElementById('resultsContent');
    const resultBanner = document.getElementById('resultBanner');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictTag = document.getElementById('verdictTag');
    const verdictTitle = document.getElementById('verdictTitle');
    const confidenceValue = document.getElementById('confidenceValue');

    const probRealText = document.getElementById('probRealText');
    const probFakeText = document.getElementById('probFakeText');
    const probBarReal = document.getElementById('probBarReal');
    const probBarFake = document.getElementById('probBarFake');

    const tabExplain = document.getElementById('tabExplain');
    const tabHighlight = document.getElementById('tabHighlight');
    const tabNLP = document.getElementById('tabNLP');
    const panelExplain = document.getElementById('panelExplain');
    const panelHighlight = document.getElementById('panelHighlight');
    const panelNLP = document.getElementById('panelNLP');

    const fakeTagsContainer = document.getElementById('fakeTagsContainer');
    const realTagsContainer = document.getElementById('realTagsContainer');
    const highlightedTextBox = document.getElementById('highlightedTextBox');

    const sensationalScore = document.getElementById('sensationalScore');
    const sensationalBar = document.getElementById('sensationalBar');
    const sensationalSubtext = document.getElementById('sensationalSubtext');
    const clickbaitScore = document.getElementById('clickbaitScore');
    const clickbaitBar = document.getElementById('clickbaitBar');
    const clickbaitSubtext = document.getElementById('clickbaitSubtext');
    const subjectivityScore = document.getElementById('subjectivityScore');
    const subjectivityBar = document.getElementById('subjectivityBar');
    const subjectivitySubtext = document.getElementById('subjectivitySubtext');
    const lexicalScore = document.getElementById('lexicalScore');
    const lexicalBar = document.getElementById('lexicalBar');

    const historyGrid = document.getElementById('historyGrid');
    const btnClearHistory = document.getElementById('btnClearHistory');

    const btnModelInfo = document.getElementById('btnModelInfo');
    const modelModal = document.getElementById('modelModal');
    const btnCloseModal = document.getElementById('btnCloseModal');

    // Session History storage
    let sessionHistory = [];
    let samplesList = [];

    // --- 1. Text Counter ---
    function updateCounter() {
        const text = articleTextInput.value.trim();
        const wordCount = text ? text.split(/\s+/).length : 0;
        const charCount = articleTextInput.value.length;
        textCounter.textContent = `${wordCount} word${wordCount !== 1 ? 's' : ''} | ${charCount} char${charCount !== 1 ? 's' : ''}`;
    }

    articleTextInput.addEventListener('input', updateCounter);
    articleTitleInput.addEventListener('input', updateCounter);

    // --- 2. Clear Form ---
    btnClear.addEventListener('click', () => {
        articleTitleInput.value = '';
        articleTextInput.value = '';
        updateCounter();
        resultsContent.style.display = 'none';
        resultsPlaceholder.style.display = 'block';
        showToast('Form cleared');
    });

    // --- 3. Fetch Samples ---
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            samplesList = data.samples || [];

            sampleChipsContainer.innerHTML = '';
            samplesList.forEach(sample => {
                const chip = document.createElement('div');
                chip.className = 'sample-chip';
                chip.innerHTML = `
                    <span class="chip-tag tag-${sample.badge_color}"></span>
                    <span>${sample.category}</span>
                `;
                chip.addEventListener('click', () => populateSample(sample));
                sampleChipsContainer.appendChild(chip);
            });
        } catch (err) {
            console.error('Error fetching samples:', err);
            sampleChipsContainer.innerHTML = '<span class="no-tags">Failed to load samples</span>';
        }
    }

    function populateSample(sample) {
        articleTitleInput.value = sample.title;
        articleTextInput.value = sample.text;
        updateCounter();
        showToast(`Loaded "${sample.category}" sample`);
        analyzeArticle();
    }

    // --- 4. Analyze Article API Call ---
    async function analyzeArticle() {
        const title = articleTitleInput.value.trim();
        const text = articleTextInput.value.trim();

        if (!title && !text) {
            showToast('Please enter an article title or body text', 'warning');
            return;
        }

        // Set Loading State
        btnAnalyze.disabled = true;
        analyzeSpinner.style.display = 'inline-block';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error during analysis');
            }

            const result = await response.json();
            renderResults(result, title, text);
            addToHistory(result, title || text.substring(0, 60) + '...');

        } catch (err) {
            console.error('Analysis error:', err);
            showToast(err.message, 'error');
        } finally {
            btnAnalyze.disabled = false;
            analyzeSpinner.style.display = 'none';
        }
    }

    analyzeForm.addEventListener('submit', (e) => {
        e.preventDefault();
        analyzeArticle();
    });

    // --- 5. Render Analysis Results ---
    function renderResults(res, title, text) {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'block';

        const isReal = res.prediction === 'Real';
        
        // Banner styling
        resultBanner.className = `result-banner ${isReal ? 'real-banner' : 'fake-banner'}`;
        verdictIcon.innerHTML = isReal ? '<i class="fa-solid fa-shield-check"></i>' : '<i class="fa-solid fa-triangle-exclamation"></i>';
        verdictTag.textContent = isReal ? 'LIKELY REAL NEWS' : 'LIKELY FAKE NEWS';
        verdictTitle.textContent = isReal ? 'Article Appears Genuine & Authentic' : 'Suspicious Phrasing & Fake Indicators Detected';
        confidenceValue.textContent = `${res.confidence}%`;

        // Dual Progress Bar
        probRealText.textContent = `${res.probabilities.real}%`;
        probFakeText.textContent = `${res.probabilities.fake}%`;
        probBarReal.style.width = `${res.probabilities.real}%`;
        probBarFake.style.width = `${res.probabilities.fake}%`;

        // Influencers Pills
        renderTags(fakeTagsContainer, res.key_influencers.fake_indicators, 'fake-pill');
        renderTags(realTagsContainer, res.key_influencers.real_indicators, 'real-pill');

        // Highlighted Text
        renderHighlightedText(res.highlighted_segments);

        // NLP Metrics Cards
        const nlp = res.linguistic_analysis;
        const summary = res.summary;

        sensationalScore.textContent = `${nlp.sensationalism_score}%`;
        sensationalBar.style.width = `${nlp.sensationalism_score}%`;
        sensationalSubtext.textContent = `${summary.sensationalism_level} sensational language`;

        clickbaitScore.textContent = `${nlp.clickbait_score}%`;
        clickbaitBar.style.width = `${nlp.clickbait_score}%`;
        clickbaitSubtext.textContent = `${summary.clickbait_risk} clickbait risk`;

        subjectivityScore.textContent = `${nlp.subjectivity_score}%`;
        subjectivityBar.style.width = `${nlp.subjectivity_score}%`;
        subjectivitySubtext.textContent = summary.subjectivity_level;

        lexicalScore.textContent = `${nlp.lexical_diversity}%`;
        lexicalBar.style.width = `${nlp.lexical_diversity}%`;
    }

    function renderTags(container, items, pillClass) {
        container.innerHTML = '';
        if (!items || items.length === 0) {
            container.innerHTML = '<span class="no-tags">None identified</span>';
            return;
        }
        items.forEach(item => {
            const pill = document.createElement('span');
            pill.className = `tag-pill ${pillClass}`;
            pill.textContent = `${item.term} (${item.weight})`;
            container.appendChild(pill);
        });
    }

    function renderHighlightedText(segments) {
        highlightedTextBox.innerHTML = '';
        if (!segments || segments.length === 0) {
            highlightedTextBox.textContent = 'No text available for highlighting.';
            return;
        }
        segments.forEach(seg => {
            if (seg.flagged) {
                const mark = document.createElement('mark');
                mark.className = 'highlighted-flag';
                mark.textContent = seg.text;
                highlightedTextBox.appendChild(mark);
            } else {
                const span = document.createElement('span');
                span.textContent = seg.text;
                highlightedTextBox.appendChild(span);
            }
        });
    }

    // --- 6. Tab Navigation ---
    const tabs = [
        { btn: tabExplain, panel: panelExplain },
        { btn: tabHighlight, panel: panelHighlight },
        { btn: tabNLP, panel: panelNLP }
    ];

    tabs.forEach(tab => {
        tab.btn.addEventListener('click', () => {
            tabs.forEach(t => {
                t.btn.classList.remove('active');
                t.panel.style.display = 'none';
                t.panel.classList.remove('active');
            });
            tab.btn.classList.add('active');
            tab.panel.style.display = 'block';
            tab.panel.classList.add('active');
        });
    });

    // --- 7. History Management ---
    function addToHistory(result, displayTitle) {
        const item = {
            id: Date.now(),
            title: displayTitle,
            prediction: result.prediction,
            confidence: result.confidence,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            raw: { title: articleTitleInput.value, text: articleTextInput.value }
        };
        sessionHistory.unshift(item);
        renderHistory();
    }

    function renderHistory() {
        historyGrid.innerHTML = '';
        if (sessionHistory.length === 0) {
            historyGrid.innerHTML = '<div class="history-empty">No articles analyzed in this session yet.</div>';
            return;
        }

        sessionHistory.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-item';
            const isReal = item.prediction === 'Real';
            card.innerHTML = `
                <div class="history-top">
                    <span class="history-tag ${isReal ? 'tag-real' : 'tag-fake'}">
                        ${isReal ? 'REAL' : 'FAKE'} (${item.confidence}%)
                    </span>
                    <span class="history-time">${item.time}</span>
                </div>
                <div class="history-title">${escapeHtml(item.title)}</div>
            `;
            card.addEventListener('click', () => {
                articleTitleInput.value = item.raw.title;
                articleTextInput.value = item.raw.text;
                updateCounter();
                analyzeArticle();
            });
            historyGrid.appendChild(card);
        });
    }

    btnClearHistory.addEventListener('click', () => {
        sessionHistory = [];
        renderHistory();
        showToast('History cleared');
    });

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // --- 8. Model Specs Modal ---
    btnModelInfo.addEventListener('click', async () => {
        modelModal.style.display = 'flex';
        try {
            const res = await fetch('/api/model-info');
            const metrics = await res.json();

            document.getElementById('specModelType').textContent = metrics.model_type || 'Logistic Regression';
            document.getElementById('specAccuracy').textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
            document.getElementById('specPrecision').textContent = metrics.precision.toFixed(3);
            document.getElementById('specRecall').textContent = metrics.recall.toFixed(3);
            document.getElementById('specF1').textContent = metrics.f1_score.toFixed(3);
            document.getElementById('specVocab').textContent = `${metrics.vocabulary_size.toLocaleString()} N-Grams`;

            const cm = metrics.confusion_matrix;
            if (cm && cm.length === 2) {
                document.getElementById('cmTN').textContent = cm[0][0]; // True Fake
                document.getElementById('cmFP').textContent = cm[0][1]; // False Real
                document.getElementById('cmFN').textContent = cm[1][0]; // False Fake
                document.getElementById('cmTP').textContent = cm[1][1]; // True Real
            }
        } catch (err) {
            console.error('Error fetching model specs:', err);
        }
    });

    btnCloseModal.addEventListener('click', () => {
        modelModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modelModal) {
            modelModal.style.display = 'none';
        }
    });

    // --- Toast Notifications ---
    function showToast(msg, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<i class="fa-solid fa-circle-info text-indigo"></i> <span>${msg}</span>`;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    // --- Initialize ---
    loadSamples();
});
