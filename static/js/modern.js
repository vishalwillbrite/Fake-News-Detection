/**
 * VERITAS AI - Modern Fake News Detection System Core Engine
 * Interactive Motions, Particle Matrix, HUD Scanner, and Live Analysis
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initParticles();
    initCounters();
    initNavbarScroll();
    initTextareaMetrics();
    initFormAnalyzer();
});

/* --------------------------------------------------------------------------
   0. Theme Toggle (Dark & Light Mode Switching with LocalStorage Persistence)
   -------------------------------------------------------------------------- */
function initThemeToggle() {
    const btn = document.getElementById("themeToggleBtn");
    if (!btn) return;

    btn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const newTheme = currentTheme === "dark" ? "light" : "dark";

        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("veritas_theme", newTheme);

        showToast(`Switched to ${newTheme.charAt(0).toUpperCase() + newTheme.slice(1)} Mode`);
    });
}

/* --------------------------------------------------------------------------
   1. Interactive Ambient Particle Canvas (Connected Cyber Grid)
   -------------------------------------------------------------------------- */
function initParticles() {
    const canvas = document.getElementById("particleCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    window.addEventListener("resize", () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const count = Math.min(Math.floor((width * height) / 18000), 45);

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.45,
            vy: (Math.random() - 0.5) * 0.45,
            radius: Math.random() * 1.8 + 0.8,
            color: Math.random() > 0.5 ? "rgba(99, 102, 241," : "rgba(6, 182, 212,"
        });
    }

    let animationFrameId;
    function render() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + "0.6)";
            ctx.fill();

            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.15 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                }
            }
        }
        animationFrameId = requestAnimationFrame(render);
    }

    render();
}

/* --------------------------------------------------------------------------
   2. Number Count-up Animation for Statistics
   -------------------------------------------------------------------------- */
function initCounters() {
    const counterElements = document.querySelectorAll("[data-counter-target]");
    if (!counterElements.length) return;

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseFloat(entry.target.getAttribute("data-counter-target"));
                const prefix = entry.target.getAttribute("data-counter-prefix") || "";
                const suffix = entry.target.getAttribute("data-counter-suffix") || "";
                const decimals = parseInt(entry.target.getAttribute("data-counter-decimals") || "0", 10);
                
                animateValue(entry.target, 0, target, 1600, decimals, prefix, suffix);
                obs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    counterElements.forEach(el => observer.observe(el));
}

function animateValue(obj, start, end, duration, decimals, prefix, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = (start + (end - start) * easeOut).toFixed(decimals);
        obj.textContent = `${prefix}${Number(current).toLocaleString()}${suffix}`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.textContent = `${prefix}${end.toLocaleString()}${suffix}`;
        }
    };
    window.requestAnimationFrame(step);
}

/* --------------------------------------------------------------------------
   3. Navbar Scroll Elevation
   -------------------------------------------------------------------------- */
function initNavbarScroll() {
    const navbar = document.querySelector(".modern-navbar");
    if (!navbar) return;

    window.addEventListener("scroll", () => {
        if (window.scrollY > 25) {
            navbar.classList.add("scrolled");
        } else {
            navbar.classList.remove("scrolled");
        }
    });
}

/* --------------------------------------------------------------------------
   4. Textarea Live Metrics & Reading Time
   -------------------------------------------------------------------------- */
function initTextareaMetrics() {
    const textarea = document.getElementById("newsTextarea");
    if (!textarea) return;

    const charBadge = document.getElementById("charCountBadge");
    const wordBadge = document.getElementById("wordCountBadge");
    const readTimeBadge = document.getElementById("readTimeBadge");

    const updateMetrics = () => {
        const text = textarea.value;
        const chars = text.length;
        const wordsArray = text.trim().split(/\s+/).filter(w => w.length > 0);
        const words = wordsArray.length;
        const readTime = Math.ceil(words / 180);

        if (charBadge) charBadge.textContent = `${chars.toLocaleString()} Chars`;
        if (wordBadge) wordBadge.textContent = `${words.toLocaleString()} Words`;
        if (readTimeBadge) {
            readTimeBadge.textContent = words > 0 ? `~${readTime} min read` : "0 min read";
        }
    };

    textarea.addEventListener("input", updateMetrics);
    updateMetrics();
}

/* --------------------------------------------------------------------------
   5. Sample News Loader
   -------------------------------------------------------------------------- */
const SAMPLE_ARTICLES = {
    real: `NASA and international space agency partners have officially finalized testing for the next-generation Earth observation satellite, scheduled for launch next month. The mission aims to deliver high-resolution meteorological telemetry, monitoring ocean surface topography and severe weather patterns with unprecedented sub-millimeter precision. According to senior project scientists, the orbital observatory will enhance global climate resilience and early disaster warnings for over two billion coastal inhabitants.`,
    
    fake: `CONFIDENTIAL DOCUMENTS LEAKED: Pentagon Whistleblower reveals that extraterrestrial technology was recovered from an ancient pyramid hidden three miles under Antarctic ice sheets. According to anonymous sources, global elites have established a secret underwater subterranean treaty with shape-shifting beings to control worldwide banking systems and atmospheric weather satellites. Mainstream media networks have been issued strict gag orders to suppress all aerial photographs.`,
    
    tech: `Leading artificial intelligence laboratories and computer science researchers have published a peer-reviewed benchmark demonstrating quantum-assisted optimization for neural network gradient descent. The new algorithmic architecture significantly decreases energy consumption during parameter training runs while achieving state-of-the-art inference efficiency across large-scale natural language understanding datasets.`
};

window.loadSample = function(type) {
    const textarea = document.getElementById("newsTextarea");
    if (!textarea) return;

    const sample = SAMPLE_ARTICLES[type];
    if (!sample) return;

    textarea.value = sample;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();

    showToast(`Loaded ${type.toUpperCase()} news sample`);
};

window.clearNewsText = function() {
    const textarea = document.getElementById("newsTextarea");
    if (!textarea) return;

    textarea.value = "";
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();

    const resultPanel = document.getElementById("resultPanel");
    if (resultPanel) {
        resultPanel.style.display = "none";
    }

    showToast("Cleared text input");
};

window.pasteFromClipboard = async function() {
    const textarea = document.getElementById("newsTextarea");
    if (!textarea) return;

    try {
        const text = await navigator.clipboard.readText();
        if (text) {
            textarea.value = text;
            textarea.dispatchEvent(new Event("input"));
            textarea.focus();
            showToast("Pasted article from clipboard");
        } else {
            showToast("Clipboard is empty");
        }
    } catch (err) {
        showToast("Clipboard permission denied");
    }
};

/* --------------------------------------------------------------------------
   6. AI HUD Scanner & Live Analyzer Form
   -------------------------------------------------------------------------- */
let latestAnalysisData = null;

function initFormAnalyzer() {
    const form = document.getElementById("newsAnalyzerForm");
    if (!form) return;

    const textarea = document.getElementById("newsTextarea");
    const textareaWrapper = document.querySelector(".textarea-wrapper");
    const scannerHud = document.getElementById("scannerHud");
    const hudFill = document.getElementById("hudProgressFill");
    const hudStepText = document.getElementById("hudStepText");
    const submitBtn = document.getElementById("analyzeSubmitBtn");
    const resultPanel = document.getElementById("resultPanel");

    form.addEventListener("submit", async (e) => {
        const text = textarea ? textarea.value.trim() : "";
        if (!text) {
            e.preventDefault();
            showToast("Please enter or paste a news article first!");
            return;
        }

        // We use AJAX for instant smooth HUD animation without page flicker
        e.preventDefault();

        // 1. Enter Scanning State
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span> Analyzing...`;
        }
        if (textareaWrapper) textareaWrapper.classList.add("scanning");
        if (resultPanel) resultPanel.style.display = "none";
        if (scannerHud) scannerHud.style.display = "block";

        const steps = [
            { text: "[01/05] Ingesting text stream & tokenizing syntax...", pct: 22 },
            { text: "[02/05] Normalizing text & stripping 179 English stopwords...", pct: 44 },
            { text: "[03/05] Computing morphological roots with Porter Stemmer...", pct: 66 },
            { text: "[04/05] Projecting into 5,000-dimensional TF-IDF space...", pct: 86 },
            { text: "[05/05] Calculating Logistic Regression decision boundary...", pct: 100 }
        ];

        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                if (hudStepText) hudStepText.textContent = steps[currentStep].text;
                if (hudFill) hudFill.style.width = `${steps[currentStep].pct}%`;
                currentStep++;
            }
        }, 160);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ news: text })
            });

            const data = await response.json();

            // Wait brief moment for realistic HUD sequence to complete
            await new Promise(r => setTimeout(r, 900));
            clearInterval(stepInterval);

            if (data.success) {
                latestAnalysisData = data;
                renderAnalysisResult(data);
            } else {
                showToast(data.error || "Analysis failed.");
            }
        } catch (error) {
            console.error("Analysis request failed:", error);
            // Fallback to standard form submit if fetch fails
            form.submit();
            return;
        } finally {
            // Restore UI state
            if (textareaWrapper) textareaWrapper.classList.remove("scanning");
            if (scannerHud) scannerHud.style.display = "none";
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="bi bi-cpu me-2"></i> Analyze News Article`;
            }
        }
    });
}

/* --------------------------------------------------------------------------
   7. Render Result Panel with Circular Meter & Probability Bars
   -------------------------------------------------------------------------- */
function renderAnalysisResult(data) {
    const panel = document.getElementById("resultPanel");
    if (!panel) return;

    // Set class (real-verdict vs fake-verdict)
    panel.classList.remove("real-verdict", "fake-verdict");
    panel.classList.add(data.result_class === "real" ? "real-verdict" : "fake-verdict");

    // Verdict Header
    const iconEl = document.getElementById("resVerdictIcon");
    const nameEl = document.getElementById("resVerdictName");
    const descEl = document.getElementById("resVerdictDesc");

    if (iconEl) iconEl.innerHTML = data.result_class === "real" ? `<i class="bi bi-shield-check"></i>` : `<i class="bi bi-shield-x"></i>`;
    if (nameEl) nameEl.textContent = data.result_class === "real" ? "VERIFIED REAL NEWS" : "DECEPTIVE / FAKE NEWS";
    if (descEl) descEl.textContent = data.verdict_summary;

    // Circular Confidence Meter
    const meterPercent = document.getElementById("resMeterPercent");
    const meterCircle = document.getElementById("resMeterCircle");

    if (meterPercent) {
        animateValue(meterPercent, 0, data.confidence, 1200, 1, "", "%");
    }

    if (meterCircle) {
        const radius = 54;
        const circumference = 2 * Math.PI * radius;
        meterCircle.style.strokeDasharray = `${circumference} ${circumference}`;
        
        // Initial offset
        meterCircle.style.strokeDashoffset = circumference;
        
        setTimeout(() => {
            const offset = circumference - (data.confidence / 100) * circumference;
            meterCircle.style.strokeDashoffset = offset;
        }, 100);
    }

    // Split Probability Bar
    const splitReal = document.getElementById("resSplitReal");
    const splitFake = document.getElementById("resSplitFake");
    const textRealProb = document.getElementById("resTextRealProb");
    const textFakeProb = document.getElementById("resTextFakeProb");

    if (splitReal) splitReal.style.width = `${data.real_prob}%`;
    if (splitFake) splitFake.style.width = `${data.fake_prob}%`;
    if (textRealProb) textRealProb.textContent = `${data.real_prob}% Real`;
    if (textFakeProb) textFakeProb.textContent = `${data.fake_prob}% Fake`;

    // Telemetry stats
    const latencyEl = document.getElementById("resLatency");
    const wordsEl = document.getElementById("resWords");
    const stemmedEl = document.getElementById("resStemmed");

    if (latencyEl) latencyEl.textContent = `${data.latency}s`;
    if (wordsEl) wordsEl.textContent = data.word_count;
    if (stemmedEl) stemmedEl.textContent = data.cleaned_word_count;

    // Notable Keywords
    const keywordsBox = document.getElementById("resKeywords");
    if (keywordsBox) {
        keywordsBox.innerHTML = "";
        if (data.keywords && data.keywords.length > 0) {
            data.keywords.forEach(kw => {
                const tag = document.createElement("span");
                tag.className = "keyword-tag";
                tag.innerHTML = `<span>#${kw.word}</span> <span class="tag-weight">${kw.weight}</span>`;
                keywordsBox.appendChild(tag);
            });
        } else {
            keywordsBox.innerHTML = `<span class="text-muted small">No prominent vocabulary matches detected in 5,000 feature space.</span>`;
        }
    }

    panel.style.display = "block";
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* --------------------------------------------------------------------------
   8. Copy Forensic Report to Clipboard
   -------------------------------------------------------------------------- */
window.copyForensicReport = function() {
    if (!latestAnalysisData) {
        showToast("No active report to copy.");
        return;
    }

    const d = latestAnalysisData;
    const keywordsList = d.keywords ? d.keywords.map(k => k.word).join(", ") : "None";

    const report = 
`=========================================================
VERITAS AI - NEWS CREDIBILITY FORENSIC REPORT
=========================================================
Timestamp: ${new Date().toUTCString()}
Verdict: ${d.label.toUpperCase()} (${d.confidence}% Confidence)
Summary: ${d.verdict_summary}

PROBABILITY DISTRIBUTION:
- Real News Likelihood : ${d.real_prob}%
- Fake News Likelihood : ${d.fake_prob}%

MODEL INFERENCE TELEMETRY:
- Architecture         : Logistic Regression (L2 Regularized)
- Feature Space        : TF-IDF Vectorizer (5,000 Dimensions)
- Inference Latency    : ${d.latency}s
- Original Word Count  : ${d.word_count} words (${d.char_count} chars)
- Stemmed Word Count   : ${d.cleaned_word_count} tokens
- Key Linguistic Roots : ${keywordsList}

ARTICLE EXCERPT:
"${d.news_snippet}"
=========================================================
Verified by Veritas AI Fake News Detection Platform`;

    navigator.clipboard.writeText(report).then(() => {
        showToast("Forensic report copied to clipboard!");
    }).catch(() => {
        showToast("Failed to copy report to clipboard.");
    });
};

/* --------------------------------------------------------------------------
   9. Lightweight Toast Notification
   -------------------------------------------------------------------------- */
function showToast(msg) {
    let toast = document.getElementById("modernToast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "modernToast";
        toast.className = "modern-toast";
        document.body.appendChild(toast);
    }

    toast.innerHTML = `<i class="bi bi-info-circle-fill text-cyan"></i> <span>${msg}</span>`;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

