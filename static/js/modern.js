/**
 * VERITAS AI - Modern Fake News Detection System Core Engine
 * Interactive Motions, Particle Matrix, HUD Scanner, and Live Analysis
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initParticles();
    initCounters();
    initScrollReveal();
    initNavbarScroll();
    initTextareaMetrics();
    initPrefillFromQuery();
    initFormAnalyzer();
});

function initScrollReveal() {
    const els = document.querySelectorAll(".reveal-on-scroll");
    if (!els.length) return;
    const obs = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });
    els.forEach(el => obs.observe(el));
}

function initPrefillFromQuery() {
    const textarea = document.getElementById("newsTextarea");
    if (!textarea) return;
    const params = new URLSearchParams(window.location.search);
    const prefill = params.get("prefill");
    const sample = params.get("sample");
    if (prefill && prefill.trim()) {
        textarea.value = prefill.trim();
        textarea.dispatchEvent(new Event("input"));
        textarea.focus();
    } else if (sample && SAMPLE_ARTICLES[sample]) {
        textarea.value = SAMPLE_ARTICLES[sample];
        textarea.dispatchEvent(new Event("input"));
        textarea.focus();
    }
}

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
   1. Subtle Editorial Background Canvas (Monochrome Ink Dots & Hairlines)
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
    const count = Math.min(Math.floor((width * height) / 32000), 28);

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.25,
            vy: (Math.random() - 0.5) * 0.25,
            radius: Math.random() * 1.4 + 0.6
        });
    }

    function render() {
        ctx.clearRect(0, 0, width, height);
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        const inkRgb = isDark ? "243, 241, 235" : "17, 19, 24";
        const accentRgb = isDark ? "59, 130, 246" : "29, 78, 216";

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
            ctx.fillStyle = i % 3 === 0 ? `rgba(${accentRgb}, 0.28)` : `rgba(${inkRgb}, 0.16)`;
            ctx.fill();

            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 145) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(${accentRgb}, ${0.11 * (1 - dist / 145)})`;
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(render);
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

    if (activeVerifyPollTimer) {
        clearTimeout(activeVerifyPollTimer);
        activeVerifyPollTimer = null;
    }

    const resultPanel = document.getElementById("resultPanel");
    if (resultPanel) {
        resultPanel.style.display = "none";
    }

    const extSection = document.getElementById("externalVerificationSection");
    if (extSection) {
        extSection.style.display = "none";
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
let activeVerifyPollTimer = null;

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function safeHttpUrl(url) {
    if (!url || typeof url !== "string") return "#";
    const trimmed = url.trim();
    if (trimmed.startsWith("https://") || trimmed.startsWith("http://")) {
        return trimmed;
    }
    return "#";
}

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
    const extToggle = document.getElementById("externalVerifyToggle");
    const extSection = document.getElementById("externalVerificationSection");
    const extProgress = document.getElementById("extVerifyProgress");
    const extResults = document.getElementById("extVerifyResults");

    form.addEventListener("submit", async (e) => {
        const text = textarea ? textarea.value.trim() : "";
        if (!text) {
            e.preventDefault();
            showToast("Please enter or paste a news article first!");
            return;
        }

        e.preventDefault();

        if (activeVerifyPollTimer) {
            clearTimeout(activeVerifyPollTimer);
            activeVerifyPollTimer = null;
        }

        const runExternal = extToggle ? extToggle.checked : false;

        // 1. Enter Scanning State
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span> Analyzing...`;
        }
        if (textareaWrapper) textareaWrapper.classList.add("scanning");
        if (resultPanel) resultPanel.style.display = "none";
        if (extSection) extSection.style.display = "none";
        if (scannerHud) scannerHud.style.display = "block";

        const steps = runExternal
            ? [
                { text: "01/05 – Processing submitted article...", pct: 33 },
                { text: "02/05 – Running NLP preprocessing...", pct: 66 },
                { text: "03/05 – Generating ML prediction...", pct: 100 }
              ]
            : [
                { text: "01/05 – Processing submitted article...", pct: 33 },
                { text: "02/05 – Running NLP preprocessing...", pct: 66 },
                { text: "03/05 – Generating ML prediction...", pct: 100 }
              ];

        let currentStep = 0;
        if (hudStepText) hudStepText.textContent = steps[0].text;
        if (hudFill) hudFill.style.width = `${steps[0].pct}%`;
        currentStep = 1;

        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                if (hudStepText) hudStepText.textContent = steps[currentStep].text;
                if (hudFill) hudFill.style.width = `${steps[currentStep].pct}%`;
                currentStep++;
            }
        }, 90);

        try {
            const endpoint = runExternal ? "/api/verify" : "/api/predict";
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ news: text })
            });

            const data = await response.json();

            await new Promise(r => setTimeout(r, 280));
            clearInterval(stepInterval);

            if (!data.success) {
                showToast(data.error || "Analysis failed.");
                return;
            }

            const mlData = runExternal ? data.ml_analysis : data;
            latestAnalysisData = mlData;
            renderAnalysisResult(mlData);

            if (runExternal && data.job_id && extSection) {
                extSection.style.display = "block";
                if (extProgress) extProgress.style.display = "block";
                if (extResults) extResults.style.display = "none";
                updateExternalProgress(60, "04/05 – Extracting claims and searching external sources...");
                pollVerificationStatus(data.job_id);
            }
        } catch (error) {
            console.error("Analysis request failed:", error);
            form.submit();
            return;
        } finally {
            if (textareaWrapper) textareaWrapper.classList.remove("scanning");
            if (scannerHud) scannerHud.style.display = "none";
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="bi bi-cpu me-2"></i> Analyze News Article`;
            }
        }
    });
}

function updateExternalProgress(pct, stageText) {
    const fill = document.getElementById("extProgressFill");
    const pctEl = document.getElementById("extProgressPct");
    const stageEl = document.getElementById("extProgressStage");
    if (fill) fill.style.width = `${pct}%`;
    if (pctEl) pctEl.textContent = `${pct}%`;
    if (stageEl && stageText) stageEl.textContent = stageText;
}

async function pollVerificationStatus(jobId) {
    try {
        const resp = await fetch(`/api/verify/status/${encodeURIComponent(jobId)}`);
        const job = await resp.json();

        if (!job.success) {
            showToast(job.error || "External verification error.");
            return;
        }

        updateExternalProgress(job.progress || 15, job.stage || "Verifying claims...");

        if (job.status === "completed" && job.result) {
            renderExternalVerification(job.result);
            return;
        }
        if (job.status === "failed") {
            const prog = document.getElementById("extVerifyProgress");
            if (prog) {
                prog.innerHTML = `<div class="text-crimson small"><i class="bi bi-exclamation-triangle me-2"></i>External verification failed: ${escapeHtml(job.error || "Unknown error")}</div>`;
            }
            return;
        }

        activeVerifyPollTimer = setTimeout(() => pollVerificationStatus(jobId), 1200);
    } catch (err) {
        console.error("Polling error:", err);
        activeVerifyPollTimer = setTimeout(() => pollVerificationStatus(jobId), 2000);
    }
}

function renderExternalVerification(payload) {
    const prog = document.getElementById("extVerifyProgress");
    const resultsEl = document.getElementById("extVerifyResults");
    if (prog) prog.style.display = "none";
    if (resultsEl) resultsEl.style.display = "block";

    const final = payload.final_assessment || {};
    const ml = payload.ml_analysis || {};
    const claims = payload.claims || [];
    const sources = payload.sources || [];

    // 1. Final Assessment Banner
    const banner = document.getElementById("extFinalBanner");
    if (banner) {
        banner.classList.remove("final-real", "final-fake", "final-mixed", "final-unverified");
        banner.classList.add(`final-${final.final_class || "unverified"}`);
    }

    const finalLabelEl = document.getElementById("extFinalLabel");
    const finalSummaryEl = document.getElementById("extFinalSummary");
    const evBadgeEl = document.getElementById("extEvidenceBadge");

    if (finalLabelEl) finalLabelEl.textContent = final.final_label || "UNVERIFIED";
    if (finalSummaryEl) finalSummaryEl.textContent = final.summary || "";

    if (evBadgeEl) {
        const verdict = final.external_verdict || "INSUFFICIENT EVIDENCE";
        evBadgeEl.textContent = verdict;
        evBadgeEl.className = "badge rounded-pill px-3 py-2 mt-1 ";
        if (verdict === "SUPPORTED") evBadgeEl.className += "bg-success-subtle text-emerald";
        else if (verdict === "CONTRADICTED") evBadgeEl.className += "bg-danger-subtle text-crimson";
        else if (verdict === "MIXED") evBadgeEl.className += "bg-warning-subtle text-warning";
        else evBadgeEl.className += "bg-secondary-subtle text-muted";
    }

    // Disagreement Alert
    const disAlert = document.getElementById("extDisagreementAlert");
    const disText = document.getElementById("extDisagreementText");
    if (disAlert && disText) {
        if (final.ml_external_disagreement && final.disagreement_note) {
            disText.textContent = final.disagreement_note;
            disAlert.style.display = "block";
        } else {
            disAlert.style.display = "none";
        }
    }

    // Independent Comparison Pills
    const cmpMl = document.getElementById("extCompareMl");
    const cmpExt = document.getElementById("extCompareExternal");
    const nliName = document.getElementById("extNliModelName");
    if (cmpMl) cmpMl.textContent = `${ml.label || "N/A"} (${ml.confidence || 0}%)`;
    if (cmpExt) cmpExt.textContent = final.external_verdict || "INSUFFICIENT EVIDENCE";
    if (nliName && payload.nli_model) nliName.textContent = payload.nli_model.split("/").pop();

    // 2. Extracted Factual Claims & Evidence
    const claimsList = document.getElementById("extClaimsList");
    const claimCountBadge = document.getElementById("extClaimCountBadge");
    if (claimCountBadge) claimCountBadge.textContent = `${claims.length} Claim${claims.length === 1 ? "" : "s"}`;

    if (claimsList) {
        claimsList.innerHTML = "";
        if (!claims.length) {
            claimsList.innerHTML = `<div class="ext-claim-card text-muted small">No verifiable factual claims could be extracted from the submitted text.</div>`;
        } else {
            claims.forEach(c => {
                let badgeCls = "bg-secondary-subtle text-muted";
                if (c.label === "SUPPORTED") badgeCls = "bg-success-subtle text-emerald";
                else if (c.label === "CONTRADICTED") badgeCls = "bg-danger-subtle text-crimson";
                else if (c.label === "MIXED") badgeCls = "bg-warning-subtle text-warning";

                const renderClaimEvidenceItem = (ev, kind) => {
                    const safeUrl = safeHttpUrl(ev.url);
                    const pubDate = ev.published_date ? escapeHtml(ev.published_date) : "Date unavailable";
                    let borderCls = "";
                    let labelHtml = "";
                    let scoreHtml = "";
                    let bodyHtml = "";

                    if (kind === "supporting") {
                        borderCls = "ev-supporting";
                        labelHtml = `<strong class="text-emerald small"><i class="bi bi-check-circle me-1"></i>SUPPORTING &bull; ${escapeHtml(ev.publisher)} (${escapeHtml(ev.domain)})</strong>`;
                        scoreHtml = `<span class="small text-muted font-monospace">NLI Entailment: ${(ev.entailment_score * 100).toFixed(0)}%</span>`;
                        bodyHtml = `<div class="text-muted mt-1">"${escapeHtml(ev.evidence_passage)}"</div>`;
                    } else if (kind === "contradicting") {
                        borderCls = "ev-contradicting";
                        labelHtml = `<strong class="text-crimson small"><i class="bi bi-x-circle me-1"></i>CONTRADICTING &bull; ${escapeHtml(ev.publisher)} (${escapeHtml(ev.domain)})</strong>`;
                        scoreHtml = `<span class="small text-muted font-monospace">NLI Contradiction: ${(ev.contradiction_score * 100).toFixed(0)}%</span>`;
                        bodyHtml = `<div class="text-muted mt-1">"${escapeHtml(ev.evidence_passage)}"</div>`;
                    } else if (kind === "neutral") {
                        borderCls = "ev-neutral";
                        labelHtml = `<strong class="text-muted small"><i class="bi bi-dash-circle me-1"></i>INCONCLUSIVE / NEUTRAL &bull; ${escapeHtml(ev.publisher)} (${escapeHtml(ev.domain)})</strong>`;
                        scoreHtml = `<span class="small text-muted font-monospace">NLI Neutral: ${(ev.neutral_score * 100).toFixed(0)}%</span>`;
                        bodyHtml = `<div class="text-muted mt-1">"${escapeHtml(ev.evidence_passage)}"</div>`;
                    } else {
                        borderCls = "ev-unretrieved";
                        labelHtml = `<strong class="text-warning small"><i class="bi bi-lock me-1"></i>CONTENT NOT RETRIEVED &bull; ${escapeHtml(ev.publisher)} (${escapeHtml(ev.domain)})</strong>`;
                        scoreHtml = `<span class="small text-warning font-monospace">${escapeHtml(ev.status_note || "Inaccessible")}</span>`;
                        bodyHtml = `<div class="text-muted small fst-italic mt-1">Article content could not be retrieved, so it was not used as verification proof.</div>`;
                    }

                    return `
                        <div class="ext-evidence-item ${borderCls} mt-2">
                            <div class="d-flex flex-wrap justify-content-between align-items-center gap-1 mb-1">
                                ${labelHtml}
                                ${scoreHtml}
                            </div>
                            <div class="small fw-semibold text-white">${escapeHtml(ev.title)}</div>
                            <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mt-1">
                                <span class="small text-muted"><i class="bi bi-calendar3 me-1"></i>${pubDate}</span>
                                <a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer" class="small text-cyan text-decoration-none">
                                    Open Original Article <i class="bi bi-box-arrow-up-right"></i>
                                </a>
                            </div>
                            ${bodyHtml}
                        </div>
                    `;
                };

                const suppHtml = (c.supporting_evidence || []).map(ev => renderClaimEvidenceItem(ev, "supporting")).join("");
                const contraHtml = (c.contradicting_evidence || []).map(ev => renderClaimEvidenceItem(ev, "contradicting")).join("");
                const neutHtml = (c.neutral_evidence || []).map(ev => renderClaimEvidenceItem(ev, "neutral")).join("");
                const unretHtml = (c.unretrieved_sources || []).map(ev => renderClaimEvidenceItem(ev, "unretrieved")).join("");

                const card = document.createElement("div");
                card.className = "ext-claim-card";
                card.innerHTML = `
                    <div class="d-flex flex-wrap justify-content-between align-items-start gap-2 mb-2">
                        <div class="d-flex flex-wrap align-items-center gap-2">
                            <span class="badge bg-dark border border-secondary text-cyan small">CLAIM ${escapeHtml(c.claim_id)}</span>
                            <span class="small text-muted font-monospace">Query: "${escapeHtml(c.search_query)}"</span>
                        </div>
                        <span class="badge ${badgeCls} rounded-pill px-3 py-1">${escapeHtml(c.label)}</span>
                    </div>
                    <p class="text-white fw-semibold mb-1" style="font-size: 14.5px;">"${escapeHtml(c.claim_text)}"</p>
                    <p class="text-muted small mb-2">${escapeHtml(c.rationale)}</p>
                    ${suppHtml}
                    ${contraHtml}
                    ${neutHtml}
                    ${unretHtml}
                `;
                claimsList.appendChild(card);
            });
        }
    }

    // 3. Retrieved External Source Cards (strictly real retrieved sources)
    const sourcesList = document.getElementById("extSourcesList");
    const sourceCountBadge = document.getElementById("extSourceCountBadge");
    if (sourceCountBadge) sourceCountBadge.textContent = `${sources.length} Source${sources.length === 1 ? "" : "s"}`;

    if (sourcesList) {
        sourcesList.innerHTML = "";
        if (!sources.length) {
            sourcesList.innerHTML = `<div class="col-12"><div class="ext-source-card text-muted small">No external news sources were retrieved for the extracted claims.</div></div>`;
        } else {
            sources.forEach(src => {
                let stanceBadge = `<span class="badge bg-secondary-subtle text-muted">NEUTRAL</span>`;
                if (src.stance === "SUPPORTING") stanceBadge = `<span class="badge bg-success-subtle text-emerald">SUPPORTING</span>`;
                else if (src.stance === "CONTRADICTING") stanceBadge = `<span class="badge bg-danger-subtle text-crimson">CONTRADICTING</span>`;
                else if (src.stance === "UNRETRIEVED" || !src.accessible) stanceBadge = `<span class="badge bg-warning-subtle text-warning">UNRETRIEVED</span>`;

                const safeUrl = safeHttpUrl(src.url);
                const dateHtml = src.published_date
                    ? `<span class="small text-muted"><i class="bi bi-calendar3 me-1"></i>${escapeHtml(src.published_date)}</span>`
                    : `<span class="small text-muted">Date unavailable</span>`;
                const claimTags = (src.claim_ids || []).map(id => `Claim ${escapeHtml(id)}`).join(", ");
                const passageHtml = src.accessible && src.evidence_passage
                    ? `"${escapeHtml(src.evidence_passage)}"`
                    : `<span class="fst-italic">Content could not be retrieved (${escapeHtml(src.status_note || "Inaccessible")}). Not used as verification proof.</span>`;
                const alignHtml = src.accessible
                    ? `Align: ${(src.entailment_score * 100).toFixed(0)}% Sup / ${(src.contradiction_score * 100).toFixed(0)}% Con`
                    : `Content unavailable`;

                const col = document.createElement("div");
                col.className = "col-md-6";
                col.innerHTML = `
                    <div class="ext-source-card">
                        <div>
                            <div class="d-flex justify-content-between align-items-center gap-2 mb-2">
                                <span class="badge bg-dark border border-secondary text-cyan small">${escapeHtml(src.publisher)} &bull; ${escapeHtml(src.domain)}</span>
                                ${stanceBadge}
                            </div>
                            <h6 class="text-white fw-bold mb-2" style="font-size: 14.5px; line-height: 1.4;">${escapeHtml(src.title)}</h6>
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                ${dateHtml}
                                <span class="small text-muted font-monospace">${claimTags ? `${claimTags} &bull; ` : ""}${src.accessible ? "Full text extracted" : escapeHtml(src.status_note || "Inaccessible")}</span>
                            </div>
                            <div class="ext-source-passage mb-3">
                                ${passageHtml}
                            </div>
                        </div>
                        <div class="d-flex justify-content-between align-items-center pt-2 border-top border-secondary-subtle">
                            <span class="small text-muted font-monospace">${alignHtml}</span>
                            <a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer" class="btn-cyber-outline btn-pill-sm">
                                <span>Open Source</span> <i class="bi bi-box-arrow-up-right"></i>
                            </a>
                        </div>
                    </div>
                `;
                sourcesList.appendChild(col);
            });
        }
    }
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

