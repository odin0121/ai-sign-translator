/**
 * AI Sign Language Translator - Dashboard & Telemetry Analytics
 * Telemetry, Latency Benchmarking & Post-Hoc XAI Joint Attribution
 */

document.addEventListener('DOMContentLoaded', () => {
    const rawData = window.DASHBOARD_DATA || {};
    const topSigns = rawData.topSigns || [];
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    // -------------------------------------------------------------------------
    // Theme Manager (Dark / Bright Mode)
    // -------------------------------------------------------------------------
    function initTheme() {
        const savedTheme = localStorage.getItem('app-theme') || 'dark';
        applyTheme(savedTheme);

        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                const isCurrentlyDark = document.body.classList.contains('dark-theme');
                const newTheme = isCurrentlyDark ? 'light' : 'dark';
                applyTheme(newTheme);
                localStorage.setItem('app-theme', newTheme);
            });
        }
    }

    function applyTheme(theme) {
        const iconSun = themeToggleBtn ? themeToggleBtn.querySelector('.icon-sun') : null;
        const iconMoon = themeToggleBtn ? themeToggleBtn.querySelector('.icon-moon') : null;
        const themeLabel = themeToggleBtn ? themeToggleBtn.querySelector('.theme-label') : null;

        if (theme === 'light') {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            if (iconSun) iconSun.style.display = 'none';
            if (iconMoon) iconMoon.style.display = 'inline-block';
            if (themeLabel) themeLabel.innerText = 'Dark';
        } else {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            if (iconSun) iconSun.style.display = 'inline-block';
            if (iconMoon) iconMoon.style.display = 'none';
            if (themeLabel) themeLabel.innerText = 'Bright';
        }
    }

    initTheme();

    function getCsrfToken() {
        const input = document.querySelector('[name=csrfmiddlewaretoken]');
        return input ? input.value : '';
    }

    // -------------------------------------------------------------------------
    // 1. Sign Frequency Distribution Bar Chart with Gradient
    // -------------------------------------------------------------------------
    const freqCanvas = document.getElementById('frequencyChart');
    if (freqCanvas && topSigns.length > 0) {
        const ctx = freqCanvas.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 240);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.7)');
        gradient.addColorStop(1, 'rgba(37, 99, 235, 0.1)');

        const labels = topSigns.map(s => s.predicted_token);
        const counts = topSigns.map(s => s.count);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Occurrences',
                    data: counts,
                    backgroundColor: gradient,
                    borderColor: '#38bdf8',
                    borderWidth: 1.5,
                    borderRadius: 8,
                    hoverBackgroundColor: '#38bdf8'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#38bdf8',
                        bodyColor: '#f8fafc',
                        borderColor: 'rgba(56, 189, 248, 0.3)',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11, weight: 'bold' } },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#64748b', stepSize: 2, font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(255, 255, 255, 0.04)' }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // 2. Hand Joint XAI Saliency Radar Chart with Glowing Polygon
    // -------------------------------------------------------------------------
    const radarCanvas = document.getElementById('saliencyRadarChart');
    if (radarCanvas) {
        const rCtx = radarCanvas.getContext('2d');
        const radarGradient = rCtx.createRadialGradient(150, 150, 20, 150, 150, 120);
        radarGradient.addColorStop(0, 'rgba(52, 211, 153, 0.4)');
        radarGradient.addColorStop(1, 'rgba(52, 211, 153, 0.05)');

        new Chart(rCtx, {
            type: 'radar',
            data: {
                labels: ['Thumb TIP', 'Index TIP', 'Middle TIP', 'Ring TIP', 'Pinky TIP', 'Knuckles (MCP)', 'Wrist Origin'],
                datasets: [
                    {
                        label: 'Illustrative attribution (example)',
                        data: [0.95, 0.89, 0.78, 0.65, 0.82, 0.58, 0.44],
                        backgroundColor: radarGradient,
                        borderColor: '#34d399',
                        pointBackgroundColor: '#34d399',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#34d399',
                        borderWidth: 2
                    },
                    {
                        label: 'Uniform Baseline Reference',
                        data: [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
                        backgroundColor: 'rgba(100, 116, 139, 0.08)',
                        borderColor: 'rgba(148, 163, 184, 0.4)',
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.06)' },
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        pointLabels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } },
                        ticks: { display: false, max: 1.0, min: 0.0 }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } }
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#34d399',
                        borderColor: 'rgba(52, 211, 153, 0.3)',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // 3. Table Search Filter
    // -------------------------------------------------------------------------
    const searchInput = document.getElementById('table-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const rows = document.querySelectorAll('#logs-table-body tr.log-row');
            rows.forEach(row => {
                const text = row.innerText.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            });
        });
    }

    // -------------------------------------------------------------------------
    // 4. Live Telemetry Auto-Refresher
    // -------------------------------------------------------------------------
    async function pollTelemetry() {
        try {
            const res = await fetch('/api/telemetry/');
            if (!res.ok) return;
            const data = await res.json();
            const countEl = document.getElementById('total-logs-count');
            if (countEl) countEl.innerText = data.total_translations;
        } catch (e) {
            // Background polling silent catch
        }
    }
    setInterval(pollTelemetry, 4000);

    // Refresh button
    const refreshBtn = document.getElementById('refresh-logs-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            window.location.reload();
        });
    }

    // Clear logs
    const clearBtn = document.getElementById('clear-logs-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to clear all translation telemetry audit records?')) return;
            try {
                const res = await fetch('/api/clear_logs/', {
                    method: 'POST',
                    headers: { 'X-CSRFToken': getCsrfToken() }
                });
                if (res.ok) window.location.reload();
            } catch (err) {
                alert('Failed to clear logs: ' + err.message);
            }
        });
    }
});
