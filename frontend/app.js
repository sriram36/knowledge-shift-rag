document.addEventListener('DOMContentLoaded', () => {
    // Routing Logic
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');

    function navigateTo(pageId) {
        navItems.forEach(item => {
            if (item.dataset.page === pageId) item.classList.add('active');
            else item.classList.remove('active');
        });
        
        pages.forEach(page => {
            if (page.id === `${pageId}-page`) page.classList.add('active');
            else page.classList.remove('active');
        });

        if (pageId === 'dashboard') {
            loadDashboard();
        }
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(item.dataset.page);
        });
    });

    // Page 1: Ask Question
    const askForm = document.getElementById('ask-form');
    const askBtn = document.getElementById('ask-btn');
    const askLoading = document.getElementById('ask-loading');
    const askResults = document.getElementById('ask-results');

    if (askForm) {
        askForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = document.getElementById('ask-question').value.trim();
            const mode = document.getElementById('ask-mode').value;
            
            if (!question) return;

            askBtn.disabled = true;
            askLoading.style.display = 'inline-block';
            askResults.innerHTML = '';

            const startTime = performance.now();
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, mode })
                });
                const data = await response.json();
                const latency = ((performance.now() - startTime) / 1000).toFixed(2);
                renderAskResults(data, latency, askResults);
            } catch (err) {
                askResults.innerHTML = `<div class="info-block danger"><p>Error: ${err.message}</p></div>`;
            } finally {
                askBtn.disabled = false;
                askLoading.style.display = 'none';
            }
        });
    }

    function renderAskResults(data, latency, container) {
        let html = '';
        
        // Final Answer
        const isInsufficient = !data.answer_text || data.answer === 'E' || data.answer_text.includes("Insufficient");
        if (isInsufficient) {
            html += `<div class="info-block danger"><h4>Answer</h4><p>Insufficient evidence in the knowledge base.</p></div>`;
        } else {
            html += `<div class="info-block success"><h4>Answer</h4><p>${data.answer_text}</p></div>`;
        }

        // Repair logic if applicable
        if (data.mode === 'knowledge_repair' && data.repair) {
            const statusBadge = data.repair.repair_successful 
                ? '<span class="badge success">Repaired</span>' 
                : '<span class="badge danger">Repair Failed</span>';
            html += `<div class="info-block repair"><h4>Repair <small>${statusBadge}</small></h4><p>${data.repair.reasoning}</p></div>`;
        } else if (data.mode === 'knowledge_repair') {
            html += `<div class="info-block repair"><h4>Repair</h4><p><span class="badge info">Not required</span></p></div>`;
        }

        // Critique
        if (data.critique) {
            let badgeClass = data.critique.status === 'SUPPORTED' ? 'success' : 
                             data.critique.status === 'UNSUPPORTED' ? 'danger' : 'warning';
            html += `<div class="info-block critique"><h4>Critique</h4><p><span class="badge ${badgeClass}">${data.critique.status}</span></p><p style="margin-top:0.5rem; font-size:0.9rem">${data.critique.reason || ''}</p></div>`;
        }

        // Evidence
        if (data.sources && data.sources.length > 0) {
            let evidenceHtml = data.sources.map((s, i) => `
                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-highlight);">
                    <strong>[Source ${i+1}] ${s.subject || 'Document'} (ID: ${s.chunk_id})</strong>
                    <p style="margin-top: 0.25rem; font-size: 0.9rem; color: var(--text-secondary);">${s.text}</p>
                </div>
            `).join('');
            html += `<div class="info-block evidence"><h4>Retrieved Evidence</h4>${evidenceHtml}</div>`;
        }

        html += `<div class="info-block latency"><h4>Latency</h4><p>${latency} seconds</p></div>`;
        container.innerHTML = html;
    }


    // Page 2: Compare Methods
    const compareForm = document.getElementById('compare-form');
    const compareBtn = document.getElementById('compare-btn');
    const compareLoading = document.getElementById('compare-loading');

    if (compareForm) {
        compareForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = document.getElementById('compare-question').value.trim();
            if (!question) return;

            compareBtn.disabled = true;
            compareLoading.style.display = 'inline-block';
            
            ['vanilla', 'self_critique', 'knowledge_repair'].forEach(mode => {
                document.getElementById(`compare-${mode}-res`).innerHTML = '<p class="text-secondary">Running...</p>';
            });

            try {
                await Promise.all([
                    runCompareMode(question, 'vanilla'),
                    runCompareMode(question, 'self_critique'),
                    runCompareMode(question, 'knowledge_repair')
                ]);
            } finally {
                compareBtn.disabled = false;
                compareLoading.style.display = 'none';
            }
        });
    }

    async function runCompareMode(question, mode) {
        const startTime = performance.now();
        const container = document.getElementById(`compare-${mode}-res`);
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, mode })
            });
            const data = await response.json();
            const latency = ((performance.now() - startTime) / 1000).toFixed(2);
            renderCompareResult(data, latency, container);
        } catch (err) {
            container.innerHTML = `<p style="color:var(--status-red)">Error: ${err.message}</p>`;
        }
    }

    function renderCompareResult(data, latency, container) {
        let html = '';
        
        const ansText = (!data.answer_text || data.answer === 'E') ? 'Insufficient evidence' : data.answer_text;
        html += `<div style="margin-bottom:1rem"><strong>Answer:</strong><p style="color:var(--text-secondary)">${ansText}</p></div>`;

        if (data.critique) {
            let badgeClass = data.critique.status === 'SUPPORTED' ? 'success' : 
                             data.critique.status === 'UNSUPPORTED' ? 'danger' : 'warning';
            html += `<div style="margin-bottom:1rem"><strong>Critique:</strong> <span class="badge ${badgeClass}">${data.critique.status}</span></div>`;
        }

        if (data.mode === 'knowledge_repair') {
            const trigger = data.repair ? 'Triggered' : 'Not triggered';
            html += `<div style="margin-bottom:1rem"><strong>Repair:</strong> <span class="badge info">${trigger}</span></div>`;
        }

        html += `<div style="margin-bottom:1rem"><strong>Latency:</strong> <span style="color:var(--text-secondary)">${latency}s</span></div>`;
        container.innerHTML = html;
    }


    // Page 3: Dashboard
    async function loadDashboard() {
        const tbody = document.getElementById('dashboard-tbody');
        if (!tbody) return;

        try {
            tbody.innerHTML = '<tr><td colspan="4">Loading data...</td></tr>';
            const res = await fetch('/api/evaluation/results');
            const data = await res.json();

            if (data.error) {
                tbody.innerHTML = `<tr><td colspan="4" style="color:var(--status-red)">${data.error}</td></tr>`;
                return;
            }

            let html = '';
            const systems = [
                { id: 'vanilla_rag', name: 'Vanilla RAG' },
                { id: 'self_critique_rag', name: 'Self-Critique RAG' },
                { id: 'knowledge_repair_rag', name: 'Knowledge-Repair RAG' }
            ];

            const sysDataMap = data.systems || {};

            systems.forEach(sys => {
                const sysData = sysDataMap[sys.id];
                if (sysData) {
                    html += `
                        <tr>
                            <td><strong>${sys.name}</strong></td>
                            <td>${(sysData.accuracy * 100).toFixed(1)}%</td>
                            <td>${sysData.avg_latency ? sysData.avg_latency.toFixed(2) : 'N/A'}s</td>
                            <td>${sysData.total || 100}</td>
                        </tr>
                    `;
                }
            });

            tbody.innerHTML = html;

            // Fill top metrics from KR if available
            if (sysDataMap.knowledge_repair_rag) {
                const kr = sysDataMap.knowledge_repair_rag;
                document.getElementById('metric-acc').textContent = `${(kr.accuracy * 100).toFixed(1)}%`;
                
                const repTotal = kr.repairs_triggered || 0;
                // Wait, repairs_successful was used earlier but in JSON it's repairs_fixed and repairs_broke.
                // If the backend doesn't explicitly expose repairs_successful, we'll use repairs_fixed if available
                // Wait, earlier I saw in compare script it prints "Repairs successful", but maybe it's saved differently.
                const repSuccess = (kr.repairs_fixed !== undefined) ? kr.repairs_fixed : (kr.repairs_successful || 0);
                document.getElementById('metric-rep-trig').textContent = repTotal;
                
                if (repTotal > 0) {
                    document.getElementById('metric-rep-succ').textContent = `${((repSuccess/repTotal)*100).toFixed(0)}%`;
                } else {
                    document.getElementById('metric-rep-succ').textContent = 'N/A';
                }
            }

        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="4" style="color:var(--status-red)">Failed to fetch results</td></tr>`;
        }
    }

});