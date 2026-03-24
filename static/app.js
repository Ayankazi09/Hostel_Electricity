document.addEventListener('DOMContentLoaded', () => {
    initDashboard();

    // SPA Navigation Router
    const titles = {
        'page-overview': { title: 'Dashboard Overview', sub: 'Real-time electricity consumption insights' },
        'page-analysis': { title: 'Deep Analysis', sub: 'Model accuracy and correlation scatters' },
        'page-prediction': { title: 'Prediction Studio', sub: 'Interactive simulations and forecasting engines' },
        'page-models': { title: 'Model Hub', sub: 'View detailed accuracy metrics and download trained model artifacts' },
        'page-rawdata': { title: 'Historic Data Explorer', sub: 'Datatables view of SQL records' }
    };
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.page-section').forEach(p => p.style.display = 'none');
            
            link.classList.add('active');
            const targetId = link.getAttribute('data-target');
            document.getElementById(targetId).style.display = 'block';
            
            document.getElementById('page-title').textContent = titles[targetId].title;
            document.getElementById('page-subtitle').textContent = titles[targetId].sub;
        });
    });

    // Slider Listeners
    const updateVal = (sliderId, textId) => {
        document.getElementById(sliderId).addEventListener('input', (e) => {
            document.getElementById(textId).innerText = e.target.value;
        });
    };
    updateVal('sim-students', 'students-val');
    updateVal('sim-cap', 'cap-val');
    updateVal('sim-temp', 'temp-val');
    updateVal('sim-warn', 'warn-val');
    updateVal('sim-rate', 'rate-val');

    // Form submission
    document.getElementById('prediction-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        runPrediction();
    });
});

let avgData = {};

async function initDashboard() {
    try {
        // Fetch KPI
        const kpiRes = await fetch('/api/kpi');
        const kpiData = await kpiRes.json();
        
        if (!kpiData.error) {
            animateValue("kpi-records", 0, kpiData.totalRecords, 1500);
            animateValue("kpi-hostels", 0, kpiData.totalHostels, 1500);
            animateValue("kpi-avg", 0, Math.round(kpiData.avgConsumption), 1500);
            animateValue("kpi-max", 0, Math.round(kpiData.maxConsumption), 1500);
        }

        // Fetch Chart Data
        const chartRes = await fetch('/api/chart-data');
        const chartData = await chartRes.json();
        
        if (!chartData.error) {
            renderTrendChart(chartData.monthly_trend.labels, chartData.monthly_trend.values);
            
            // Populate Hostels Select
            const select = document.getElementById('sim-hostel');
            const fSelect = document.getElementById('forecast-hostel');
            select.innerHTML = '';
            fSelect.innerHTML = '';
            chartData.hostels.forEach(h => {
                select.appendChild(new Option(`Hostel ${h}`, h));
                fSelect.appendChild(new Option(`Hostel ${h}`, h));
            });
            avgData = chartData.avg_by_hostel;
            
            // Forecast event listener
            const genBtn = document.getElementById('gen-forecast-btn');
            if(genBtn) {
                genBtn.addEventListener('click', runForecast);
            }
        }

        // Fetch deep analytics
        const perfRes = await fetch('/api/model-performance');
        const perfData = await perfRes.json();
        if (perfData.metrics && perfData.metrics.length > 0) {
            renderBarChart('rmseChart', perfData.metrics.map(m => m.Model), perfData.metrics.map(m => m.RMSE), '#a855f7');
        }
        if (perfData.importances && perfData.importances.length > 0) {
            renderHorizontalBarChart('featureChart', perfData.importances.map(m => m.feature.replace('_', ' ')), perfData.importances.map(m => m.importance), '#10b981');
        }

        const scatterRes = await fetch('/api/scatter-data');
        const scatterData = await scatterRes.json();
        if(!scatterData.error) {
            renderScatterChart('scatterStudentChart', scatterData.students, scatterData.electricity, 'Students', 'Consumption');
            renderScatterChart('scatterTempChart', scatterData.temperature, scatterData.electricity, 'Temperature (°C)', 'Consumption');
        }

        // Fetch deep supplementary analysis charts
        const extRes = await fetch('/api/analysis-extra');
        const extData = await extRes.json();
        if(extData.monthly_avg) {
            renderBarChart('monthlyAvgChart', ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], extData.monthly_avg, '#a855f7');
            renderLineAreaChart('distChart', extData.dist_labels, extData.dist_counts, '#06b6d4');
            renderDoughnutChart('seasonChart', ['Exam', 'Normal', 'Vacation'], extData.season_split, ['#f43f5e', '#a855f7', '#fbbf24']);
            renderScatterChart('utilChart', extData.utilization_x, extData.utilization_y, 'Utilization Rate', 'Consumption');
        }

        // Initialize DataTable
        const rawRes = await fetch('/api/raw-data');
        const rawData = await rawRes.json();
        if(rawData.data) {
            $('#historyTable').DataTable({
                data: rawData.data,
                columns: [
                    { data: 'hostel_id' },
                    { data: 'year' },
                    { data: 'month' },
                    { data: 'num_students' },
                    { data: 'hostel_capacity' },
                    { data: 'avg_temperature' },
                    { data: 'exam_month', render: data => data ? 'Yes' : 'No' },
                    { data: 'vacation_month', render: data => data ? 'Yes' : 'No' },
                    { 
                        data: 'electricity_kwh', 
                        render: data => Number(data).toFixed(2), 
                        defaultContent: "0.00" 
                    }
                ],
                pageLength: 10,
                responsive: true
            });
        }

        // Fetch Model Metrics Full Update
        try {
            const metricsRes = await fetch('/api/model-metrics-full');
            const metricsData = await metricsRes.json();
            if (metricsData.metrics && metricsData.metrics.length > 0) {
                $('#metricsTable').DataTable({
                    data: metricsData.metrics,
                    columns: [
                        { data: 'Model' },
                        { data: 'RMSE', render: $.fn.dataTable.render.number(',', '.', 2) },
                        { data: 'MAPE', render: $.fn.dataTable.render.number(',', '.', 4) },
                        { data: 'R2', render: $.fn.dataTable.render.number(',', '.', 4) },
                        { data: 'CV_RMSE', render: $.fn.dataTable.render.number(',', '.', 2) },
                        { data: 'Score', render: $.fn.dataTable.render.number(',', '.', 4) }
                    ],
                    pageLength: 5,
                    responsive: true,
                    searching: false,
                    lengthChange: false
                });
            }
        } catch (e) {
            console.error("Failed to load model metrics table", e);
        }

        // Initialize Past Logs Table with Dummy Data
        try {
            const dummyLogs = [
                { date: "2026-01-15", accuracy: "87.5%" },
                { date: "2026-02-10", accuracy: "88.2%" },
                { date: "2026-03-05", accuracy: "89.1%" },
                { date: "2026-03-24", accuracy: "90.5%" }
            ];
            $('#pastLogsTable').DataTable({
                data: dummyLogs,
                columns: [
                    { data: 'date' },
                    { data: 'accuracy' }
                ],
                pageLength: 5,
                responsive: true,
                searching: false,
                lengthChange: false
            });
        } catch (e) {
            console.error("Failed to load past logs table", e);
        }

        // Fetch Models List
        try {
            const modelsRes = await fetch('/api/models-list');
            const modelsData = await modelsRes.json();
            const container = document.getElementById('models-list-container');
            container.innerHTML = '';
            
            if (modelsData.models && modelsData.models.length > 0) {
                modelsData.models.forEach(model => {
                    const card = document.createElement('div');
                    card.className = 'kpi-card glass';
                    card.style.minWidth = '250px';
                    card.style.flex = '1';
                    card.innerHTML = `
                        <i class="fa-solid fa-file-export kpi-icon" style="color: #a855f7;"></i>
                        <div class="kpi-info" style="margin-bottom: 1rem;">
                            <h3 style="word-break: break-all; margin-bottom: 0.2rem;">${model.filename}</h3>
                            <p style="font-size: 0.9rem; margin-bottom: 0.2rem;">Size: ${model.size_kb} KB</p>
                            ${model.modified ? `<p style="font-size: 0.8rem; color: #8b949e;">Changed: ${model.modified}</p>` : ''}
                        </div>
                        <a href="/api/download-model/${model.filename}" target="_blank" class="cta-button" style="text-decoration: none; display: inline-block; width: 100%; text-align: center; padding: 0.5rem;">
                            <i class="fa-solid fa-download"></i> Download
                        </a>
                    `;
                    container.appendChild(card);
                });
            } else {
                container.innerHTML = '<p>No model files found.</p>';
            }
        } catch(e) {
             console.error("Failed to load models list", e);
             document.getElementById('models-list-container').innerHTML = '<p>Error loading model files.</p>';
        }

    } catch (err) {
        console.error("Failed to load dashboard data.", err);
    }
}

function renderTrendChart(labels, values) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    
    // Gradient fill
    let gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(88, 166, 255, 0.5)');   
    gradient.addColorStop(1, 'rgba(88, 166, 255, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Global Avg Consumption',
                data: values,
                borderColor: '#58a6ff',
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#58a6ff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { 
                    backgroundColor: 'rgba(22, 27, 34, 0.9)',
                    titleColor: '#8b949e',
                    bodyColor: '#f0f6fc',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } }
            }
        }
    });
}

function animateValue(id, start, end, duration) {
    let obj = document.getElementById(id);
    if (!obj) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        let val = Math.floor(progress * (end - start) + start);
        // format with commas
        obj.innerHTML = val.toLocaleString() + (id.includes('avg') || id.includes('max') ? ' <small>kWh</small>' : '');
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

async function runPrediction() {
    const btn = document.querySelector('.cta-button');
    btn.innerHTML = 'Connecting AI... <i class="fa-solid fa-spinner fa-spin"></i>';
    
    // Check if hostel value matches logic
    const hostelIdVal = document.getElementById('sim-hostel').value;
    const hostelInt = parseInt(hostelIdVal);
    
    const payload = {
        hostel_id: isNaN(hostelInt) ? 1 : hostelInt,
        month: parseInt(document.getElementById('sim-month').value),
        year: parseInt(document.getElementById('sim-year').value),
        num_students: parseInt(document.getElementById('sim-students').value),
        hostel_capacity: parseInt(document.getElementById('sim-cap').value),
        avg_temperature: parseFloat(document.getElementById('sim-temp').value),
        exam_month: document.getElementById('sim-exam').checked,
        vacation_month: document.getElementById('sim-vacation').checked
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if (data.prediction) {
            const predText = document.getElementById('pred-result');
            animateValueText(predText, 0, data.prediction, 1000);
            
            const baseAvg = avgData[payload.hostel_id] || Math.max(0.1, data.prediction);
            const diff = data.prediction - baseAvg;
            const diffPct = Math.abs((diff / baseAvg) * 100);
            const userThreshold = parseFloat(document.getElementById('sim-warn').value);
            
            let context = `This is <strong style="color: ${diff > 0 ? '#f43f5e' : '#10b981'}">${diffPct.toFixed(1)}% ${diff > 0 ? 'Higher' : 'Lower'}</strong> than the typical average for Hostel ${payload.hostel_id}.`;
            if (payload.num_students > payload.hostel_capacity) {
                context += `<br><span style="color:#fbbf24"><i class="fa-solid fa-triangle-exclamation"></i> Warning: Operating over capacity.</span>`;
            }
            if (diff > 0 && diffPct >= userThreshold) {
                context += `<br><span style="color:#f43f5e"><i class="fa-solid fa-bell"></i> <strong>ALERT:</strong> Custom Threshold (${userThreshold}%) Breach!</span>`;
            }
            document.getElementById('pred-context').innerHTML = context;
            
            // Bar animation
            let fillPct = (data.prediction / (baseAvg * 2)) * 100;
            fillPct = Math.min(100, Math.max(0, fillPct));
            document.getElementById('bar-fill').style.width = fillPct + '%';
            if(diff > 0 && diffPct >= userThreshold) {
                 document.getElementById('bar-fill').style.background = 'linear-gradient(90deg, #fbbf24, #f43f5e)';
                 predText.style.color = '#f43f5e';
            } else {
                 document.getElementById('bar-fill').style.background = 'linear-gradient(90deg, #10b981, #059669)';
                 predText.style.color = '#10b981';
            }

            // Calculate Financial Cost
            const rate = parseFloat(document.getElementById('sim-rate').value);
            const cost = data.prediction * rate;
            document.getElementById('cost-box').style.display = 'block';
            document.getElementById('pred-cost').innerHTML = `$${cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

            // Show and setup export button for PDF
            const exportBtn = document.getElementById('export-btn');
            exportBtn.style.display = 'inline-block';
            exportBtn.onclick = () => {
                const element = document.querySelector('.result-card');
                const btnBackup = exportBtn.style.display;
                exportBtn.style.display = 'none'; // hide button while generating pdf
                const opt = {
                    margin: 0.5,
                    filename: `Hostel_${payload.hostel_id}_AI_Report.pdf`,
                    image: { type: 'jpeg', quality: 0.98 },
                    html2canvas: { scale: 2 },
                    jsPDF: { unit: 'in', format: 'letter', orientation: 'landscape' }
                };
                html2pdf().set(opt).from(element).save().then(() => {
                    exportBtn.style.display = btnBackup;
                });
            };

        } else {
            console.error(data);
        }
    } catch(err) {
        console.error(err);
    } finally {
        btn.innerHTML = 'Run AI Simulator <i class="fa-solid fa-play"></i>';
    }
}

function animateValueText(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        let val = progress * (end - start) + start;
        obj.innerHTML = val.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' <small>kWh</small>';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Analytics Helpers
function renderBarChart(canvasId, labels, data, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'RMSE Value',
                data: data,
                backgroundColor: color + 'aa',
                borderColor: color,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#8b949e' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } }
            }
        }
    });
}

function renderHorizontalBarChart(canvasId, labels, data, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: data,
                backgroundColor: color + 'aa',
                borderColor: color,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } },
                y: { ticks: { color: '#f0f6fc', font: {size: 10} } }
            }
        }
    });
}

function renderScatterChart(canvasId, xData, yData, xLabel, yLabel) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const points = xData.map((x, i) => ({ x: x, y: yData[i] }));
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Point',
                data: points,
                backgroundColor: 'rgba(168, 85, 247, 0.4)',
                borderColor: '#a855f7',
                borderWidth: 1,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    title: { display: true, text: xLabel, color: '#a1a1aa' },
                    grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a1a1aa' } 
                },
                y: { 
                    title: { display: true, text: yLabel, color: '#a1a1aa' },
                    grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a1a1aa' } 
                }
            }
        }
    });
}

let forecastChartInstance = null;
async function runForecast() {
    const btn = document.getElementById('gen-forecast-btn');
    const hostelId = document.getElementById('forecast-hostel').value;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    
    try {
        const res = await fetch(`/api/forecast/${hostelId}`);
        const data = await res.json();
        
        if (data.predictions) {
            const ctx = document.getElementById('forecastChart').getContext('2d');
            if (forecastChartInstance) { forecastChartInstance.destroy(); }
            
            let gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(210, 153, 34, 0.4)');   
            gradient.addColorStop(1, 'rgba(210, 153, 34, 0)');

            forecastChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.months,
                    datasets: [{
                        label: `Predicted 2026 Consumption (kWh)`,
                        data: data.predictions,
                        borderColor: '#fbbf24',
                        backgroundColor: gradient,
                        borderWidth: 3,
                        pointBackgroundColor: '#09090b',
                        pointBorderColor: '#a855f7',
                        pointBorderWidth: 2,
                        pointRadius: 5,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } }
                    }
                }
            });
        }
    } catch(err) {
        console.error(err);
    } finally {
        btn.innerHTML = 'Generate Outlook';
    }
}

function renderLineAreaChart(canvasId, labels, data, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    let gradient = ctx.createLinearGradient(0, 0, 0, 250);
    gradient.addColorStop(0, color + '66');   
    gradient.addColorStop(1, color + '00');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                borderColor: color,
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', maxTicksLimit: 8 } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } }
            }
        }
    });
}

function renderDoughnutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { 
                legend: { position: 'right', labels: { color: '#8b949e', font: {size: 11} } }
            },
            cutout: '70%'
        }
    });
}
