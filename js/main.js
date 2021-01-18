console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs);

const enableWebcamButton = document.getElementById('webcamButton');
const fileInput = document.getElementById('file-input');
const inputError = document.getElementById('input-error');
const video = document.getElementById('webcam');
const output = document.getElementById('output');
const traceContainer = document.getElementById('sticky-header');
const trace = document.getElementById('trace-data');

/******************
 * INPUTS
 *****************/

fileInput.addEventListener('change', function (event) {
    inputError.classList.add('hide');
    var file = this.files[0];
    if (!video.canPlayType(file.type)) {
        console.log('cannot play ' + file.type);
        inputError.innerHTML = 'The selected file type (' + file.type + ') cannot be played by your browser. Try transcoding it to a <a href="https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs">commonly used web codec</a>.';
        inputError.classList.remove('hide');
        return false;
    }

    var URL = window.URL || window.webkitURL;
    var fileUrl = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = fileUrl;
});

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start classification.
function enableCam(event) {

    video.removeEventListener('loadeddata', predictLoop);

    // Only continue if the model has finished loading.
    if (!model) {
        return;
    }

    // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictLoop);
    });
}

/********************
 * ROI
 *******************/

const roi = document.getElementById('roi');
const rx = document.getElementById('roi-left');
const ry = document.getElementById('roi-top');
const rs = document.getElementById('roi-size');

// const rw = document.getElementById('roi-width');
// const rh = document.getElementById('roi-height');

var updatePredictionTimeout;

function updatePrediction(delay) {
    delay = delay ?? 100;
    clearTimeout(updatePredictionTimeout);
    if (video.readyState >= 2) // video has data for at least one frame
        updatePredictionTimeout = setTimeout(predictOnce, delay);
}

function updateRoi() {
    let left = parseInt(rx.value);
    let top = parseInt(ry.value);
    let size = parseInt(rs.value);
    let roiStyle = window.getComputedStyle(roi);
    let border = roiStyle.borderWidth || roiStyle.borderTopWidth;
    border = Math.round(parseFloat(border));

    roi.style.left = (left - border) + 'px';
    roi.style.top = (top - border) + 'px';
    roi.style.width = size + 'px';
    roi.style.height = size + 'px';

    updatePrediction();
}

rx.addEventListener('input', updateRoi);
ry.addEventListener('input', updateRoi);
rs.addEventListener('input', updateRoi);

function resetRoi() {
    let width = video.videoWidth;
    let height = video.videoHeight;
    let side = Math.min(width, height);

    rx.value = Math.floor((width - side) / 2);
    ry.value = Math.floor((height - side) / 2);
    rs.value = side;

    updateRoi();
}

video.addEventListener('loadeddata', resetRoi);
video.addEventListener('loadeddata', () => {
    video.muted = true;
    video.volume = 0;
})

var dragOffset = undefined;

function dragstart(event) {
    event.dataTransfer.setData('application/node type', this);
    let style = window.getComputedStyle(event.target, null);

    let offsetX = (parseInt(style.getPropertyValue("left")) - event.clientX);
    let offsetY = (parseInt(style.getPropertyValue("top")) - event.clientY);

    dragOffset = [offsetX, offsetY];

    event.dataTransfer.setDragImage(new Image(), 0, 0);
    event.dataTransfer.effectAllowed = "move";
}

function dragover(event) {
    if (dragOffset) {
        let [offsetX, offsetY] = dragOffset;

        let size = parseInt(rs.value);
        let roiStyle = window.getComputedStyle(roi);
        let border = roiStyle.borderWidth || roiStyle.borderTopWidth;
        border = Math.round(parseFloat(border));

        let newX = Math.floor(event.clientX + parseInt(offsetX));
        let newY = Math.floor(event.clientY + parseInt(offsetY));

        let maxX = video.videoWidth - size + border;
        let maxY = video.videoHeight - size + border;

        newX = Math.max(-border, Math.min(newX, maxX));
        newY = Math.max(-border, Math.min(newY, maxY));

        roi.style.left = newX + 'px';
        roi.style.top = newY + 'px';

        rx.value = newX + border;
        ry.value = newY + border;

        event.preventDefault();
        return false;
    }
}

function drop(event) {
    dragOffset = undefined;
    event.preventDefault();
    return false;
}

let observer = new MutationObserver(function (mutations) {
    let width = video.videoWidth;
    let height = video.videoHeight;
    let maxSize = Math.min(width - rx.value, height - ry.value);
    let elem = mutations[0].target;

    let newSize = Math.min(elem.clientWidth, elem.clientHeight);
    newSize = Math.min(newSize, maxSize);

    elem.style.width = newSize + 'px';
    elem.style.height = newSize + 'px';
    rs.value = newSize;

    updatePrediction(30);
}).observe(roi, {
    attributes: true
});

roi.addEventListener('dragstart', dragstart);
document.body.addEventListener('dragover', dragover);
document.body.addEventListener('drop', drop);

/****************
 * PUPIL LOCATOR
 ***************/

const pupilXLocator = document.getElementById('pupil-x');
const pupilYLocator = document.getElementById('pupil-y');

const pupilXLabel = pupilXLocator.firstElementChild;
const pupilYLabel = pupilYLocator.firstElementChild;

function updatePupilLocator(x, y) {
    if (x < 0 || y < 0) {
        pupilXLocator.style.display = 'none';
        pupilYLocator.style.display = 'none';
    } else {
        pupilXLocator.style.left = x + 'px';
        pupilXLocator.style.height = video.videoHeight + 'px';
        pupilXLabel.textContent = `X=${x.toFixed(1)}`;

        pupilYLocator.style.width = video.videoWidth + 'px';
        pupilYLocator.style.top = y + 'px';
        pupilYLabel.textContent = `Y=${y.toFixed(1)}`;

        pupilXLocator.style.display = 'block';
        pupilYLocator.style.display = 'block';
    }
}

/**************
 * TRIGGERS
 *************/
var nTriggers = 2;
var triggers = new Array(nTriggers).fill(0);
var triggersToReset = [];

function setTrigger(triggerId) {
    triggers[triggerId] = 1;
}

function resetTrigger(triggerId) {
    triggers[triggerId] = 0;
}

function KeyTDownHandler(event) {
    var key = event.key || event.keyCode;
    if (key == "T" || key == "t") {
        document.removeEventListener('keydown', KeyTDownHandler);
        setTrigger(0);
    }
    return true;
}

function KeyYDownHandler(event) {
    var key = event.key || event.keyCode;
    if (key == "Y" || key == "y") {
        document.removeEventListener('keydown', KeyYDownHandler);
        setTrigger(1);
    }
    return true;
}

function KeyTUpHandler(event) {
    var key = event.key || event.keyCode;
    if (key == "T" || key == "t") {
        resetTrigger(0);
        document.addEventListener('keydown', KeyTDownHandler);
    }
    return true;
}

function KeyYUpHandler(event) {
    var key = event.key || event.keyCode;
    if (key == "Y" || key == "y") {
        resetTrigger(1);
        document.addEventListener('keydown', KeyYDownHandler);
    }
    return true;
}

document.addEventListener('keydown', KeyTDownHandler, false);
document.addEventListener('keydown', KeyYDownHandler, false);
document.addEventListener('keyup', KeyTUpHandler, false);
document.addEventListener('keyup', KeyYUpHandler, false);

function spikeTrigger(event) {
    const triggerId = parseInt(event.target.dataset.triggerId);
    triggers[triggerId] = 1;
    triggersToReset.push(triggerId);
}

document.getElementById('control-trigger-1').addEventListener('click', spikeTrigger);
document.getElementById('control-trigger-2').addEventListener('click', spikeTrigger);


/****************
 * MODEL
 ***************/

var model = undefined;
const modelUrl = 'models/meye-segmentation_i128_s4_c1_f16_g1_a-relu/model.json'
tf.loadLayersModel(modelUrl).then(function (loadedModel) {
    model = loadedModel;
    tf.tidy(() => {
        model.predict(tf.zeros([1, 128, 128, 1]))[0].data().then(() => {
            document.getElementById('loading').style.display = 'none';
        });
    });
    video.addEventListener('play', predictLoop);
    video.addEventListener('seeked', predictOnce);
});

var period = 0;
var timeoutHandler = null;
var threshold = 0.5;

const rgb = tf.tensor1d([0.2989, 0.587, 0.114]);
const _255 = tf.scalar(255);

function predictFrame() {
    let timestamp = new Date();
    let timecode = video.currentTime;

    let x = parseInt(rx.value);
    let y = parseInt(ry.value);
    let s = parseInt(rs.value);

    // Now let's start classifying a frame in the stream.
    const frame = tf.browser.fromPixels(video, 3)
        .slice([y, x], [s, s])
        .resizeBilinear([128, 128])
        .mul(rgb).sum(2)
        .expandDims(0).expandDims(-1)
        .toFloat().div(_255);

    let [maps, eb] = model.predict(frame);
    let [pupil, glint] = maps.squeeze().split(2, 2);
    let [eye, blink] = eb.squeeze().split(2);

    pupil = tf.cast(pupil.greaterEqual(threshold), 'float32').squeeze();

    let m00 = pupil.sum();
    let m01 = tf.range(0, 128).expandDims(0).mul(pupil).sum();
    let m10 = tf.range(0, 128).expandDims(1).mul(pupil).sum();

    let pupilX = m01.div(m00).mul(s / 128).add(x).data();
    let pupilY = m10.div(m00).mul(s / 128).add(y).data();

    let pupilArea = m00.data();
    let blinkProb = blink.data();

    let drawn = tf.browser.toPixels(pupil, output);
    return [drawn, timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY];
}

function predictOnce() {
    if (!model) return;
    let outs = tf.tidy(predictFrame);
    Promise.all(outs).then(outs => {
        let [drawn, timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY] = outs;
        pupilArea = pupilArea[0];
        pupilX = (pupilArea > 0) ? pupilX[0] : -1;
        pupilY = (pupilArea > 0) ? pupilY[0] : -1;
        updatePupilLocator(pupilX, pupilY);
    })
}

function predictLoop() {
    if (!model) return;
    let outs = tf.tidy(predictFrame);
    Promise.all(outs).then(outs => {

        let [drawn, timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY] = outs;
        pupilArea = pupilArea[0];
        blinkProb = blinkProb[0];
        pupilX = (pupilArea > 0) ? pupilX[0] : -1;
        pupilY = (pupilArea > 0) ? pupilY[0] : -1;
        updatePupilLocator(pupilX, pupilY);

        // pause prediction when video is paused
        if (!video.paused)
            if (period == 0)
                window.requestAnimationFrame(predictLoop);
            else
                timeoutHandler = setTimeout(predictLoop, period);

        // add samples to csv and chart data
        let sample = [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY, triggers];
        addSample(sample);

        // reset triggers
        triggersToReset.forEach(resetTrigger);
        triggersToReset = [];

        // update FPS counter
        framesThisSecond++;
    });
}

/***************
 * CONTROLS
 **************/

const controlClear = document.getElementById('control-clear');
const controlTrigger = document.getElementById('control-trigger');
const controlThreshold = document.getElementById('control-thr');
const controlThresholdPreview = document.getElementById('control-thr-preview');
const controlPeriod = document.getElementById('control-period');
const controlLivePlot = document.getElementById('control-liveplot');
const controlExportCsv = document.getElementById('control-export-csv');
const controlWindow = document.getElementById('control-window');
const controlWindowEnable = document.getElementById('control-window-enable');

function clearData() {
    samples.length = 0; // clear array
    while (trace.firstChild) trace.removeChild(trace.lastChild);
    while (chartContainer.firstChild) chartContainer.removeChild(chartContainer.lastChild);
    chartData = null;
    chart = null;
    ctx = output.getContext("2d");
    ctx.clearRect(0, 0, output.width, output.height);
    // video.pause();
    // video.currentTime = 0;
}

function setThreshold(event) {
    threshold = event.target.value / 100;
    controlThresholdPreview.textContent = threshold.toFixed(2);
    updatePrediction(5);
}

function setPeriod(event) {
    period = event.target.value;
}

function strTimestamp() {
    let now = new Date();
    let y = String(now.getFullYear());
    let M = String(now.getMonth() + 1).padStart(2, '0');
    let d = String(now.getDate()).padStart(2, '0');
    let h = String(now.getHours()).padStart(2, '0');
    let m = String(now.getMinutes()).padStart(2, '0');
    return `${y}${M}${d}-${h}${m}`;
}

function exportCsv() {
    let header = ['timestamp', 'timecode', 'pupil-area', 'blink', 'pupil-x', 'pupil-y', 'trigger1', 'trigger2']
    let csvHeader = ["data:text/csv;charset=utf-8," + header.join(',')];
    let csvLines = samples
        .map(r => [r[0].toISOString()].concat(r.slice(1)))
        .map(r => r.join(','));
    let csvContent = csvHeader.concat(csvLines).join('\r\n');
    let csvUri = encodeURI(csvContent);

    let usingWebcam = video.srcObject != null;
    let filename;

    if (usingWebcam)
        filename = strTimestamp() + ".csv";
    else if (fileInput.value)
        filename = fileInput.value.split(/[\\/]/).pop().replace(/\.[^/.]+$/, ".csv");
    else
        filename = "export.csv";

    let a = document.createElement("a");
    a.download = filename;
    a.href = csvUri;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function togglePlotWindow(event) {
    if (event.target.checked) {
        controlWindow.disabled = false;
        chartWindow = controlWindow.value;
    } else {
        controlWindow.disabled = true;
        chartWindow = Infinity;
    }
}

function resizePlotWindow(event) {
    if (event.target.disabled == false)
        chartWindow = event.target.value;
}

controlClear.addEventListener('click', clearData);
controlThreshold.addEventListener('input', setThreshold);
controlPeriod.addEventListener('change', setPeriod);
controlExportCsv.addEventListener('click', exportCsv);
controlWindowEnable.addEventListener('change', togglePlotWindow);
controlWindow.addEventListener('change', resizePlotWindow);

/***************
 * OUTPUTS
 **************/

var chartWindow = Infinity;
var chartContainer = document.getElementById('chart-container');
var chartOptions = null;
var chartData = null;
var chart = null;


// Load the Visualization API and the corechart package.
google.charts.load('current', {
    'packages': ['corechart']
});

// Set a callback to run when the Google Visualization API is loaded.
// google.charts.setOnLoadCallback(initChart);

function initChart() {
    // Create the data table.
    chartData = new google.visualization.DataTable();
    chartData.addColumn('number', 'timecode');
    chartData.addColumn('number', 'pupil-area');
    chartData.addColumn('number', 'blink');
    // chartData.addColumn('number', 'pupil-x');
    // chartData.addColumn('number', 'pupil-y');

    let series = {
        0: {
            targetAxisIndex: 0
        },
        1: {
            targetAxisIndex: 1
        }
    }

    for (var i = 0; i < nTriggers; ++i) {
        series[i + 2] = {
            targetAxisIndex: 1
        };
        chartData.addColumn('number', 'trigger' + (i + 1));
    }

    // Set chart options
    chartOptions = {
        // curveType: 'function',
        'chartArea': {
            'width': '85%',
            'height': '80%'
        },
        series: series,
        vAxes: {
            // Adds titles to each axis.
            0: {
                title: 'Pupil Area (px²)',
                minValue: 0,
                textPosition: 'in'
            },
            1: {
                title: 'Blink (%)',
                format: 'percent',
                minValue: 0,
                maxValue: 1,
                textPosition: 'in'
            }
        },
        hAxis: {
            title: 'time (s)',
            textPosition: 'in'
        },
        height: 225,
        legend: {
            position: 'top'
        },
    };

    // Instantiate and draw our chart, passing in some options.
    chart = new google.visualization.LineChart(chartContainer);
    chart.draw(chartData, chartOptions);
}

var samples = [];

function addSample(sample) {

    let [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY, triggers] = sample;

    let flatSample = sample.slice(0, -1).concat(triggers);
    samples.push(flatSample);

    sampleRow = document.createElement('tr');
    sampleRow.innerHTML = (
        '<td>' + timestamp.toISOString() + '</td>' +
        '<td>' + timecode.toFixed(3) + '</td>' +
        '<td>' + pupilArea.toFixed(2) + '</td>' +
        '<td>' + blinkProb.toFixed(1) + '</td>' +
        '<td>' + pupilX.toFixed(1) + '</td>' +
        '<td>' + pupilY.toFixed(1) + '</td>' +
        triggers.map(t => '<td>' + t + '</td>').join(''));

    trace.appendChild(sampleRow);
    traceContainer.scrollTop = traceContainer.scrollHeight;

    if (!chart) initChart();
    let chartSample = [timecode, pupilArea, blinkProb].concat(triggers);
    chartData.addRow(chartSample);

    let N = chartData.getNumberOfRows();
    if (N > chartWindow)
        chartData.removeRows(0, N - chartWindow);

    if (controlLivePlot.checked)
        chart.draw(chartData, chartOptions);
}

/*****
 * FPS
 *****/

var avgFps = 0.0;
var alpha = 0.5;
var framesThisSecond = 0;

const fpsPreview = document.getElementById('fps-preview');
const fpsMeter = document.getElementById('fps-meter');

function computeFps() {
    avgFps = alpha * avgFps + (1.0 - alpha) * framesThisSecond;
    fpsPreview.textContent = avgFps.toFixed(1);
    fpsMeter.value = avgFps;
    framesThisSecond = 0;
}

setInterval(computeFps, 1000);
