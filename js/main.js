console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs);

const enableWebcamButton = document.getElementById('webcamButton');
const fileInput = document.getElementById('file-input')
const video = document.getElementById('webcam');
const output = document.getElementById('output');
const traceContainer = document.getElementById('sticky-header');
const trace = document.getElementById('trace-data');

video.addEventListener('loadeddata', setRoi);

/******************
 * INPUTS
 *****************/

fileInput.addEventListener('change', function () {
    var file = this.files[0];
    if (!video.canPlayType(file.type)) {
        console.log('cannot play ' + file.type);
        return;
    }

    var URL = window.URL || window.webkitURL;
    var fileUrl = URL.createObjectURL(file);
    console.log(fileUrl);
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

    video.removeEventListener('loadeddata', predictFrame);

    // Only continue if the model has finished loading.
    if (!model) {
        return;
    }

    // Hide the button once clicked.
    event.target.classList.add('removed');

    // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictFrame);
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


function updateRoi() {
    var left = parseInt(rx.value);
    var top = parseInt(ry.value);
    var size = parseInt(rs.value);

    roi.style.left = left + 'px';
    roi.style.top = top + 'px';
    roi.style.width = size + 'px';
    roi.style.height = size + 'px';
}

rx.addEventListener('change', updateRoi);
ry.addEventListener('change', updateRoi);
rs.addEventListener('change', updateRoi);

var dragOffset = undefined;

function dragstart(event) {

    var style = window.getComputedStyle(event.target, null);

    var offsetX = (parseInt(style.getPropertyValue("left")) - event.clientX);
    var offsetY = (parseInt(style.getPropertyValue("top")) - event.clientY);

    dragOffset = [offsetX, offsetY];
    console.log(dragOffset);

    event.dataTransfer.setDragImage(new Image(), 0, 0);
    event.dataTransfer.effectAllowed = "move";
}

function drag(event) {
    if (dragOffset) {
        let [offsetX, offsetY] = dragOffset;

        let size = parseInt(rs.value);
        let maxX = video.videoWidth - size;
        let maxY = video.videoHeight - size;

        let newX = Math.floor(event.clientX + parseInt(offsetX));
        let newY = Math.floor(event.clientY + parseInt(offsetY));

        newX = Math.max(0, Math.min(newX, maxX));
        newY = Math.max(0, Math.min(newY, maxY));

        roi.style.left = newX + 'px';
        roi.style.top = newY + 'px';

        rx.value = newX;
        ry.value = newY;
    }

    event.preventDefault();
    return false;
}

function dragover(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
}

function drop(event) {
    dragOffset = undefined;
    event.preventDefault();
    return false;
}

let observer = new MutationObserver(function (mutations) {
    let width = video.videoWidth;
    let height = video.videoHeight;
    let maxSize = Math.min(width, height);
    let elem = mutations[0].target;

    let newSize = Math.min(elem.offsetWidth, elem.offsetHeight);
    newSize = Math.min(newSize, maxSize);

    elem.style.width = newSize + 'px';
    elem.style.height = newSize + 'px';
    rs.value = newSize;
}).observe(roi, {
    attributes: true
});

roi.addEventListener('dragstart', dragstart);
document.body.addEventListener('drag', drag);
document.body.addEventListener('dragover', dragover);
document.body.addEventListener('drop', drop);


function setRoi() {
    let width = this.videoWidth;
    let height = this.videoHeight;
    let side = Math.min(width, height);

    rx.value = Math.floor((width - side) / 2);
    ry.value = Math.floor((height - side) / 2);
    rs.value = side;
    // rw.value = side;
    // rh.value = side;
    updateRoi();
}


/****************
 * MODEL
 ***************/

var model = undefined;
const modelUrl = 'models/meye-segmentation_i128_s4_c1_f16_g1_a-relu/model.json'
tf.loadLayersModel(modelUrl).then(function (loadedModel) {
    model = loadedModel;
    video.addEventListener('loadeddata', setRoi);
    video.addEventListener('play', predictFrame);
});

var period = 0;
var timeoutHandler = null;
var nTriggers = 2;
var triggers = new Array(nTriggers).fill(0);
var threshold = 0.5;

const rgb = tf.tensor1d([0.2989, 0.587, 0.114]);
const _255 = tf.scalar(255);

function predictFrame() {

    let timestamp = new Date();
    let timecode = video.currentTime;
    let pupilArea = null;
    let blinkProb = null;

    var x = parseInt(rx.value);
    var y = parseInt(ry.value);
    var s = parseInt(rs.value);

    tf.tidy(() => {
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

        pupil = tf.cast(pupil.greaterEqual(threshold), 'float32');
        tf.browser.toPixels(pupil.squeeze(), output);

        // this is synchronous, mh..
        pupilArea = pupil.sum().dataSync()[0];
        blinkProb = blink.dataSync()[0];
        if (!video.paused)
            if (period == 0)
                window.requestAnimationFrame(predictFrame);
            else
                timeoutHandler = setTimeout(predictFrame, period);

    });

    let sample = [timestamp, timecode, pupilArea, blinkProb, triggers];
    addSample(sample);
    triggers.fill(0);
    framesThisSecond++;
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
}

function setPeriod(event) {
    period = event.target.value;
    console.log(period);
}

function exportCsv() {
    let header = ['timestamp', 'timecode', 'pupil-area', 'blink', 'trigger']
    let csvHeader = ["data:text/csv;charset=utf-8," + header.join(',')];
    let csvLines = samples
        .map(r => [r[0].toISOString()].concat(r.slice(1)))
        .map(r => r.join(','));
    let csvContent = csvHeader.concat(csvLines).join('\r\n');
    let csvUri = encodeURI(csvContent);

    let a = document.createElement("a");
    a.download = "export.csv";
    a.href = csvUri;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

controlClear.addEventListener('click', clearData);
controlThreshold.addEventListener('input', setThreshold);
controlPeriod.addEventListener('change', setPeriod);
controlExportCsv.addEventListener('click', exportCsv);

/***************
 * OUTPUTS
 **************/

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

    console.log(series);

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

    let [timestamp, timecode, pupilArea, blinkProb, triggers] = sample;
    let flatSample = sample.slice(0, -1).concat(sample.slice(-1)[0]);

    samples.push(flatSample);

    sampleRow = document.createElement('tr');
    sampleRow.innerHTML = (
        '<td>' + timestamp.toISOString() + '</td>' +
        '<td>' + timecode.toFixed(3) + '</td>' +
        '<td>' + pupilArea.toFixed(2) + '</td>' +
        '<td>' + blinkProb.toFixed(1) + '</td>' +
        triggers.map(t => '<td>' + t + '</td>').join(''));

    trace.appendChild(sampleRow);
    // console.log(traceContainer.scrollTop, traceContainer.scrollHeight);
    traceContainer.scrollTop = traceContainer.scrollHeight;

    if (!chart) initChart();
    chartData.addRow(flatSample.slice(1))

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
