console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs);

const webcamButton = document.getElementById('webcamButton');
const fileInput = document.getElementById('file-input');
const inputError = document.getElementById('input-error');
const video = document.getElementById('webcam');
const input = document.getElementById('net-input');
const output = document.getElementById('output');
const traceContainer = document.getElementById('sticky-header');
const trace = document.getElementById('trace-data');

const demoButtons = document.querySelectorAll('.demo-button');

/******************
 * INPUTS
 *****************/
fileInput.addEventListener('change', function (event) {
    inputError.classList.add('hide');
    var file = this.files[0];
    if (!video.canPlayType(file.type)) {
        console.log('cannot play ' + file.type);
        inputError.innerHTML = 'The selected file type (' + file.type + ') cannot be played by your browser.<br>Try transcoding it to a <a href="https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs" target="_blank">commonly used web codec</a>, e.g., using an <a href="https://www.freeconvert.com/avi-to-mp4" target="_blank">online video converter</a>.';
        inputError.classList.remove('hide');
        return false;
    }

    hideRoi();
    clearPreview();
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
    webcamButton.addEventListener('click', toggleCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

var videoStream = null;
// Enable the live webcam view and start classification.
function toggleCam(event) {

    if (videoStream) { // cam is active
        video.pause();
        videoStream.getTracks().forEach(t => {
            t.stop();
        });
        webcamButton.value = "Enable Webcam";
        videoStream = null;
        return;
    }

    // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true,
        audio: false
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        clearPreview();
        videoStream = stream;
        video.srcObject = stream;
        video.play();

        webcamButton.value = "Disable Webcam";
    });
}

function loadDemo(event) {
    video.pause();
    hideRoi();

    let demoData = event.target.dataset;

    video.removeEventListener('loadedmetadata', resetRoi);
    video.addEventListener('loadedmetadata', event => {
        // reactivate handler after the demo video has loaded
        video.addEventListener('loadedmetadata', resetRoi);
        return true;
    }, {
        once: true
    });

    video.addEventListener('canplay', event => {
        setRoi();
        showRoi();
        return true;
    }, {
        once: true
    });

    video.addEventListener('ended', event => {
        controlPlotUpdate.dispatchEvent(new Event('click'));
        controlTableUpdate.dispatchEvent(new Event('click'));
    }, {
        once: true
    });

    video.src = demoData.src;
    clearData();

    rx.value = parseInt(demoData.rx);
    ry.value = parseInt(demoData.ry);
    rs.value = parseInt(demoData.rs);
    rt.checked = parseInt(demoData.rt);

    controlPeriod.value = parseInt(demoData.period);
    controlThresholdPreview.value = parseFloat(demoData.thr);
    controlMorphology.checked = parseInt(demoData.morph);
    controlPlotAutoUpdate.checked = false;
    controlPlotWindowEnable.checked = false;
    controlTableAutoUpdate.checked = false;

    controlPeriod.dispatchEvent(new Event('change'));
    controlThresholdPreview.dispatchEvent(new Event('input'));
    controlMorphology.dispatchEvent(new Event('change'));
    controlPlotAutoUpdate.dispatchEvent(new Event('change'));
    controlPlotWindowEnable.dispatchEvent(new Event('change'));
    controlTableAutoUpdate.dispatchEvent(new Event('change'));
}

demoButtons.forEach(element => {
    element.addEventListener('click', loadDemo);
});


/********************
 * ROI
 *******************/
const roi = document.getElementById('roi');
const rx = document.getElementById('roi-left');
const ry = document.getElementById('roi-top');
const rs = document.getElementById('roi-size');
const rt = document.getElementById('roi-track');

const roiDragger = document.getElementById('roi-dragger');
const roiResizer = document.getElementById('roi-resizer');

function hideRoi() {
    roi.classList.add('hide');
    pupilXLocator.classList.add('hide');
    pupilYLocator.classList.add('hide');
    pupilXLabel.classList.add('hide');
    pupilYLabel.classList.add('hide');
}

function showRoi() {
    roi.classList.remove('hide');
    pupilXLocator.classList.remove('hide');
    pupilYLocator.classList.remove('hide');
    pupilXLabel.classList.remove('hide');
    pupilYLabel.classList.remove('hide');
}

var updatePredictionTimeout;

function updatePrediction(delay) {
    delay = delay ?? 100;
    clearTimeout(updatePredictionTimeout);
    if (video.readyState >= 2) // video has data for at least one frame
        updatePredictionTimeout = setTimeout(predictOnce, delay);
}

function setRoi() {
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
}

function updateRoi() {
    setRoi();
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

video.addEventListener('loadedmetadata', showRoi);
video.addEventListener('loadedmetadata', resetRoi);
video.addEventListener('loadeddata', () => {
    video.muted = true;
    video.volume = 0;
})
video.addEventListener('canplaythrough', () => {
    updatePrediction();
});

var dragOffset = undefined;

function dragstart(event) {
    event.dataTransfer.setData('application/node type', this);
    let style = window.getComputedStyle(roi, null);

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

        let maxX = video.videoWidth - size - border;
        let maxY = video.videoHeight - size - border;

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

var resizeSize = undefined;

function resizestart(event) {
    event.dataTransfer.setData('application/node type', this);
    let style = window.getComputedStyle(roi, null);

    let offsetX = (parseInt(style.getPropertyValue("width")) - event.clientX);
    let offsetY = (parseInt(style.getPropertyValue("height")) - event.clientY);

    resizeSize = [offsetX, offsetY];

    event.dataTransfer.setDragImage(new Image(), 0, 0);
    event.dataTransfer.effectAllowed = "move";
}


function resizeover(event) {
    if (resizeSize) {
        let [offsetX, offsetY] = resizeSize;

        let width = video.videoWidth;
        let height = video.videoHeight;
        let maxSize = Math.min(width - rx.value, height - ry.value);

        let newW = Math.floor(event.clientX + parseInt(offsetX));
        let newH = Math.floor(event.clientY + parseInt(offsetY));

        let newSize = Math.min(Math.min(newW, newH), maxSize);

        roi.style.width = newSize + 'px';
        roi.style.height = newSize + 'px';

        rs.value = newSize;

        event.preventDefault();
        return false;
    }
}

function drop(event) {
    dragOffset = undefined;
    resizeSize = undefined;
    event.preventDefault();
    return false;
}

let observer = new MutationObserver(() => {
    updatePrediction(30);
}).observe(roi, {
    attributes: true
});

roiDragger.addEventListener('dragstart', dragstart);
document.body.addEventListener('dragover', dragover);

roiResizer.addEventListener('dragstart', resizestart);
document.body.addEventListener('dragover', resizeover);

document.body.addEventListener('drop', drop); // in common

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
var triggerButtons = {};
var triggerColors = ['#FF9900', '#109618', '#990099', '#0099C6'];
var activeTriggers = {q: 0, w: 1, e: 2, r: 3};
var nTriggers = Object.keys(activeTriggers).length;
var triggers = new Array(nTriggers).fill(0);
var triggersToReset = [];

function setTrigger(triggerId) {
    triggers[triggerId] = 1;
    triggerButtons[triggerId].style.backgroundColor = triggerColors[triggerId];
}

function resetTrigger(triggerId) {
    triggers[triggerId] = 0;
    triggerButtons[triggerId].style.backgroundColor = 'inherit';
}

function KeyDownHandler(event) {
    var key = event.key || event.keyCode;
    if (key in activeTriggers)
        setTrigger(activeTriggers[key]);

    return true;
}

function KeyUpHandler(event) {
    var key = event.key || event.keyCode;
    if (key in activeTriggers)
        resetTrigger(activeTriggers[key]);

    return true;
}

document.addEventListener('keydown', KeyDownHandler, true);
document.addEventListener('keyup', KeyUpHandler, true);

function spikeTrigger(event) {
    const triggerId = parseInt(event.target.dataset.triggerId);
    setTrigger(triggerId);
    triggersToReset.push(triggerId);
}

document.getElementsByClassName('control-trigger').forEach(element => {
    const triggerId = parseInt(element.dataset.triggerId);
    triggerButtons[triggerId] = element;
    element.addEventListener('click', spikeTrigger);
});

/****************
 * MODEL
 ***************/

const loadingOverlay = document.getElementById('loading');
const modelSelect = document.getElementById('model-selector');
const backendIndicator = document.getElementById('backend-text');
var model = undefined;

function loadModel() {
    loadingOverlay.style.display = 'inherit';
    let modelUrl = 'models/' + modelSelect.value + '/model.json';

    return tf.loadGraphModel(modelUrl).then(function (loadedModel) {
        model = loadedModel;
        tf.tidy(() => {
            model.predict(tf.zeros([1, 128, 128, 1]))[0].data().then(() => {
                loadingOverlay.style.display = 'none';
            });
        });
    });
}

modelSelect.addEventListener('change', loadModel);
loadModel().then(() => {
    let backend = tf.getBackend();
    let styleClass = (backend != "cpu") ? 'accelerated' : 'non-accelerated';

    backend = (backend == "webgl") ? "WebGL" : backend.toUpperCase();
    backendIndicator.textContent = backend;
    backendIndicator.classList.add(styleClass);

    // start demo 1
    demoButtons[0].dispatchEvent(new Event('click'));
});

var contrastFactor = 1;
var brightnessFactor = 0;
var gammaFactor = 1;

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
    let frame = tf.browser.fromPixels(video, 3)
        .slice([y, x], [s, s])
        .resizeBilinear([128, 128])
        .mul(rgb).sum(2);

    if (contrastFactor != 1) {
        const mean = frame.mean()
        frame = frame.sub(mean).mul(tf.scalar(contrastFactor)).add(mean);
    }
    if (brightnessFactor != 0) frame = frame.add(tf.scalar(brightnessFactor).mul(_255));
    if (gammaFactor != 1)  frame = frame.pow(tf.scalar(gammaFactor));

    frame = frame.clipByValue(0, 255);

    if (controlInvert.checked) frame = _255.sub(frame);

    frame = frame.div(_255);

    tf.browser.toPixels(frame, input);

    let [maps, eb] = model.predict(frame.expandDims(0).expandDims(-1));

    // some older models have their output order swapped
    if (maps.rank < 4) [maps, eb] = [eb, maps];

    // let [pupil, glint] = maps.squeeze().split(2, 2);
    // take first channel in last dimension
    let pupil = maps.slice([0, 0, 0, 0], [-1, -1, -1, 1]).squeeze();

    let [eye, blink] = eb.squeeze().split(2);

    pupil = tf.cast(pupil.greaterEqual(threshold), 'float32').squeeze();

    let pupilArea = pupil.sum().data();
    let blinkProb = blink.data();

    pupil = pupil.array();

    // let drawn = tf.browser.toPixels(pupil, output);
    return [pupil, timestamp, timecode, pupilArea, blinkProb];
}

function keepLargestComponent(array) {

    let h = array.length;
    let w = array[0].length;

    // invert binary map
    for (let i = 0; i < h; ++i)
        for (let j = 0; j < w; ++j)
            array[i][j] = -array[i][j];

    // label and measure connected components
    // using iterative depth first search
    let label = 1;
    let maxCount = 0;
    let maxLabel = 0;

    for (let i = 0; i < h; ++i) {
        for (let j = 0; j < w; ++j) {
            if (array[i][j] >= 0) continue;

            let stack = [(i * h + j)];
            let pool = new Set();
            let count = 0;

            while (stack.length) {
                let node = stack.pop();
                if (pool.has(node)) continue;

                pool.add(node);
                let c = node % h;
                let r = Math.floor(node / h);
                if (array[r][c] === -1) {
                    array[r][c] = label;
                    if (r > 0 + 1) stack.push((r - 1) * h + c);
                    if (r < h - 1) stack.push((r + 1) * h + c);
                    if (c > 0 + 1) stack.push(r * h + c - 1);
                    if (c < w - 1) stack.push(r * h + c + 1);
                    ++count;
                }
            }

            if (count > maxCount) {
                maxCount = count;
                maxLabel = label;
            }

            ++label;
        }
    }

    // keeping largest component
    for (let i = 0; i < h; ++i) {
        for (let j = 0; j < w; ++j) {
            // array[i][j] = (array[i][j] == maxLabel) ? 1 : 0;
            if (array[i][j] > 0)
                array[i][j] = (array[i][j] == maxLabel) ? 1 : 0.3; // for debug purposes
        }
    }

    // returning area of the largest component
    return maxCount;
}

function findZero(array) {
    let h = array.length;
    let w = array[0].length;
    for (let i = 0; i < h; ++i)
        for (let j = 0; j < w; ++j)
            if (array[i][j] < 0.5)
                return [i, j];
    return null;
}

function floodFill(array) {
    let h = array.length;
    let w = array[0].length;
    let [r0, c0] = findZero(array);
    let stack = [r0 * h + c0];

    while (stack.length) {
        let node = stack.pop();
        let [r, c] = [Math.floor(node / h), node % h];
        if (array[r][c] != 1) {
            array[r][c] = 1;
            if (r > 0 + 1) stack.push((r - 1) * h + c);
            if (r < h - 1) stack.push((r + 1) * h + c);
            if (c > 0 + 1) stack.push(r * h + c - 1);
            if (c < w - 1) stack.push(r * h + c + 1);
        }
    }
}

function fillHoles(array) {
    let h = array.length;
    let w = array[0].length;
    let filled = array.map(r => r.map(c => c));
    floodFill(filled);

    let filledCount = 0;
    for (let i = 0; i < h; ++i) {
        for (let j = 0; j < w; ++j) {
            if (filled[i][j] == 0) {
                array[i][j] = 0.7; // debug
                ++filledCount;
            }
        }
    }

    return filledCount;
}


function findCentroid(array) {
    let nRows = array.length;
    let nCols = array[0].length;

    let m01 = 0,
        m10 = 0,
        m00 = 0;
    for (let i = 0; i < nRows; ++i) {
        for (let j = 0; j < nCols; ++j) {
            let v = (array[i][j] > 0.5) ? 1 : 0;
            m01 += j * v;
            m10 += i * v;
            m00 += v;
        }
    }

    return [(m01 / m00), (m10 / m00)];
}

function predictOnce() {
    if (!model) return;

    let outs = tf.tidy(predictFrame);

    return Promise.all(outs).then(outs => {
        let [pupil, timestamp, timecode, pupilArea, blinkProb] = outs;

        pupilArea = pupilArea[0];
        blinkProb = blinkProb[0];

        let pupilX = -1,
            pupilY = -1;

        if (pupilArea > 0) {

            if (controlMorphology.checked) {
                pupilArea = keepLargestComponent(pupil);
                pupilArea += fillHoles(pupil);
            }

            [pupilX, pupilY] = findCentroid(pupil);
            let x = parseInt(rx.value);
            let y = parseInt(ry.value);
            let s = parseInt(rs.value);

            pupilX = (pupilX * s / 128) + x;
            pupilY = (pupilY * s / 128) + y;
        }

        updatePupilLocator(pupilX, pupilY);

        // for Array, toPixel wants [0, 255] values
        for (let i = 0; i < pupil.length; ++i)
            for (let j = 0; j < pupil[0].length; ++j)
                if (pupil[i][j] > 0.5)
                    pupil[i][j] = [255 * pupil[i][j], 0, 0]; // red
                else {
                    let v = Math.round(pupil[i][j] * 255);
                    pupil[i][j] = [v, v, v]; // gray
                }

        tf.browser.toPixels(pupil, output);

        return [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY];
    });
}

function predictLoop() {
    if (!model) return;

    // playback is started but video is not loaded, wait
    if (!video.paused && video.readyState < 3) {
        window.requestAnimationFrame(predictLoop)
        return;
    }

    predictOnce().then(outs => {
        let [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY] = outs;

        // follow eye
        if (rt.checked && pupilX > 0) {
            let curX = rx.value;
            let curY = ry.value;

            let newX = Math.round(pupilX - rs.value / 2);
            let newY = Math.round(pupilY - rs.value / 2);

            newX = Math.round((1 - blinkProb) * newX + blinkProb * curX);
            newY = Math.round((1 - blinkProb) * newY + blinkProb * curY);

            let m = 0.2;
            newX = Math.round((1 - m) * newX + m * curX);
            newY = Math.round((1 - m) * newY + m * curY);

            let maxX = video.videoWidth - rs.value;
            let maxY = video.videoHeight - rs.value;

            newX = Math.min(Math.max(0, newX), maxX);
            newY = Math.min(Math.max(0, newY), maxY);

            rx.value = newX;
            ry.value = newY;

            setRoi();
        }

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

function startPredictionLoop() {
    if (controlClearOnResume.checked)
        clearData();
    else
        clearAfterCurTime();

    predictLoop();
}

video.addEventListener('play', startPredictionLoop);
video.addEventListener('seeked', predictOnce);

/***************
 * CONTROLS
 **************/

const controlContrast = document.getElementById('control-contrast');
const controlContrastPreview = document.getElementById('control-contrast-preview');
const controlBrightness = document.getElementById('control-brightness');
const controlBrightnessPreview = document.getElementById('control-brightness-preview');
const controlGamma = document.getElementById('control-gamma');
const controlGammaPreview = document.getElementById('control-gamma-preview');

const controlInvert = document.getElementById('control-invert');
const controlPreprocReset = document.getElementById('control-preproc-reset');

const controlPeriod = document.getElementById('control-period');
const controlThreshold = document.getElementById('control-thr');
const controlThresholdPreview = document.getElementById('control-thr-preview');
const controlMorphology = document.getElementById('control-morphology');

const controlClearOnResume = document.getElementById('control-clear-on-resume');

const controlPlotAutoUpdate = document.getElementById('control-plot-autoupdate');
const controlPlotWindow = document.getElementById('control-plot-window');
const controlPlotWindowEnable = document.getElementById('control-plot-window-enable');
const controlPlotSmooth = document.getElementById('control-plot-smooth');
const controlPlotUpdate = document.getElementById('control-plot-update');

const controlTableAutoUpdate = document.getElementById('control-table-autoupdate');
const controlTableUpdate = document.getElementById('control-table-update');
const controlExportCsv = document.getElementById('control-export-csv');
const controlClear = document.getElementById('control-clear');

function clearPreview() {
    ctx = input.getContext("2d");
    ctx.clearRect(0, 0, input.width, input.height);

    ctx = output.getContext("2d");
    ctx.clearRect(0, 0, output.width, output.height);
}

function clearData() {
    samples.length = 0; // clear array
    while (trace.firstChild) trace.removeChild(trace.lastChild);
    while (chartContainer.firstChild) chartContainer.removeChild(chartContainer.lastChild);
    chartData = null;
    chart = null;
    // video.pause();
    // video.currentTime = 0;
}

function clearAfterCurTime() {
    const curTime = video.currentTime - 0.01;
    samples = samples.filter(s => s[1] < curTime);
    updateChart();
    updateTable();
}

function setContrast(event) {
    if (event.target == controlContrast) {
        contrastFactor = event.target.value / 50;  // [0, 100] -> [0, 2]
        controlContrastPreview.value = contrastFactor.toFixed(2);
    } else if (event.target == controlContrastPreview) {
        contrastFactor = parseFloat(event.target.value);
        controlContrast.value = contrastFactor * 50;  // [0, 2] -> [0, 100]
    } else return;

    updatePrediction(5);
}

function setBrightness(event) {
    if (event.target == controlBrightness) {
        brightnessFactor = (event.target.value - 50) / 50;  // [0, 100] -> [-1, 1]
        controlBrightnessPreview.value = brightnessFactor.toFixed(2);
    } else if (event.target == controlBrightnessPreview) {
        brightnessFactor = parseFloat(event.target.value);
        controlBrightness.value = (brightnessFactor * 50) + 50;  // [-1, 1] -> [0, 100]
    } else return;

    updatePrediction(5);
}

function setGamma(event) {
    if (event.target == controlGamma) {
        gammaFactor = event.target.value / 50;  // [0, 100] -> [0, 2]
        controlGammaPreview.value = gammaFactor.toFixed(2);
    } else if (event.target == controlGammaPreview) {
        gammaFactor = parseFloat(event.target.value);
        controlGamma.value = gammaFactor * 50;  // [0, 2] -> [0, 100]
    } else return;

    updatePrediction(5);
}

function resetPreproc(event) {
    controlContrastPreview.value = "1.00";
    controlBrightnessPreview.value = "0.00";
    controlGammaPreview.value = "1.00";
    controlInvert.checked = false;

    controlContrastPreview.dispatchEvent(new Event('input'));
    controlBrightnessPreview.dispatchEvent(new Event('input'));
    controlGammaPreview.dispatchEvent(new Event('input'));
    controlInvert.dispatchEvent(new Event('change'));
}

function setThreshold(event) {
    if (event.target == controlThreshold) {
        threshold = event.target.value / 100;
        controlThresholdPreview.value = threshold.toFixed(2);
    } else if (event.target == controlThresholdPreview) {
        threshold = parseFloat(event.target.value);
        controlThreshold.value = threshold * 100;
    } else return;

    updatePrediction(5);
}

function setPeriod(event) {
    period = event.target.value;
}

controlContrastPreview.addEventListener('input', setContrast);
controlContrast.addEventListener('input', setContrast);
controlBrightnessPreview.addEventListener('input', setBrightness);
controlBrightness.addEventListener('input', setBrightness);
controlGammaPreview.addEventListener('input', setGamma);
controlGamma.addEventListener('input', setGamma);
controlInvert.addEventListener('change', () => { updatePrediction(5); });
controlPreprocReset.addEventListener('click', resetPreproc);
controlPeriod.addEventListener('change', setPeriod);
controlThresholdPreview.addEventListener('input', setThreshold);
controlThreshold.addEventListener('input', setThreshold);
controlMorphology.addEventListener('change', () => { updatePrediction(5); });

/***************
 * CHART & TABLE
 **************/

const sampleCount = document.getElementById('sample-count');
const chartContainer = document.getElementById('chart-container');

var chartWindow = parseInt(controlPlotWindow.value);
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
    // Instantiate and draw our chart, passing in some options.
    chart = new google.visualization.LineChart(chartContainer);
    google.visualization.events.addListener(chart, 'select', jumpFromChart);
}

function initChartData() {
    // Create the data table.
    chartData = new google.visualization.DataTable();
    chartData.addColumn('number', 'timecode');
    chartData.addColumn('number', 'pupil-area');
    chartData.addColumn('number', 'blink');
    // chartData.addColumn('number', 'pupil-x');
    // chartData.addColumn('number', 'pupil-y');
    for (var i = 0; i < nTriggers; ++i)
        chartData.addColumn('number', 'trigger' + (i + 1));

}

function initChartOptions() {
    let series = {
        0: {
            targetAxisIndex: 0
        },
        1: {
            targetAxisIndex: 1
        }
    }

    for (var i = 0; i < nTriggers; ++i)
        series[i + 2] = {
            targetAxisIndex: 1
        };

    // Set chart options
    chartOptions = {
        curveType: (controlPlotSmooth.checked) ? 'function' : 'none',
        chartArea: {
            width: '85%',
            height: '80%'
        },
        explorer: {
            axis: 'horizontal'
        },
        series: series,
        vAxes: {
            // Adds titles to each axis.
            0: {
                title: 'Pupil Area (px²)',
                minValue: 0,
                textPosition: 'in',
                viewWindow: {
                    min: 0
                }
            },
            1: {
                title: 'Blink (%)',
                format: 'percent',
                minValue: 0,
                maxValue: 1,
                textPosition: 'in',
                viewWindow: {
                    min: 0,
                    max: 1
                }
            }
        },
        hAxis: {
            title: 'time (s)',
            textPosition: 'in',
            viewWindowMode: 'maximized'
        },
        height: 225,
        legend: {
            position: 'top'
        },
    };
}

function jumpTo(timecode) {
    // jump only when not predicting
    if (video.paused) video.currentTime = timecode;
}

function jumpFromChart(event) {
    const selection = chart.getSelection();
    if (selection[0] && selection[0].row) {
        const timecode = chartData.getValue(selection[0].row, 0);
        jumpTo(timecode);
    }
}

function jumpFromTableRow(event) {
    if (this.children[1] && this.children[1].textContent) {
        const timecode = parseFloat(this.children[1].textContent);
        jumpTo(timecode);
    }
}

var samples = [];

function addSample(sample) {

    let [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY, triggers] = sample;

    let flatSample = sample.slice(0, -1).concat(triggers);
    samples.push(flatSample);
    sampleCount.textContent = "" + samples.length;

    if (controlTableAutoUpdate.checked) {

        let sampleRow = document.createElement('tr');
        sampleRow.innerHTML = (
            '<td>' + timestamp.toISOString() + '</td>' +
            '<td>' + timecode.toFixed(3) + '</td>' +
            '<td>' + pupilArea.toFixed(2) + '</td>' +
            '<td>' + blinkProb.toFixed(1) + '</td>' +
            '<td>' + pupilX.toFixed(1) + '</td>' +
            '<td>' + pupilY.toFixed(1) + '</td>' +
            triggers.map(t => '<td>' + t + '</td>').join(''));

        trace.appendChild(sampleRow);
        sampleRow.addEventListener('click', jumpFromTableRow, false);
        traceContainer.scrollTop = traceContainer.scrollHeight;
    }

    if (!chart) {
        initChart();
        initChartData();
        initChartOptions();
    }

    let chartSample = [timecode, pupilArea, blinkProb].concat(triggers);
    chartData.addRow(chartSample);

    let N = chartData.getNumberOfRows();
    if (N > chartWindow)
        chartData.removeRows(0, N - chartWindow);

    if (controlPlotAutoUpdate.checked)
        chart.draw(chartData, chartOptions);
}

function updateChart() {
    if (!chart) initChart();

    initChartData();
    initChartOptions();
    let chartSamples = samples.map(r => r.slice(1, 4).concat(r.slice(-nTriggers)));
    chartData.addRows(chartSamples);

    let N = chartData.getNumberOfRows();
    if (N > chartWindow)
        chartData.removeRows(0, N - chartWindow);

    chart.draw(chartData, chartOptions);
}

function updateTable() {
    trace.innerHTML = "";
    samples.forEach(sample => {
        let [timestamp, timecode, pupilArea, blinkProb, pupilX, pupilY] = sample.slice(0, -nTriggers);
        let triggers = sample.slice(-nTriggers);
        let sampleRow = document.createElement('tr');
        sampleRow.innerHTML = (
            '<td>' + timestamp.toISOString() + '</td>' +
            '<td>' + timecode.toFixed(3) + '</td>' +
            '<td>' + pupilArea.toFixed(2) + '</td>' +
            '<td>' + blinkProb.toFixed(1) + '</td>' +
            '<td>' + pupilX.toFixed(1) + '</td>' +
            '<td>' + pupilY.toFixed(1) + '</td>' +
            triggers.map(t => '<td>' + t + '</td>').join(''));

        trace.appendChild(sampleRow);
        sampleRow.addEventListener('click', jumpFromTableRow, false);
    });
    traceContainer.scrollTop = traceContainer.scrollHeight;
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
    let header = ['timestamp', 'timecode', 'pupil-area', 'blink', 'pupil-x', 'pupil-y'];
    let triggerHeader = Array.from(new Array(nTriggers), (val, index) => 'trigger' + (index+1));
    header = header.concat(triggerHeader);

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
        controlPlotWindow.disabled = false;
        chartWindow = controlPlotWindow.value;
    } else {
        controlPlotWindow.disabled = true;
        chartWindow = Infinity;
    }
}

function resizePlotWindow(event) {
    if (event.target.disabled == false)
        chartWindow = event.target.value;
}

controlPlotWindowEnable.addEventListener('change', togglePlotWindow);
controlPlotWindow.addEventListener('change', resizePlotWindow);
controlPlotSmooth.addEventListener('change', initChartOptions);

controlPlotUpdate.addEventListener('click', updateChart);
controlTableUpdate.addEventListener('click', updateTable);

controlClear.addEventListener('click', clearData);
controlExportCsv.addEventListener('click', exportCsv);

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
