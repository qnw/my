
import * as faceapi from './dist/face-api.esm.js';
const modelPath = './model/';
const minScore = 0.2; // minimum score
const maxResults = 5; // maximum number of results to return
let optionsSSDMobileNet;


    const video = document.getElementById('video');
    const nameInput = document.getElementById('nameInput');
    const statusEl = document.getElementById('status');
    const uploadStatus = document.getElementById('uploadStatus');
    const descriptorsMap = {}; // 名稱對應 descriptor 陣列


function str(json) {
  let text = '<font color="lightblue">';
  text += json ? JSON.stringify(json).replace(/{|}|"|\[|\]/g, '').replace(/,/g, ', ') : '';
  text += '</font>';
  return text;
}

function log(...txt) {
  console.log(...txt); // eslint-disable-line no-console
  const div = document.getElementById('log');
  if (div) div.innerHTML += `<br>${txt}`;
}





//	var faceapi="";

    async function startVideo() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
      video.srcObject = stream;
    }

	async function loadModels() {
	  // load face-api models
	  // await faceapi.nets.tinyFaceDetector.load(modelPath); // using ssdMobilenetv1
	  await faceapi.nets.ssdMobilenetv1.load(modelPath);
//	  await faceapi.nets.ageGenderNet.load(modelPath);
	  await faceapi.nets.faceLandmark68Net.load(modelPath);
//	  await faceapi.nets.faceRecognitionNet.load(modelPath);
//	  await faceapi.nets.faceExpressionNet.load(modelPath);
	  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({ minConfidence: minScore, maxResults });
	  // check tf engine state
	  log(`Models loaded: ${str(faceapi.tf.engine().state.numTensors)} tensors`);
	}

    async function loadModels_old() {
      const MODEL_URL = './model/';
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    }

    async function registerFace() {
      const name = nameInput.value.trim();
      if (!name) return statusEl.innerText = '❗ 請輸入姓名';

      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) return statusEl.innerText = '❌ 沒偵測到臉';
      if (!descriptorsMap[name]) descriptorsMap[name] = [];
      descriptorsMap[name].push(detection.descriptor);
      statusEl.innerText = `✅ 為 ${name} 註冊 1 筆樣本`;
    }

    async function handleImageUpload(event) {
      const file = event.target.files[0];
      const name = document.getElementById('uploadName').value.trim();
      if (!file || !name) return uploadStatus.innerText = '❗ 請選圖 + 姓名';

      const img = await loadImageFromFile(file);
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) return uploadStatus.innerText = '❌ 無臉部偵測';
      if (!descriptorsMap[name]) descriptorsMap[name] = [];
      descriptorsMap[name].push(detection.descriptor);
      uploadStatus.innerText = `✅ 為 ${name} 加入一張圖片樣本`;
    }

    function loadImageFromFile(file) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
      });
    }

    function downloadData() {
      const jsonData = {};
      for (const [name, descriptors] of Object.entries(descriptorsMap)) {
        jsonData[name] = descriptors.map(d => Array.from(d));
      }
      const blob = new Blob([JSON.stringify(jsonData)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'face_descriptors.json';
      a.click();
      URL.revokeObjectURL(url);
      statusEl.innerText = '✅ 匯出完成';
    }

    function loadData(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const jsonData = JSON.parse(reader.result);
        for (const [name, list] of Object.entries(jsonData)) {
          descriptorsMap[name] = list.map(arr => new Float32Array(arr));
        }
        statusEl.innerText = `✅ 載入 ${Object.keys(descriptorsMap).length} 人樣本資料`;
      };
      reader.readAsText(file);
    }

    video.addEventListener('play', () => {
      const canvas = faceapi.createCanvasFromMedia(video);
      document.body.append(canvas);
      const displaySize = { width: video.width, height: video.height };
      faceapi.matchDimensions(canvas, displaySize);

      setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors();

        const resized = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        faceapi.draw.drawDetections(canvas, resized);
        faceapi.draw.drawFaceLandmarks(canvas, resized);

        const labeledDescriptors = Object.entries(descriptorsMap).map(([name, descriptors]) =>
          new faceapi.LabeledFaceDescriptors(name, descriptors)
        );

        if (labeledDescriptors.length > 0) {
          const matcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
          resized.forEach(detection => {
            const best = matcher.findBestMatch(detection.descriptor);
            const box = detection.detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: best.toString() });
            drawBox.draw(canvas);
          });
        }
      }, 200);
    });

    loadModels().then(startVideo);
