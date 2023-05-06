// Original source https://hackernoon.com/how-to-run-machine-learning-models-in-the-browser-using-onnx
const ort = require('onnxruntime-web');

const canvas = document.createElement("canvas"),
  ctx = canvas.getContext("2d");

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target || window.event.src,
    files = target.files;

  if (FileReader && files && files.length) {
      var fileReader = new FileReader();
      fileReader.onload = () => onLoadImage(fileReader);
      fileReader.readAsDataURL(files[0]);
  }
}

function onLoadImage(fileReader) {
    var img = document.getElementById("input-image");
    img.onload = () => handleImage(img);
    img.src = fileReader.result;
}

function handleImage(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0, img.width, img.height);
  var inputTensor = imageDataToTensor(ctx.getImageData(0, 0, img.width, img.height).data, [1, 3, img.height, img.width]);
  run(inputTensor);
}

function imageDataToTensor(data, dims) {
  // 1a. Extract the R, G, and B channels from the data
  const [R, G, B] = [[], [], []]
  for (let i = 0; i < data.length; i += 4) {
    R.push(data[i]);
    G.push(data[i + 1]);
    B.push(data[i + 2]);
    // 2. skip data[i + 3] thus filtering out the alpha channel
  }
  // 1b. concatenate RGB ~= transpose [224, 224, 3] -> [3, 224, 224]
  const transposedData = R.concat(G).concat(B);

  // 3. convert to float32
  let i, l = transposedData.length; // length, we need this for the loop
  const float32Data = new Float32Array(dims[0]*dims[1]*dims[2]*dims[3]); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }

  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

async function run(inputTensor) {
  try {
    const start = Date.now();

    const sessionOption1 = { executionProviders: ['webgl'], logSeverityLevel: 0 };
    const session1 = await ort.InferenceSession.create('./esrgan-small-pre.onnx', sessionOption1);
    const sessionOption2 = { executionProviders: ['wasm'], logSeverityLevel: 0 };
    const session2 = await ort.InferenceSession.create('./esrgan-small-end.onnx', sessionOption2);

    // prepare feeds. use model input names as keys.
    //const feeds = { input: new ort.Tensor('float32', inputData, dims) };
    const feeds = { input: inputTensor };

    // feed inputs and run
    const resultspre = await session1.run(feeds);
    const end1 = Date.now();
    console.log(`Execution time pre: ${end1 - start} ms`);
    const feed2 = { input: inputTensor, input_pre: resultspre.output };
    const results = await session2.run(feed2);
    const end2 = Date.now();
    console.log(`Execution time end: ${end2 - end1} ms`);

    //display result
    const canvas = document.createElement("canvas");
    const context = canvas.getContext('2d');
    canvas.height = results.output.dims[2];
    canvas.width = results.output.dims[3];
    const img = results.output.toImageData({ format: 'RGB', tensorLayout: 'NCHW'});
    context.putImageData(img, 0, 0);
    document.getElementById("canvas-image").src = canvas.toDataURL();

    const end = Date.now();
    console.log(`Execution time: ${end - start} ms`);
    document.getElementById("executeTime").innerHTML = `Execution time: ${end - start} ms`;
  } catch (e) {
    console.log(e);
  }
}
