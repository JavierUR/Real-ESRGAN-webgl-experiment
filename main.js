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

async function tileProc(inputTensor){
  const sessionOption1 = { executionProviders: ['webgl'], logSeverityLevel: 0 };
  const session1 = await ort.InferenceSession.create('./esrgan-small-pre.onnx', sessionOption1);
  const sessionOption2 = { executionProviders: ['wasm'], logSeverityLevel: 0 };
  const session2 = await ort.InferenceSession.create('./esrgan-small-end.onnx', sessionOption2);

  const inputDims = inputTensor.dims;
  const imageW = inputDims[3];
  const imageH = inputDims[2];

  const rOffset = 0;
  const gOffset = imageW*imageH;
  const bOffset = imageW*imageH*2;

  const outputDims = [inputDims[0], inputDims[1], inputDims[2]*4, inputDims[3]*4];
  const outputTensor = new ort.Tensor("float32", new Float32Array(outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]), outputDims);

  const outImageW = outputDims[3];
  const outImageH = outputDims[2];
  const outROffset = 0;
  const outGOffset = outImageW*outImageH;
  const outBOffset = outImageW*outImageH*2;

  const tileSize = 128;
  const tilesx = Math.ceil( inputDims[3] / tileSize );
  const tilesy = Math.ceil( inputDims[2] / tileSize );

  const data = inputTensor.data;

  console.log(inputTensor);
  const numTiles = tilesx*tilesy;
  var currentTile = 0;

  for (let i = 0; i < tilesx; i++) {
    for (let j = 0; j < tilesy; j++) {
      const ti = Date.now();
      const tileW = Math.min(tileSize, imageW - i * tileSize);
      const tileH = Math.min(tileSize, imageH - j * tileSize);
      console.log("tileW: " + tileW + " tileH: " + tileH);
      const tileROffset = 0;
      const tileGOffset = tileSize*tileSize;
      const tileBOffset = tileSize*tileSize*2;

      const tileData = new Float32Array(tileSize*tileSize*3);
      for (let xp = 0; xp < tileSize; xp++) {
        for (let yp = 0; yp < tileSize; yp++) {
          const x = xp < tileW ? xp : tileW - 1;
          const y = yp < tileH ? yp : tileH - 1;
          const idx = (i * tileSize + x) + (j * tileSize + y) * imageW;
          tileData[x + y * tileSize + tileROffset] = data[idx + rOffset];
          tileData[x + y * tileSize + tileGOffset] = data[idx + gOffset];
          tileData[x + y * tileSize + tileBOffset] = data[idx + bOffset];
        }
      }

      const tile = new ort.Tensor("float32", tileData, [1, 3, tileSize, tileSize]);
      const resultspre = await session1.run({input: tile});
      console.log("pre dims:" + resultspre.output.dims);
      const feed2 = { input: tile, input_pre: resultspre.output };
      const results = await session2.run(feed2);
      console.log("proc tile dims:" + results.output.dims);

      const outTileW = tileW*4;
      const outTileH = tileH*4;
      const outTileSize = tileSize*4;

      const outTileROffset = 0;
      const outTileGOffset = outTileSize*outTileSize;
      const outTileBOffset = outTileSize*outTileSize*2;

      // add tile to output
      for (let x = 0; x < outTileW; x++) {
        for (let y = 0; y < outTileH; y++) {
          const idx = (i * outTileSize + x) + (j * outTileSize + y) * outImageW ;
          outputTensor.data[idx + outROffset] = results.output.data[x + y * outTileSize + outTileROffset];
          outputTensor.data[idx + outGOffset] = results.output.data[x + y * outTileSize + outTileGOffset];
          outputTensor.data[idx + outBOffset] = results.output.data[x + y * outTileSize + outTileBOffset];
        }
      }
      currentTile++;
      const dt = Date.now() - ti;
      const remTime = (numTiles - currentTile) * dt;
      console.log("tile " + currentTile + " of " + numTiles + " took " + dt + " ms, remaining time: " + remTime + " ms");
    }
  }
  console.log("output dims:" + outputTensor.dims);
  return outputTensor;

}

async function run(inputTensor) {
  try {
    const start = Date.now();

    const result = await tileProc(inputTensor);
    console.log("output dims:" + result.dims);

    //display result
    const canvas = document.createElement("canvas");
    const context = canvas.getContext('2d');
    canvas.height = result.dims[2];
    canvas.width = result.dims[3];
    const img = result.toImageData({ format: 'RGB', tensorLayout: 'NCHW'});
    context.putImageData(img, 0, 0);
    document.getElementById("canvas-image").src = canvas.toDataURL();

    const end = Date.now();
    console.log(`Execution time: ${end - start} ms`);
    //document.getElementById("executeTime").innerHTML = `Execution time: ${end - start} ms`;
  } catch (e) {
    console.log(e);
  }
}
