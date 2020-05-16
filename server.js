
const express = require("express");
const path = require("path");

const app = express();
const server = require('http').Server(app);
const io = require('socket.io')(server);
const {processImage} = require('./storeFace')
const {findFace} = require('./detectFace')
const Multer = require('multer');
const fs = require("fs");
const multer = Multer({
  storage: Multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024 // no larger than 5mb, you can change as needed.
  }
});

require("@tensorflow/tfjs-node");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({
  Canvas,
  Image,
  ImageData,
});

let faceMatchers = [];

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const viewsDir = path.join(__dirname, "views");
app.use(express.static(viewsDir));
app.use(express.static(path.join(__dirname, "./public")));
app.use(express.static(path.join(__dirname, "./images")));
app.use(express.static(path.join(__dirname, "../media")));

app.get("/", (req, res) =>  {
  res.sendFile(path.join(__dirname + '/views/index.html'));
});


app.get("/admin", (req, res) => {
  res.sendFile(path.join(__dirname + '/views/admin.html'));
});

app.post("/savePersonImage", multer.array('faceImages', 2), (req, res) => {
  let personName = req.body.personName
  let file = req.files;
  if (file) {
    processImage(file, personName, faceapi, canvas).then(async (descriptors) => {
      faceMatchers.push(descriptors);
      await writeToFile(descriptors);
      res.redirect('http://' + req.headers.host + "/admin");
    }).catch((error) => {
      console.error(error);
    });
  }
})

async function writeToFile(newResult) {
  const existingResults = await readFromFile()
  existingResults.push(newResult)
  await fs.promises.writeFile("results.json", JSON.stringify(existingResults))
}

 async function readFromFile() {
  const existingResults = await fs.promises.readFile('results.json');
  return JSON.parse(existingResults);
}

io.on('connection', (socket) => {
  console.log("new user");
  
  socket.on('new_image', async (data) => {
    console.log("image received")
    let result = "empty";
    if (faceMatchers.length !== 0 ) {
      try {
        result = await findFace(faceapi, faceMatchers, data.image, canvas)
      } catch (e) {
        console.log("Error in detecting face", e)
      } 
    }
    io.emit('result', result)
  })
})

async function loadModels() {
  const MODEL_URL = "./models";
  const faceDetectionNet = faceapi.nets.ssdMobilenetv1

  
  await faceDetectionNet.loadFromDisk(MODEL_URL)
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
  console.log("models loaded");
}

loadModels().then(async () => {
  faceMatchers = await readFromFile()
  server.listen(3000, () => console.log("Listening on port 3000!"));
})
