// Second approach to the project
// This is the main file for the project

import '@marcellejs/core/dist/marcelle.css';
import * as marcelle from '@marcellejs/core';

import { createCanvas, loadImage } from 'canvas';
// import { from,defer,fromEvent } from 'rxjs';
// import { map,filter } from 'rxjs/operators';

const featureExtractor = marcelle.mobileNet();

//data store
const store = marcelle.dataStore('localStorage');
const trainingSet = marcelle.dataset('TrainingSet', store);
const model = marcelle.mlpClassifier({ layers: [64, 32], epochs: 20 }).sync(store, 'classifier');
const termSet = marcelle.dataset('Term', store);


//init data
await termSet.ready.then( () => {
  fetch('input.txt')
  .then(response => response.text())
  .then(data => {
    let arr = data.split(' ');
    for (var i=0,len=arr.length; i<len; i++)
    {
      console.log(arr[i]);
      termSet.create({ terms :  arr[i] });
    }
     termSet.distinct('terms').then((labels) => {
        console.log(labels.length);
      });
  })
  .catch(error => {
    console.log(error);
  });
});



//Teaching Pad
const teachingPad = marcelle.sketchPad({
});
const teachingLabel = marcelle.textInput();
const teach = marcelle.button("Teach");
teach.title = 'Teach machine!';



//Display of term.
let classLabelValue = "Term";
const classLabel = marcelle.text("<center><h2>"+classLabelValue+"</h2></center>");
classLabel.title  = "Term";



//Drawing
const userDrawing = marcelle.sketchPad({
});



//Score of drawing
let userScore = 0;
const score = marcelle.text("<center><h2>"+Math.round(userScore) + ' /100'+"</h2></center>");
score.title = 'Your score is:';




//Prediction
let wln = "</br></br></br>";
let predicitonValue = "prediction";
const predicitonLabel = marcelle.text(wln+wln+"<center><h3>This is a "+predicitonValue+".</h3></center>");
predicitonLabel.title  = "Prediction";



//Buttons
const save = marcelle.button("Submit");
save.title = 'Save your drawing!';
const start = marcelle.button("Generate");
start.title = 'Generate a term!';




//Picture change
userDrawing.$images.subscribe(imgData => console.log(imgData.data));
save.$click.subscribe(()=>{
  console.log(userDrawing.$images);
});




//Teach
const $instancesTeach = teach.$click
  .sample(teachingPad.$images.zip((thumbnail, data) => ({ thumbnail, data }), teachingPad.$thumbnails))
  .map(async ({ thumbnail, data }) => ({
    x: await featureExtractor.process(data),
    y: teachingLabel.$value.get(),
    note: 10,
    thumbnail,
  }))
  .awaitPromises();


$instancesTeach.subscribe((x)=>{
  //new term
  {
    let y = x['y'];
    y =  y[0].toUpperCase()+y.substring(1);
    x['y'] = y;
    console.log(x['y']);
    termSet.create({ terms :  y });
    termSet.distinct('terms').then((labels) => {
      console.log(labels)
      });
  };
  trainingSet.create(x);
});








//training
trainingSet.$changes.subscribe(async (changes) => {
  await trainingSet.ready;
  const labels = await trainingSet.distinct('y');
  if (changes.length === 0 || changes[0].level === 'dataset') return;
  if (labels.length < 2) {
    marcelle.notification({
      title: 'Tip',
      message: 'You need to have at least two classes to train the model',
      duration: 5000,
    });
  } else if (trainingSet.$count.get() < 4) {
    marcelle.notification({
      title: 'Tip',
      message: 'You need to have at least two example in each class',
      duration: 5000,
    });
  } else {
    model.train(trainingSet);
    console.log("train finish");
  }
});



//features
const $features = userDrawing.$images.map((imgData) => featureExtractor.process(imgData)).awaitPromises();



//training
const $trainingSuccess = model.$training.filter((x) => x.status === 'success');



//prediction
const $predictions = $features
  .merge($trainingSuccess.sample($features))
  .map((features) => model.predict(features))
  .awaitPromises();




//display
$predictions.subscribe(({ label,confidences }) => {
  predicitonLabel.$value.set(wln+wln+" <center><h3>This is a "+label+".</h3></center>");
  userScore = confidences[classLabelValue] * 100;
  score.$value.set("<center><h2>"+Math.round(userScore) + ' /100'+"</h2></center>");
});




//Button clicks
//Save
const $instancesSave = save.$click
  .sample(userDrawing.$images.zip((thumbnail, data) => ({ thumbnail, data }), userDrawing.$thumbnails))
  .map(async ({ thumbnail, data }) => ({
    x: await featureExtractor.process(data),
    y: classLabelValue,//classLabel.$value.get(),
    note: 10,
    thumbnail,
  }))
  .awaitPromises();

$instancesSave.subscribe((x)=>{
  trainingSet.create(x);
});



//Generate term
start.$click.subscribe(() => {
  termSet.distinct('terms').then((labels) => {
        let r = getRandomInt(0,labels.length-1);
        let theme = labels[r];
        theme = theme[0].toUpperCase()+theme.substring(1);
        classLabelValue = theme;
        classLabel.$value.set("<center><h2>"+classLabelValue+"</h2></center>");
      });
});






//Dashboard
let myDashboard = marcelle.dashboard({
  title: 'Project',
  author: 'Teach me to draw',
});

myDashboard
  .page('Draw')
  .use(start)
  .use(classLabel)
  .use([userDrawing,predicitonLabel])
  .use(save)
  .use(score)
  .sidebar(teachingPad)
  .sidebar(teachingLabel,teach)
  ;































const generateImage = marcelle.button("Start!");
generateImage.title = 'Generate Image';



let ik;
generateImage.$click.subscribe(() => {
  console.log(imageViewer.$images);
  trainingSet
  .items()
  .query({ y: 'Car' })
  .select(['id','y', 'thumbnail'])
  .toArray()
  .then((x)=>{
    console.log(x);
      ik = getImageData(x[0]["thumbnail"],300,300);
      imageViewer(getImageData(x[0]["thumbnail"],300,300));
      console.log(x[0]["thumbnail"]);
      console.log(getImageData(x[0]["thumbnail"],300,300));

  });

});
const imageViewer = marcelle.imageDisplay(ik);




myDashboard
  .page('Guess')
  .use(imageViewer)
  .sidebar( generateImage);



myDashboard.show();








//Tool
function getRandomInt (min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

async function getImageData(path,width,height) {

  const img = await loadImage(path);

  const canvas = createCanvas(width,height);
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, width,height);

  const imageData = ctx.getImageData(0, 0, width,height);

  return imageData;
}