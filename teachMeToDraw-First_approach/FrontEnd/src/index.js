// First approach to the problem
// This is the main file for the project

import '@marcellejs/core/dist/marcelle.css';
import {
	dashboard,
	sketchPad,
	button,
	textInput,
	dataStore,
	dataset,
	datasetBrowser,
	mobileNet,
	imageDisplay,
	imageUpload,
	onnxModel
} from '@marcellejs/core';

let userScore = 0;

//.................original image.....................
const originalImage = imageUpload({
	width: 100,
	height: 100
});
const instanceViewer = imageDisplay(originalImage.$images);

//.................Sketch pad.....................
const featureExtractor = mobileNet();
const userDrawing = sketchPad();
const saveButton = button('Submit drawing');
saveButton.title = 'Save your drawing to get your score!';

//................Name of the image to draw.....................
const username = textInput('Apple');
username.title = 'What do you want to draw?';

//..................Display the score.....................
const score = textInput(userScore + ' /100');
score.title = 'Your score is:';	

//..................Function to get the score and save good drawings to new dataset.....................
  
async function extractFeatures(img) {
const model = onnxModel({
	inputType: 'image',
	taskType: 'classification',
	inputShape: [1, 1, 300, 300]
});
await model.loadFromUrl("latest_version4_model.onnx");
const $features = userDrawing.$images
	.map((img) => featureExtractor.process(img))
	.awaitPromises();

const $predictions = $features.map((features) => model.predict(features)).awaitPromises();
userScore = $predictions.value;
score.$value.set(userScore + ' /100');

if(userScore > 90){
	return {
		x: await featureExtractor.process(img),
		y: username.$value.get(),
		thumbnail: userDrawing.$thumbnails.get(),
	};
}
return null;
}

const $instances = saveButton.$click
	.sample(userDrawing.$images)
	.map(extractFeatures)
	.awaitPromises();

const store = dataStore('localStorage');
const trainingSet = dataset('TrainingSet-Umap', store);

$instances.subscribe(trainingSet.create);

const trainingSetBrowser = datasetBrowser(trainingSet);

const dash = dashboard({
	title: 'Teach me to Draw!',
	author: '',
});

dash
	.page('Draw')
	.use(username)
	.use(userDrawing)
	.use(saveButton)
	.sidebar(originalImage)
	.sidebar(instanceViewer)
	.use(score);


dash
	.page('Stats')
	.use(trainingSetBrowser);

dash.show();