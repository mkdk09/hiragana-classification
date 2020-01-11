let drawElement, signaturePad;
let model;
let output;
document.addEventListener('DOMContentLoaded', () =>{
    output = document.getElementById('output');
    drawElement = document.getElementById('canvas');
    signaturePad = new SignaturePad(drawElement, {
        minWidth: 12,
        maxWidth: 12,
        penColor: 'black',
        backgroundColor: 'white',
    });
});

tf.loadModel('./tfjs/model.json').then(pretrainedModel => model = pretrainedModel);

function getImageData() {
    const inputWidth = inputHeight = 28;
    const canvas = document.getElementById('canvas').getContext('2d');
    canvas.drawImage(drawElement, 0, 0, inputWidth, inputHeight);

    let imageData = canvas.getImageData(0, 0, inputWidth, inputHeight);
    /*
    for (let i = 0; i < imageData.data.length; i+=4) {
        const avg = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
        imageData.data[i] = imageData.data[i+1] = imageData.data[i+2] = avg;
    }
    */
    return imageData;
}

function getAccuracyScores(imageData) {
    const score = tf.tidy(() => {
        const channels = 1;
        let input = tf.fromPixels(imageData, channels);
        input = tf.cast(input, 'float32').div(tf.scalar(255));
        input = input.expandDims();
        return model.predict(input).dataSync();
    })
    return score;
}

function reset() {
    signaturePad.clear();
    output.innerText = "";
}

function prediction() {
    const imageData = getImageData();
    const accuracyScores = getAccuracyScores(imageData);
    const maxAccuracy = accuracyScores.indexOf(Math.max.apply(null, accuracyScores));
    const maxscore = Math.max.apply(null, accuracyScores);

    console.log(maxscore);
    console.log(maxAccuracy);
    output.innerText = maxAccuracy;
}
