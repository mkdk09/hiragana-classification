const hiragana = ['あ', 'い', 'う', 'え', 'お',
                  'か', 'が', 'き', 'ぎ', 'く',
                  'ぐ', 'け', 'げ', 'こ', 'ご',
                  'さ', 'ざ', 'し', 'じ', 'す',
                  'ず', 'せ', 'ぜ', 'そ', 'ぞ',
                  'た', 'だ', 'ち', 'ぢ', 'つ',
                  'づ', 'て', 'で', 'と', 'ど',
                  'な', 'に', 'ぬ', 'ね', 'の',
                  'は', 'ば', 'ひ', 'び', 'ふ',
                  'ぶ', 'へ', 'べ', 'ほ', 'ぼ',
                  'ま', 'み', 'む', 'め', 'も',
                  'や', 'ゆ', 'よ',
                  'ら', 'り', 'る', 'れ', 'ろ',
                  'わ', 'ゐ', 'ゑ', 'を', 'ん'];
let drawElement, signaturePad;
let model;
let output;
document.addEventListener('DOMContentLoaded', () =>{
    output = document.getElementById('output');
    drawElement = document.getElementById('canvas');
    signaturePad = new SignaturePad(drawElement, {
        minWidth: 6,
        maxWidth: 6,
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
    /* output.innerText = "";*/
    const elements = document.querySelectorAll(".accuracy");
    elements.forEach(el => {
	el.parentNode.classList.remove('is-selected');
        el.innerText = '-';
    })
}

function prediction() {
    const imageData = getImageData();
    const accuracyScores = getAccuracyScores(imageData);
    const prediction = accuracyScores.indexOf(Math.max.apply(null, accuracyScores));
    const maxscore = Math.max.apply(null, accuracyScores);

    const elements = document.querySelectorAll(".accuracy");
    elements.forEach(el => {
	el.parentNode.classList.remove('is-selected');
	const rowIndex = Number(el.dataset.rowIndex);
	if (prediction === rowIndex) {
	    el.parentNode.classList.add('is-selected');
	}
	el.innerText = `${Math.round(accuracyScores[rowIndex] * 100 * 100) / 100}%`;
    })
    console.log(maxscore);
    console.log(prediction);
    /* output.innerText = `${hiragana[prediction]} : ${Math.round(maxscore * 100 * 100) / 100}%`; */
}
