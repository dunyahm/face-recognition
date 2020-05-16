
const faceapi = require('face-api.js');
// SsdMobilenetv1Options
const minConfidence = 0.5;
const getFaceDetectorOptions = () => new faceapi.SsdMobilenetv1Options(
    {minConfidence});
const faceDetectionOptions = getFaceDetectorOptions();


const findFace = async (faceapi, faceMatchers, imageReceived, canvas) => {

    const queryImage = await canvas.loadImage(imageReceived)

    const resultsQuery = await faceapi.detectSingleFace(queryImage,
        faceDetectionOptions).withFaceLandmarks().withFaceDescriptor();

      
        let bestMatches = [];
        
        for (let descriptor of faceMatchers) {
            const { personName, descriptors } = descriptor;
            const descriptorsFloat32 = [];
            descriptors.forEach(element => {
                descriptorsFloat32.push(new Float32Array(element));
            });
            const labelledDescriptors = new faceapi.LabeledFaceDescriptors(
                personName,
                descriptorsFloat32,
                );
            const faceMatcher = new faceapi.FaceMatcher(labelledDescriptors);
            const bestMatch = faceMatcher.findBestMatch(resultsQuery.descriptor)
            bestMatches.push(bestMatch.toString())
        }
        const bestMatchesResults =  bestMatches.filter(bestMatch => !bestMatch.includes("unknown"))
    
        return bestMatchesResults[0];
      
}

module.exports = {findFace}