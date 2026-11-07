self.exports = {};
importScripts('./assets/vision_bundle.cjs');
const {FilesetResolver, PoseLandmarker} = self.exports;
const mathReady = import('./pose_math.js');
let engine;
self.onmessage = async ({data}) => {
 try {
  if(data.type==='init') {
   engine?.close();
   const files=await FilesetResolver.forVisionTasks('./assets/wasm');
   engine=await PoseLandmarker.createFromOptions(files,{
    baseOptions:{modelAssetPath:'./assets/'+data.model+'.task',delegate:'CPU'},
    runningMode:data.mode,numPoses:2,outputSegmentationMasks:false,
    minPoseDetectionConfidence:0.5,minPosePresenceConfidence:0.5,minTrackingConfidence:0.5
   });
   self.postMessage({type:'ready'});
  } else if(data.type==='frame') {
   try {
    const {packResult}=await mathReady;
    const result=data.mode==='VIDEO'?engine.detectForVideo(data.bitmap,Math.round(data.time*1000)):engine.detect(data.bitmap);
    self.postMessage({type:'result',result:packResult(result,data.time)});
   } finally {data.bitmap.close();}
  }
 } catch(error) {self.postMessage({type:'error',message:error.message||String(error)});}
};
