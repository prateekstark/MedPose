// Geometric landmark measures, deliberately distinct from OpenSim joint coordinates.
export const VISIBILITY_MIN = 0.5; // Provisional display gate; not a clinical threshold.
export const EDGES = [[11,12],[11,13],[13,15],[15,17],[15,19],[17,19],[15,21],
 [12,14],[14,16],[16,18],[16,20],[18,20],[16,22],[11,23],[12,24],[23,24],
 [23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32],
 [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10]];
export const MEASURES = [
 {name:'landmark_knee_bend_l',label:'Left knee bend',points:[23,25,27]},
 {name:'landmark_knee_bend_r',label:'Right knee bend',points:[24,26,28]},
 {name:'landmark_elbow_bend_l',label:'Left elbow bend',points:[11,13,15]},
 {name:'landmark_elbow_bend_r',label:'Right elbow bend',points:[12,14,16]}
];
export function confidence(p) {
 if (!p || ![p.x,p.y,p.z,p.visibility].every(Number.isFinite)) return 0;
 // This runtime emits visibility but may omit per-landmark presence.
 const score = p.presence === undefined ? p.visibility : Math.min(p.visibility, p.presence);
 return Number.isFinite(score) ? Math.max(0,Math.min(1,score)) : 0;
}
export function bend(points, indices) {
 const p = indices.map(i=>points?.[i]);
 const score = Math.min(...p.map(confidence));
 if (score < VISIBILITY_MIN) return {value:null,confidence:score};
 const a=[p[0].x-p[1].x,p[0].y-p[1].y,p[0].z-p[1].z];
 const b=[p[2].x-p[1].x,p[2].y-p[1].y,p[2].z-p[1].z];
 const norm=Math.hypot(...a)*Math.hypot(...b);
 if (norm<1e-10) return {value:null,confidence:score};
 const cosine=Math.max(-1,Math.min(1,a.reduce((s,v,i)=>s+v*b[i],0)/norm));
 return {value:180-Math.acos(cosine)*180/Math.PI,confidence:score};
}
export function packResult(result,time) {
 const count=result.landmarks.length;
 const world=count===1?result.worldLandmarks[0]:null;
 const valid=world?.length===33&&world.every(p=>[p.x,p.y,p.z].every(Number.isFinite));
 const points=valid?world:null;
 return {time,pose_count:count,status:count>1?'multiple_people':points?'detected':'no_pose',
  landmarks:points?result.landmarks[0]:null,world:points,
  measures:MEASURES.map(m=>bend(points,m.points))};
}
export function toTrial(frames,rate,kind) {
 if (!frames.length || frames.some((f,i)=>!Number.isFinite(f.time)||(i&&f.time<=frames[i-1].time))) throw Error('Invalid frame times');
 return {schema_version:1,time:frames.map(f=>f.time),coordinate_names:MEASURES.map(m=>m.name),
 units:MEASURES.map(()=> 'deg'),values:frames.map(f=>f.measures.map(m=>m.value)),
 confidence:frames.map(f=>f.measures.map(m=>m.confidence)),recording_mode:'1_phone',
 engine_name:'MediaPipe Pose Landmarker full — geometric landmark bends ('+kind+')',
 engine_version:'tasks-vision 0.10.21; model float16/1',camera_count:1,frame_rate:rate,supports:[]};
}
