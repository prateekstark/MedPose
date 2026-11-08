(async()=>{
while(!document.getElementById('sample'))await new Promise(r=>setTimeout(r,100));
await new Promise(r=>setTimeout(r,600));
document.getElementById('sample').click();
while(!document.getElementById('status').textContent.includes('complete')){
 await new Promise(r=>setTimeout(r,1000));
 const text=document.getElementById('status').textContent;
 if(/Error|failed|exceeds|incomplete/i.test(text))throw Error(text);
}
if(document.getElementById('empty').hidden!==true)throw Error('Missing mesh');
if(document.getElementById('export').disabled)throw Error('Export disabled');
return {status:document.getElementById('status').textContent,metrics:document.getElementById('metrics').textContent};
})()
