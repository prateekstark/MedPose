"use strict";
// Test-only Chrome DevTools transport over OS pipes, never a TCP debug port.
// Usage: node tools/pose_browser_check.cjs chrome page-url width height profile screenshot checks.js
const {spawn} = require("node:child_process");
const fs = require("node:fs");
const [chrome, pageURL, widthText, heightText, profile, screenshot] = process.argv.slice(2);
const width = Number(widthText), height = Number(heightText);
const browser = spawn(chrome, [
    "--headless", "--no-sandbox", "--disable-gpu", "--disable-background-networking",
    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost", "--disable-extensions", "--disable-sync", "--disable-crash-reporter", "--no-first-run",
    "--no-default-browser-check", "--remote-debugging-pipe", "--user-data-dir=" + profile
], {stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"]});
let nextId = 1, buffer = "", stderr = "";
const pending = new Map();
const requests = [];
browser.stderr.on("data", chunk => {stderr += chunk.toString();});
browser.stdio[4].setEncoding("utf8");
browser.stdio[4].on("data", chunk => {
    buffer += chunk;
    while (buffer.includes("\0")) {
        const end = buffer.indexOf("\0");
        const message = JSON.parse(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
        if (message.method === "Network.requestWillBeSent") requests.push({url:message.params.request.url, method:message.params.request.method, hasPostData:!!message.params.request.hasPostData});
        if (message.method === "Target.attachedToTarget") {
            const workerSession = message.params.sessionId;
            call("Network.enable", {}, workerSession).then(()=>call("Runtime.runIfWaitingForDebugger", {}, workerSession)).catch(()=>{});
        }
        const request = pending.get(message.id);
        if (request) {
            pending.delete(message.id);
            if (message.error) request.reject(Error(JSON.stringify(message.error)));
            else request.resolve(message.result);
        }
    }
});
function call(method, params = {}, sessionId) {
    return new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, {resolve, reject});
        browser.stdio[3].write(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}) + "\0");
    });
}
const checks = fs.readFileSync(process.argv[8], "utf8");
const timeout = setTimeout(() => {browser.kill(); console.error("Browser timeout", stderr.slice(-1000)); process.exit(1);}, 180000);
(async () => {
    try {
        const {targetId} = await call("Target.createTarget", {url: "about:blank"});
        const {sessionId} = await call("Target.attachToTarget", {targetId, flatten: true});
        await call("Page.enable", {}, sessionId);
        await call("Network.enable", {}, sessionId);
        await call("Target.setAutoAttach", {autoAttach:true,waitForDebuggerOnStart:true,flatten:true}, sessionId);
        await call("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: width < 600}, sessionId);
        await call("Page.navigate", {url: pageURL}, sessionId);
        const result = await call("Runtime.evaluate", {
            expression: checks, awaitPromise: true, returnByValue: true
        }, sessionId);
        if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
        const layout = await call("Page.getLayoutMetrics", {}, sessionId);
        const capture = await call("Page.captureScreenshot", {format: "png", captureBeyondViewport: true,
            clip: {x: 0, y: 0, width, height: Math.min(layout.cssContentSize.height, 5000), scale: 1}}, sessionId);
        fs.writeFileSync(screenshot, Buffer.from(capture.data, "base64"));
        process.stdout.write(JSON.stringify({...result.result.value,browserRequests:requests}));
        await call("Browser.close");
    } catch (error) {
        console.error(error.message, stderr.slice(-1000));
        process.exitCode = 1;
    } finally {
        clearTimeout(timeout);
        browser.kill();
    }
})();
