"""Serve the local pose workspace, never repository files or uploaded media."""
from __future__ import annotations

import argparse
import json
from medpose.engine.smpl_assets import status as smpl_status
from medpose.ui.models import MODELS, catalog, installed, model_path, start_install
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.conda-envs' / 'pose-web'
UI = Path(__file__).resolve().parent
ROUTES = {
    '/': UI / 'pose.html', '/pose.html': UI / 'pose.html',
    '/pose.js': UI / 'pose.js', '/pose.css': UI / 'pose.css',
    '/pose_math.js': UI / 'pose_math.js', '/pose_worker.js': UI / 'pose_worker.js',
    '/review': UI / 'demo.html',
    '/samples/synthetic-adult.png': UI / 'samples' / 'synthetic-adult.png',
    '/samples/synthetic-still.mp4': ASSETS / 'synthetic-still.mp4',
    '/assets/vision_bundle.cjs': ASSETS / 'package' / 'vision_bundle.cjs',
}
for model in MODELS:
    ROUTES['/assets/' + model + '.task'] = model_path(model)
for name in ('vision_wasm_internal.js', 'vision_wasm_internal.wasm',
             'vision_wasm_nosimd_internal.js', 'vision_wasm_nosimd_internal.wasm'):
    ROUTES['/assets/wasm/' + name] = ASSETS / 'package' / 'wasm' / name
CSP = ("default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; "
       "style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob: data:; "
       "connect-src 'self'; worker-src 'self'; base-uri 'none'; "
       "form-action 'none'; frame-ancestors 'none'")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        # Do not record filenames or URLs in access logs.
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        return self.headers.get('Host', '') in ('127.0.0.1:%d' % self.server.server_port, 'localhost:%d' % self.server.server_port)

    def do_POST(self):
        origin = self.headers.get('Origin', '')
        if (not self.valid_host() or self.headers.get('X-MedPose-Action') != 'install'
                or origin != 'http://' + self.headers.get('Host', '')
                or self.headers.get('Content-Length', '0') != '0'
                or self.headers.get('Transfer-Encoding')):
            self.send_error(403)
            return
        name = self.path.removeprefix('/api/models/')
        if not self.path.startswith('/api/models/') or name not in MODELS:
            self.send_error(404)
            return
        ok = start_install(name)
        self.send_json({'started': ok}, 202 if ok else 409)

    def do_GET(self):
        host = self.headers.get('Host', '')
        if host not in ('127.0.0.1:%d' % self.server.server_port,
                        'localhost:%d' % self.server.server_port):
            self.send_error(403)
            return
        if urlsplit(self.path).path == '/api/advanced':
            self.send_json(smpl_status())
            return
        if urlsplit(self.path).path == '/api/models':
            self.send_json(catalog())
            return
        route = urlsplit(self.path).path
        if route.startswith('/assets/') and route.endswith('.task'):
            name = route.split('/')[-1][:-5]
            if name not in MODELS or not installed(name):
                self.send_error(404, 'Model not installed or checksum failed')
                return
        path = ROUTES.get(route)
        if path is None or not path.is_file():
            self.send_error(404, 'Local asset missing; run tools/setup_pose.py')
            return
        self.send_response(200)
        kind = {'.cjs': 'text/javascript', '.mjs': 'text/javascript', '.js': 'text/javascript',
                '.wasm': 'application/wasm', '.task': 'application/octet-stream'}.get(path.suffix)
        self.send_header('Content-Type', kind or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        # The legacy standalone demo carries its own inline-script CSP.
        if urlsplit(self.path).path != '/review':
            self.send_header('Content-Security-Policy', CSP)
        self.end_headers()
        try:
            with path.open('rb') as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    missing = [str(p.relative_to(ROOT)) for p in ROUTES.values() if p.suffix != '.task' and not p.is_file()]
    if missing:
        parser.error('Missing local assets. Run .conda-envs/opencap/bin/python tools/setup_pose.py first. Missing: ' + ', '.join(missing))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = 'http://127.0.0.1:%d/' % server.server_port
    print('MedPose ready: ' + url + ' (local files only; no uploads)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
