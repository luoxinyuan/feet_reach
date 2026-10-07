"""Local HTTP control and JPEG viewer; no Isaac APIs run in server threads."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Viewer:
    def __init__(self, host, port):
        self.lock = threading.Lock()
        self.jpeg = b''
        self.state = {}
        self.commands = []
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split('?')[0]
                with viewer.lock:
                    if path == '/frame.jpg':
                        body, mime = viewer.jpeg, 'image/jpeg'
                    elif path == '/state':
                        body, mime = json.dumps(viewer.state).encode(), 'application/json'
                    elif path == '/':
                        body, mime = Path(__file__).with_name('viewer.html').read_bytes(), 'text/html; charset=utf-8'
                    else:
                        self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if self.path != '/command':
                    self.send_error(404); return
                if self.headers.get('Content-Type') != 'application/json':
                    self.send_error(415); return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length < 1024: raise ValueError()
                    cmd = json.loads(self.rfile.read(length))
                    if cmd not in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-', 'reset', 'pause']:
                        raise ValueError()
                    with viewer.lock:
                        if len(viewer.commands) < 100: viewer.commands.append(cmd)
                except (ValueError, TypeError):
                    self.send_error(400); return
                self.send_response(204); self.end_headers()

        self.server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take_commands(self):
        with self.lock:
            commands, self.commands = self.commands, []
        return commands

    def publish(self, jpeg, state):
        with self.lock:
            self.jpeg, self.state = jpeg, state

    def close(self):
        self.server.shutdown()
        self.server.server_close()
