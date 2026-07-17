import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class OnceOtlpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        print(
            f"OTLP_MOCK_RECEIVED path={self.path} content_type={self.headers.get('Content-Type')} length={len(body)}",
            flush=True,
        )
        self.send_response(200)
        self.end_headers()
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4318
    server = HTTPServer(("127.0.0.1", port), OnceOtlpHandler)
    print(f"OTLP_MOCK_LISTENING port={port}", flush=True)
    server.serve_forever()
    print("OTLP_MOCK_EXIT", flush=True)
