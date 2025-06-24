import http.server
import socketserver
import threading
import time
import os
import sys

dot_path = None
last_modified = None

def get_dot_content():
    with open(dot_path, "r", encoding="utf-8") as f:
        return f.read().replace("`", "\\`")

def generate_html(dot_output="graph.html"):
    html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Live DOT Graph</title>
  <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
  <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
</head>
<body>
  <h2>Live DOT Graph</h2>
  <div id="graph">Loading...</div>
  <script>
    const viz = new Viz();

    async function fetchAndRender() {{
      try {{
        const res = await fetch('/graph.dot?' + Date.now());
        const dot = await res.text();
        const svg = await viz.renderSVGElement(dot);
        const container = document.getElementById('graph');
        container.innerHTML = '';
        container.appendChild(svg);
      }} catch (err) {{
        document.getElementById('graph').innerHTML = '<pre>' + err + '</pre>';
      }}
    }}

    fetchAndRender();
    setInterval(fetchAndRender, 50);  // Refresh every 2 seconds
  </script>
</body>
</html>"""
    with open(dot_output, "w", encoding="utf-8") as f:
        f.write(html_template)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/graph.dot"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(get_dot_content().encode("utf-8"))
        else:
            super().do_GET()

def serve_dot(dot_file, port=8000):
    global dot_path
    dot_path = dot_file
    while not os.path.exists(dot_path):
        print(f"Waiting for {dot_path} to be created...")
        time.sleep(1)
    generate_html("graph.html")

    handler = CustomHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}/graph.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python live_dot_server.py path/to/graph.dot [port]")
        sys.exit(1)

    dot_file = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    serve_dot(dot_file, port)
