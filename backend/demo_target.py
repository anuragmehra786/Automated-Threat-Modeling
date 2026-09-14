from http.server import BaseHTTPRequestHandler, HTTPServer


class DemoTargetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        pages = {
            "/": """
                <html>
                <head><title>Acme Customer Portal</title></head>
                <body>
                    <h1>Acme Customer Portal</h1>
                    <a href="/login">Login</a>
                    <a href="/admin">Admin</a>
                    <a href="/profile">Profile</a>
                    <a href="/checkout">Checkout</a>
                </body>
                </html>
            """,
            "/login": """
                <html>
                <head><title>Login</title></head>
                <body>
                    <h1>Login</h1>
                    <form method="POST" action="/login">
                        <input name="username">
                        <input name="password" type="password">
                        <button type="submit">Login</button>
                    </form>
                </body>
                </html>
            """,
            "/admin": """
                <html>
                <head><title>Admin Control Panel</title></head>
                <body>
                    <h1>Admin Control Panel</h1>
                </body>
                </html>
            """,
            "/profile": """
                <html>
                <head><title>User Profile</title></head>
                <body>
                    <h1>User Profile</h1>
                </body>
                </html>
            """,
            "/checkout": """
                <html>
                <head><title>Checkout</title></head>
                <body>
                    <h1>Checkout</h1>
                    <form method="POST" action="/checkout">
                        <input name="card_number">
                        <button type="submit">Pay</button>
                    </form>
                </body>
                </html>
            """,
        }

        body = pages.get(
            self.path,
            "<html><head><title>Not Found</title></head><body>404</body></html>",
        )

        self.send_response(200 if self.path in pages else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")

        # Deliberately simple demo session cookie for defensive analysis.
        self.send_header("Set-Cookie", "session=demo-session; SameSite=Lax")

        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body>Demo POST received.</body></html>")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9000), DemoTargetHandler)
    print("Demo target running at http://127.0.0.1:9000")
    print("Press Ctrl+C to stop.")
    server.serve_forever()