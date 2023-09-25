import http.server
import ssl

server_address = ('localhost', 8000)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# Create an SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

# Wrap the server's socket with the SSL context
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

httpd.serve_forever()
