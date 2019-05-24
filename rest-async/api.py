from zmq.eventloop import zmqstream, ioloop
import tornado.web
from tornado import websocket


class MyBroadcastWebsocket(websocket.WebSocketHandler):
    clients = set()

    def open(self):
        self.clients.add(self)

    @classmethod
    def broadcast_zmq_message(cls, msg):
        for client in cls.clients:
            client.write_message('Message:' + str(msg))  # Send the message to all connected clients

    def on_close(self):
        self.clients.remove(self)


def run():
    ioloop.install()
    my_stream = zmqstream.ZMQStream(8080)  # i.e. a pull socket
    my_stream.on_recv(MyBroadcastWebsocket.broadcast_zmq_message)  # call this callback whenever there's a message


if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/websocket", MyBroadcastWebsocket),
    ])
    application.listen(8888)
    ioloop.IOLoop.instance()
