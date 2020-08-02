import time
import sys
import urllib
import random
import datetime
import stomp

# install package stomp.py
class Listener():
    def on_message(self, headers, message):
        print("[MSG]", message)

    def on_error(self, headers, message):
        print("[ERR]", message)
        stomp.PrintingListener()
class MessagingServer():
    def __init__(self, listener):
        self.listener = listener
        host_and_ports = [('localhost', 61613)]
        self.conn = stomp.Connection(host_and_ports=host_and_ports)
        
    def open_network(self):
        self.conn.set_listener("", self.listener)
        self.conn.connect(username='admin', passcode='password', wait=True)
        self.conn.subscribe(destination='/topic/FROM_UNITY', id = 123)
        self.conn.auto_content_length = False

    def close_network(self):
        self.conn.disconnect()

    def send_JSON(self, msg):
        self.conn.send(body=msg, destination='/topic/UNITY_JSON')

    def send_msg(self, prefix, msg):
        msg = prefix + " " + urllib.quote_plus(msg)
        #print(msg)
        self.conn.send(body=msg, headers=self.headers, destination='/topic/DEFAULT_SCOPE')