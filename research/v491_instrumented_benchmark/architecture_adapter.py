
from architecture_core import Architecture

class Adapter:
    name="architecture"
    def __init__(self):
        self.engine=Architecture()
    def reset(self):
        self.engine.reset()
    def respond(self,text):
        return self.engine.respond(text)

def create_adapter(name="architecture"):
    return Adapter()
