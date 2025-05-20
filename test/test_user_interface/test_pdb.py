from unittest import skip

import pdbpp as pdb
import readline
import rlcompleter


class MyPdb(pdb.Pdb):

    readline.parse_and_bind("tab: complete")

    def onecmd(self, line):
        print(f"[Captured Input]: {line.strip()}")
        return super().onecmd(line)


class A:

    def __init__(self, a, b=10):
        self.x = 10
        self.y = 20
        self.z = 30
        self.a = a
        self.b = b

    def a_method(self):
        print("Hello from a_method")


def foo():
    x = 3
    ahmed = 20
    debugger = MyPdb()
    debugger.set_trace()
    print(x)

# foo()
