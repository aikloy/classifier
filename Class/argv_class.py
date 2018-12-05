class Argv(object):
    class __Argv:
        def __init__(self):
            self._filename = None

        def set_setting(self, _filename):
            self._filename = _filename

        def get_setting(self):
            return self._filename

    instance = None

    def __new__(cls):
        if not Argv.instance:
            Argv.instance = Argv.__Argv()
        return Argv.instance
