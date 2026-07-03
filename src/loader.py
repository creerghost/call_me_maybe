import json


class Loader():

    def __init__(self, fdef_name, fcall_name) -> None:
        self.filenames = [fdef_name, fcall_name]
        self.jsons = []
        self._load_jsons()

    def _load_jsons(self) -> None:
        for filename in self.filenames:
            with open(filename, "r") as f:
                self.jsons.append(json.load(f))
