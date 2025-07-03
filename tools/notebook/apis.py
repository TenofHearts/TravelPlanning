class Notebook:
    def __init__(self):
        self.note = ""

    def write(self, description: str, content: str):
        self.note += description.strip() + "\n"
        self.note += content.strip() + "\n"
        return "NoteBook updated."

    def read(self):
        return self.note

    def reset(self):
        self.note = ""
