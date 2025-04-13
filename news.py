from datetime import datetime

class News:
    def __init__(self, article):
        self.summary = None
        self.article = article
        self.timestamp = datetime.now()

    def set_summary(self,summary):
        self.summary = summary
