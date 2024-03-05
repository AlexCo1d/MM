from model.architecture import MM


class MyVQAModel(MM):
    def __init__(self):
        super(MyVQAModel, self).__init__()
        # TODO: adjust the resolution: Patch Embed and position embed.