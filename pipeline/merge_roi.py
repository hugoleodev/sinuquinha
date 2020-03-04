from .pipeline import Pipeline


class MergeRoi(Pipeline):

    def map(self, data):

        ROI = data["ROI"]
        image = data["image"]

        image[88:413, 100:660] = ROI[88:413, 100:660]

        return data
