import numpy as np


class RegionExtractor:

    def __init__(self, segmentation_mask):
        """
        segmentation_mask:
            H x W array containing class IDs predicted by BiSeNet.
        """
        self.mask = segmentation_mask

    def _mask(self, class_ids):
        """
        Create binary mask for one or more segmentation classes.
        """
        mask = np.isin(self.mask, class_ids)

        return (mask.astype(np.uint8) * 255)

    def get_skin(self):
        return self._mask([1])

    def get_nose(self):
        return self._mask([2])

    def get_glasses(self):
        return self._mask([3])

    def get_left_eye(self):
        return self._mask([4])

    def get_right_eye(self):
        return self._mask([5])

    def get_eyes(self):
        return self._mask([4, 5])

    def get_left_eyebrow(self):
        return self._mask([6])

    def get_right_eyebrow(self):
        return self._mask([7])

    def get_eyebrows(self):
        return self._mask([6, 7])

    def get_left_ear(self):
        return self._mask([8])

    def get_right_ear(self):
        return self._mask([9])

    def get_ears(self):
        return self._mask([8, 9])

    def get_mouth(self):
        return self._mask([10])

    def get_upper_lip(self):
        return self._mask([11])

    def get_lower_lip(self):
        return self._mask([12])

    def get_lips(self):
        return self._mask([11, 12])

    def get_hair(self):
        return self._mask([13])

    def get_hat(self):
        return self._mask([14])

    def get_earring(self):
        return self._mask([15])

    def get_neck_lower(self):
        return self._mask([16])

    def get_neck(self):
        return self._mask([17])

    def get_cloth(self):
        return self._mask([18])