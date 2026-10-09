import cv2
import numpy as np


class SkinAnalyzer:

    def analyze(self, image, skin_mask):

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Get skin pixels
        skin_pixels = rgb[skin_mask > 0]

        if len(skin_pixels) == 0:
            return {
                "status": "failed",
                "reason": "No skin detected"
            }

        # Average skin color
        mean_color = np.mean(
            skin_pixels,
            axis=0
        )

        # Standard deviation
        color_std = np.std(
            skin_pixels,
            axis=0
        )

        return {
            "status": "success",

            "skin_pixel_count": int(
                len(skin_pixels)
            ),

            "mean_rgb": {
                "r": round(float(mean_color[0]), 2),
                "g": round(float(mean_color[1]), 2),
                "b": round(float(mean_color[2]), 2)
            },

            "color_variation": {
                "r": round(float(color_std[0]), 2),
                "g": round(float(color_std[1]), 2),
                "b": round(float(color_std[2]), 2)
            }
        }