from landmarks.regions import FaceRegions


regions = FaceRegions()


print("Left eye landmarks:")
print(regions.get_left_eye())

print("\nRight eye landmarks:")
print(regions.get_right_eye())

print("\nIris landmarks:")
print(regions.get_iris())

print("\nMouth landmarks:")
print(regions.get_mouth())

print("\nFace contour landmarks:")
print(regions.get_face_contour())