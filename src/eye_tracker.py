

def get_iris_center(face_landmarks, indices, width, height):

    points = []

    for index in indices:

        landmark = face_landmarks[index]

        x = landmark.x * width
        y = landmark.y * height

        points.append((x, y))

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)

    return center_x, center_y

def get_normalized_iris_position(face_landmark, iris_indices, eye_corner_indices, width, height):
    iris_x, iris_y = get_iris_center(face_landmark, iris_indices, width, height)

    left_corner = face_landmark[eye_corner_indices[0]]
    right_corner = face_landmark[eye_corner_indices[1]]

    left_x = left_corner.x * width
    right_x = right_corner.x * width

    print(
        f"iris={iris_x:.1f}, "
        f"corner1={left_x:.1f}, "
        f"corner2={right_x:.1f}"
    )

    normalized_x = (iris_x - left_x) / (right_x - left_x)

    return normalized_x
