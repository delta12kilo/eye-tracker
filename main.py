import cv2

from src.eye_tracker import get_iris_center, get_normalized_iris_position
from src.camera import Camera
from src.face_tracker import FaceTracker


def draw_landmarks(frame, result):
    if not result.face_landmarks:
        return None

    height, width, _ = frame.shape

    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # cv2.circle(frame, (x,y), 1, (0, 255, 0), -1)

        #  Iris Landmark
        left_iris_indices = [474, 475, 476, 477]
        right_iris_indices = [469, 470, 472, 472]

        for index in left_iris_indices:
            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

        for index in right_iris_indices:
            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

        left_x, left_y = get_iris_center(
            face_landmarks,
            left_iris_indices,
            width,
            height
        )

        right_x, right_y = get_iris_center(
            face_landmarks,
            right_iris_indices,
            width,
            height
        )

        #Eye corner

        left_eye_corner = [33, 133]
        right_eye_corner = [362, 263]

        for idx in left_eye_corner:
            landmark = face_landmarks[idx]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x,y), 6, (255, 0, 0), -1)

        for idx in right_eye_corner:
            landmark = face_landmarks[idx]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        left_position = get_normalized_iris_position(
            face_landmarks,
            left_iris_indices,
            left_eye_corner,
            width,
            height
        )

        right_position = get_normalized_iris_position(
            face_landmarks,
            right_iris_indices,
            right_eye_corner,
            width,
            height
        )

        # print(
        #     f"Left: {left_position:.3f}, "
        #     f"Right: {right_position:.3f}"
        # )

        return (left_x, left_y), (right_x, right_y)

    return None


def main():
    camera = Camera()
    face_tracker = FaceTracker()

    print(f"Camera resolution: {camera.width}x{camera.height}")
    print("Press Q to exit.")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Fail to read.")
                break

            result = face_tracker.process(frame)
            if result.face_landmarks:
                cv2.putText(
                    frame,"Face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2
                )

                iris_centers = draw_landmarks(frame, result)

                # if iris_centers:
                #     left_iris, right_iris = iris_centers
                #
                #     print(
                #         f"Left: {left_iris}, "
                #         f"Right: {right_iris}"
                #     )
                #

            cv2.imshow("Eye Mouse", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    finally:
        face_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()