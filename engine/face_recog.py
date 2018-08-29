import face_recognition
import time
import pickle


class FaceRecog:

    def __init__(self, path=None):
        if path is None:
            self.trained_NN_path = 'weight/face_rec/EVTA1T_trained_knn_model.clf'
        else:
            self.trained_NN_path = path

    def recognition_image_from_ndarray(self, X_img):
        time_before_prediction = time.time()
        predictions = predict(X_img, model_path=self.trained_NN_path)
        print('Prediction time:', time.time() - time_before_prediction)
        print('###Predicitons:', predictions)
        output = {
            'status': 'Ok',
            'faces': []
        }
        for name, (top, right, bottom, left) in predictions:
            output['faces'].append({
                'id': name,
                'position': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left,
                }
            })

        return output

    def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):
        """
        Recognizes faces in given image using a trained KNN classifier

        :param X_img_path: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
            of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """
        # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        #     raise Exception("Invalid image path: {}".format(X_img_path))

        if knn_clf is None and model_path is None:
            raise Exception(
                "Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        # X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(
            X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <=
                       distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
