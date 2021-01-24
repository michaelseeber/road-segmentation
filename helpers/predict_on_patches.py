from helpers.augment import *

def predict_on_patches5x5_with_probabilities(model, test_dataset):
    predictions = []
    for idx in range (94):
        predictions.append(tf.zeros([608, 608, 1], tf.float32))
    print("Start: predict on patch [0, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_01).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_12).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_03).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_04).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_14).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_24).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_25).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_34).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_45).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_36).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_37).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_47).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_48).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_58).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_67).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test5x5_78).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    
    return predictions
