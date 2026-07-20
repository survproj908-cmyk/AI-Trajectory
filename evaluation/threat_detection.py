def inside_zone(x, y):

    x_min = 1000
    y_min = 500

    x_max = 1500
    y_max = 1000

    return (
        x_min <= x <= x_max
        and
        y_min <= y <= y_max
    )


def check_intrusion(predicted_path):

    for x, y in predicted_path:

        if inside_zone(x, y):
            return True

    return False