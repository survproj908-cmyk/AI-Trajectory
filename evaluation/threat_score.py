def calculate_threat_score(
    future_points,
    object_class,
    zone
):

    score = 0

    x1, y1, x2, y2 = zone

    # --------------------------------
    # Rule 1: Predicted Intrusion
    # --------------------------------
    intrusion = False

    for px, py in future_points:

        if (
            x1 <= px <= x2
            and
            y1 <= py <= y2
        ):
            intrusion = True
            break

    if intrusion:
        score += 50

    # --------------------------------
    # Rule 2: Vehicle Bonus
    # --------------------------------
    if object_class in [
        "car",
        "truck",
        "bus",
        "motorcycle"
    ]:
        score += 20

    # --------------------------------
    # Rule 3: Near Zone Bonus
    # --------------------------------
    for px, py in future_points:

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        distance = (
            (px - cx) ** 2 +
            (py - cy) ** 2
        ) ** 0.5

        if distance < 300:
            score += 20
            break

    # --------------------------------
    # Rule 4: Moving Toward Zone
    # --------------------------------
    if len(future_points) >= 2:

        start_x, start_y = future_points[0]
        end_x, end_y = future_points[-1]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        start_dist = (
            (start_x - cx) ** 2 +
            (start_y - cy) ** 2
        ) ** 0.5

        end_dist = (
            (end_x - cx) ** 2 +
            (end_y - cy) ** 2
        ) ** 0.5

        if end_dist < start_dist:
            score += 10

    return min(score, 100)