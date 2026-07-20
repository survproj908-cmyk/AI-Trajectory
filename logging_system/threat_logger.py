import csv
import os
from datetime import datetime

LOG_FILE = "results/threat_log.csv"


def log_threat(
    track_id,
    object_class,
    score
):

    file_exists = os.path.isfile(
        LOG_FILE
    )

    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:

            writer.writerow(
                [
                    "Timestamp",
                    "Object ID",
                    "Object Type",
                    "Threat Score"
                ]
            )

        writer.writerow(
            [
                datetime.now(),
                track_id,
                object_class,
                score
            ]
        )

    print(
        "Threat Logged!"
    )