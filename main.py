import os
from tkinter import Tk
from tkinter.messagebox import showerror, showinfo
from tkinter.simpledialog import askinteger
from typing import Optional

import cv2
from perspectives import get_vanishing_point

from detect_field import get_field_mask
from detect_players import PlayerClassifier, PlayerDetector
from detect_players import get_anchors, is_leftside_point, show_teams


def coalesce(*values: Optional[int]):
    for value in values:
        if value is not None:
            return value
    return -1


def get_field_vp(img: cv2.Mat, flip_horizontal=False):
    """
    Get vanishing point of field lines only.
    Flip original img if necessary to have the attacking direction of right -> left.
    """
    if flip_horizontal:
        img = cv2.flip(img, 1, img)
    field_mask = get_field_mask(img)
    inside_field = cv2.bitwise_and(img, img, mask=field_mask)
    vp = get_vanishing_point(inside_field)
    # check if the vanishing point is in the right side
    if vp[0] < img.shape[0] / 2:
        return get_field_vp(img, flip_horizontal=True)
    return vp


def main():
    root = Tk()

    base_folder = "samples"
    sample_files = os.listdir(base_folder)

    player_detector = PlayerDetector()
    classifier = PlayerClassifier()

    for filename in ["42.jpg", "37.jpg", "18.jpg"]:
        img = cv2.imread(os.path.join(base_folder, filename))
        vp = get_field_vp(img)
        players = player_detector.detect_with_yolo(img)
        teams, player_teams = classifier.classify_players(players, img)
        show_teams(img, players, player_teams)
        attack_team_index = askinteger(
            "Choose attacking team",
            "Which team is attacking team?"
            "(blue = 0, green = 1, red = 2, yellow = 3, aqua = 4)",
        )
        defend_team_index = askinteger(
            "Choose defending team",
            "Which team is defending team?"
            "(blue = 0, green = 1, red = 2, yellow = 3, aqua = 4)",
        )
        attack_team = teams[coalesce(attack_team_index)]
        defend_team = teams[coalesce(defend_team_index)]
        leftmost_attacker = classifier.get_leftmost_player(players, attack_team, vp)
        leftmost_defender = classifier.get_leftmost_player(players, defend_team, vp)

        attacking_anchors = get_anchors(leftmost_attacker)
        attacking_bottom_left = int(attacking_anchors[3]), int(attacking_anchors[2])
        print("leftmost attacker:", attacking_bottom_left)
        defending_anchors = get_anchors(leftmost_defender)
        defending_bottom_left = int(defending_anchors[3]), int(defending_anchors[2])
        print("leftmost defender:", defending_bottom_left)
        result = img.copy()
        cv2.line(result, vp, attacking_bottom_left, (0, 255, 255), 3)
        cv2.line(result, vp, defending_bottom_left, (255, 0, 255), 3)

        is_offside = is_leftside_point(
            attacking_bottom_left, (vp, defending_bottom_left)
        )

        cv2.imshow("Offside detection result", result)
        if is_offside:
            showerror("Possible Offside result", "Possible offside!")
        else:
            showinfo("Possible Offside result", "No offside detected")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
