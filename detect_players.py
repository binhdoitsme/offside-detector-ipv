import itertools
from ctypes import c_float
from typing import NamedTuple, Optional, TypedDict

import cv2
import numpy as np

from darknet import detect, load_default_meta, load_default_net, open_yolo_img
from image_helpers import (
    draw_bounding_box,
    get_anchors,
    no_grass_mask,
    single_hist_distance,
)
from perspectives import get_vanishing_point, is_leftside_point

BoundingBox = tuple[float, ...]


class DetectionResult(NamedTuple):
    label: bytes
    confidence: float
    bounding_box: BoundingBox


class PlayerMeta(TypedDict):
    hist: tuple[cv2.Mat, ...]
    bounding_box: BoundingBox


def jersey_distance(h1: PlayerMeta, h2: PlayerMeta):
    """Histogram distance in L*A*B*"""
    hist_distances = np.array(
        tuple(
            single_hist_distance(h1["hist"][channel], h2["hist"][channel])
            for channel in range(3)
        ),
        dtype=np.float32,
    )

    return hist_distances.mean() + hist_distances.std()


def show_teams(img: cv2.Mat, players: list, player_teams: dict[int, int]):
    team_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (255, 255, 255),
    ]
    for i, (label, confidence, bounding_box) in enumerate(players):
        if label not in ("person", b"person") or confidence < 0.65:
            continue
        # print("maybe player {i} is of team {j}".format(i=i, j=player_teams[i]))
        center_x = int(bounding_box[0])
        center_y = int(bounding_box[1])
        width = int(bounding_box[2] / 2)
        height = int(bounding_box[3] / 2)
        top_left = (center_x - width, center_y - height)
        bottom_right = (center_x + width, center_y + height)
        origin = (center_x - width, center_y + height)
        text_color = (255, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(img, top_left, bottom_right, team_colors[player_teams[i]], 3)
        cv2.putText(img, f"{i}", origin, font, 2, text_color, 3)
    cv2.imshow("img", img)


class PlayerDetector:
    def __init__(self, debug=True) -> None:
        self.net = load_default_net()
        self.meta = load_default_meta()
        self.debug = debug

    def detect_players(self, img):
        return detect(self.net, self.meta, img)

    def detect_with_yolo(self, frame: np.ndarray):
        yolo_img = open_yolo_img(frame)
        res = self.detect_players(yolo_img)

        if self.debug:
            print("players detected using yolo")

        # r = (label, confidence, bounding_box)
        res = [r for r in res if r[0] in ("person", b"person")]
        res = [r for r in res if r[1] > 0.6]
        ## player width is less than 200.
        res = [r for r in res if r[2][3] < 200]

        return res


class PlayerClassifier:
    def __init__(self):
        self.player_data: list[Optional[PlayerMeta]] = []

    def analyze_player(
        self, frame: np.ndarray, bounding_box: BoundingBox
    ) -> Optional[PlayerMeta]:
        mask = no_grass_mask(frame, bounding_box)
        if (
            np.count_nonzero(mask) < 200
        ):  # not enough non-green pixels to calculate dist
            return None
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        histogram = tuple(
            cv2.calcHist([lab_frame], [channel], mask, [255], [0, 255])
            for channel in range(3)
        )
        return {"hist": histogram, "bounding_box": bounding_box}

    def analyze(self, players, frame: np.ndarray):
        self.player_data.clear()
        for i in range(len(players)):
            bb = players[i]
            bounding_box = bb[2]
            player_meta = self.analyze_player(frame, bounding_box)
            self.player_data.append(player_meta)

    # 0.49 = 0.43 (MEAN) + 0.06 (STD)
    def classify_players(self, players, frame: cv2.Mat, threshhold=0.49):
        self.analyze(players, frame)
        teams: list[list[int]] = []  # refer by index
        player_teams: dict[int, int] = {}
        for i, player in enumerate(self.player_data):
            possible_team: Optional[int] = None
            smallest_distance: Optional[float] = None
            for j, team in enumerate(teams):
                team_first_player_idx, *_ = team
                ref_player = self.player_data[team_first_player_idx]
                if player is None or ref_player is None:
                    continue
                distance = jersey_distance(player, ref_player)
                is_closer = smallest_distance is None or distance < smallest_distance
                if distance < threshhold and is_closer:
                    possible_team = j
                    smallest_distance = distance

            if possible_team is not None:
                teams[possible_team].append(i)
                player_teams[i] = possible_team

            if i not in player_teams:
                # no matching team
                player_teams[i] = len(teams)
                teams.append([i])
        return teams, player_teams

    def get_leftmost_player(
        self, players: list[DetectionResult], team: list[int], vp: tuple[int, int]
    ):
        leftmost_position: Optional[BoundingBox] = None
        for player_index in team:
            player = players[player_index]
            _, _, bounding_box = player
            if leftmost_position is None:
                leftmost_position = bounding_box
                continue
            _, _, bottom, left = get_anchors(bounding_box)
            _, _, current_bottom, current_left = get_anchors(leftmost_position)
            if is_leftside_point((left, bottom), (vp, (current_left, current_bottom))):
                leftmost_position = bounding_box
        if leftmost_position is None:
            raise ValueError("cannot find leftmost player")
        return leftmost_position


if __name__ == "__main__":

    player_detector = PlayerDetector()
    classifier = PlayerClassifier()
    img = cv2.imread("samples/4.jpg")
    img_h, img_w, channels = img.shape

    # get vanishing point
    original_vp = get_vanishing_point(img)
    original_vp = [int(x) for x in original_vp] if original_vp is not None else []
    print("vanishing point", original_vp)

    # -1 = left-right, 1 = right-left
    orientation = original_vp[0] - (img_w / 2)

    img = cv2.flip(img, 1) if orientation < 0 else img

    # get vanishing point
    vp = get_vanishing_point(img)
    # vp = tuple([int(x) for x in vp]) if vp is not None else []
    print("vanishing point", vp)

    # detect & classify players
    players: list[DetectionResult] = player_detector.detect_with_yolo(img)
    teams, player_teams = classifier.classify_players(players, img)
    print(teams, player_teams)
    show_teams(img, players, player_teams)

    # set by user
    attacking_team = 0
    defending_team = 1

    # Select GK/No GK

    """TODO: Improve left-most condition by using vanishing point"""
    # get leftmost attacking team player position (bottom-left)
    leftmost_attacking_position = classifier.get_leftmost_player(
        players, teams[attacking_team], vp
    )

    # get leftmost defending team player position excluding GK (bottom-left)
    leftmost_defending_position = classifier.get_leftmost_player(
        players, teams[defending_team], vp
    )

    # classifier.debug_show_teams(players, player_teams)
    draw_bounding_box(img, leftmost_attacking_position, (255, 0, 0))
    draw_bounding_box(img, leftmost_defending_position, (0, 255, 0))

    # draw offside lines
    attacking_anchors = get_anchors(leftmost_attacking_position)
    attacking_bottom_left = int(attacking_anchors[3]), int(attacking_anchors[2])
    print("leftmost attacker:", attacking_bottom_left)
    defending_anchors = get_anchors(leftmost_defending_position)
    defending_bottom_left = int(defending_anchors[3]), int(defending_anchors[2])
    print("leftmost defender:", defending_bottom_left)
    attack_line = cv2.line(img, vp, attacking_bottom_left, (0, 255, 255), 3)
    defend_line = cv2.line(img, vp, defending_bottom_left, (255, 0, 255), 3)

    cv2.imshow("img", img)
    cv2.waitKey(0)

    # conclusion
    is_offside = is_leftside_point(attacking_bottom_left, (vp, defending_bottom_left))
    print("is offside?", is_offside)
