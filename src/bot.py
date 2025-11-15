from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot
from rlbot_flatbuffers import DesiredBallState, DesiredCarState, DesiredPhysics, MatchPhase, RotatorPartial, Vector3Partial, Vector3, Rotator
import numpy as np


def vec3_to_np(vector: Vector3) -> np.ndarray:
    return np.array([vector.x, vector.y, vector.z])


def rot_to_np(rotator: Rotator) -> np.ndarray:
    cp, sp = np.cos(rotator.pitch), np.sin(rotator.pitch)
    cy, sy = np.cos(rotator.yaw), np.sin(rotator.yaw)
    cr, sr = np.cos(rotator.roll), np.sin(rotator.roll)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


def np_to_rot(matrix: np.ndarray) -> RotatorPartial:
    pitch = np.arcsin(-matrix[2, 0])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    return RotatorPartial(pitch=float(pitch), yaw=float(yaw), roll=float(roll))


def np_to_vec3(array: np.ndarray) -> Vector3Partial:
    return Vector3Partial(x=float(array[0]), y=float(array[1]), z=float(array[2]))


mirror_x = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
mirror_y = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
])
mirror_xy = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
])
mirror_matrices = [mirror_x, mirror_y, mirror_xy]


class MirrorBot(Bot):

    def get_output(self, packet: GamePacket) -> ControllerState:

        if packet.match_info.match_phase in [
            MatchPhase.Inactive,
            MatchPhase.Paused,
            MatchPhase.Replay,
            MatchPhase.Ended,
            MatchPhase.Countdown,
        ]:
            return ControllerState()

        target_index = None
        impossible_ball = False

        # if it's a 1v1 against a mirror bot
        if len(packet.players) == 2 and self.team != packet.players[1 - self.index].team:
            target_index = 1 - self.index
            mirror_matrix = mirror_y
            impossible_ball = True

        else:
            # find a car on own team that is not a mirror bot
            # and also find our index among team mirror bots
            team_mirror_index_counter = -1
            for i, car in enumerate(packet.players):
                if car.team == self.team:
                    if "mirror" in car.name.lower() and car.is_bot:
                        team_mirror_index_counter += 1
                        if i == self.index:
                            mirror_matrix = mirror_matrices[team_mirror_index_counter]
                    else:
                        target_index = i

            if target_index is None:
                return ControllerState()

        target_car = packet.players[target_index]

        # Convert to numpy
        target_location = vec3_to_np(target_car.physics.location)
        target_rotation = rot_to_np(target_car.physics.rotation)
        target_velocity = vec3_to_np(target_car.physics.velocity)
        target_angular_velocity = vec3_to_np(target_car.physics.angular_velocity)

        # Apply mirroring
        mirrored_location = mirror_matrix @ target_location
        mirrored_rotation = mirror_matrix @ target_rotation
        mirrored_velocity = mirror_matrix @ target_velocity
        mirrored_angular_velocity = mirror_matrix @ target_angular_velocity

        # Angular velocity needs sign flip for non-xy mirrors
        if mirror_matrix is not mirror_xy:
            mirrored_angular_velocity *= -1

        if impossible_ball:
            ball_state = DesiredBallState(DesiredPhysics(
                location=Vector3Partial(y=0),
                velocity=Vector3Partial(y=0),
            ))

        car_state = DesiredCarState(DesiredPhysics(
            location=np_to_vec3(mirrored_location),
            rotation=np_to_rot(mirrored_rotation),
            velocity=np_to_vec3(mirrored_velocity),
            angular_velocity=np_to_vec3(mirrored_angular_velocity),
        ), boost_amount=target_car.boost)

        if mirror_matrix is not mirror_xy:
            car_state.physics.rotation.roll *= -1 # type: ignore

        self.set_game_state(
            cars={self.index: car_state},
            balls={0: ball_state} if impossible_ball else {},
        )

        target_controls = packet.players[target_index].last_input
        if mirror_matrix is not mirror_xy:
            target_controls.steer *= -1
            target_controls.yaw *= -1
            target_controls.roll *= -1

        return target_controls


if __name__ == "__main__":
    MirrorBot("darxeal/mirror-bot").run()
