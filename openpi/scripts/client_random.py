import numpy as np
from openpi_client import websocket_client_policy


def make_random_rgb_chw(height: int = 224, width: int = 224) -> np.ndarray:
    return np.random.randint(0, 256, size=(3, height, width), dtype=np.uint8)


def main() -> None:
    # Initialize the policy client once.
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    action_horizon = 50
    action_dim = 14
    state_dim = 14

    for step in range(5):
        # Random 14-dim state.
        state = np.random.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)

        # Random action chunk with 14 dims per step.
        action = np.random.uniform(-1.0, 1.0, size=(action_horizon, action_dim)).astype(np.float32)

        # Build observation using keys expected by AlohaInputs in serving.
        observation = {
            "images": {
                "cam_high": make_random_rgb_chw(224, 224),
                "cam_low": make_random_rgb_chw(224, 224),
                "cam_left_wrist": make_random_rgb_chw(224, 224),
                "cam_right_wrist": make_random_rgb_chw(224, 224),
            },
            "state": state,
            "actions": action,
            "prompt": "test prompt",
        }

        result = client.infer(observation)
        action_chunk = result["actions"]
        print(
            f"step={step} | input_state_shape={state.shape} | "
            f"input_action_shape={action.shape} | output_action_shape={action_chunk.shape}"
        )

def save_img(save_path=f"/root/storage/CODE/txy/dsrl_pi0_lx/test/{current_time}", img_key="all"):
    img_key_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    '''
    
    这里我可以根据img_key来选择保存哪张图，默认都保存。'''
    pass

if __name__ == "__main__":
    main()