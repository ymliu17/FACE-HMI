import os
import argparse
import csv
from face2_script import get_block_fatigue


def infer_all_blocks(session_dir, model_path):
    session_dir = os.path.abspath(session_dir)
    block_items = sorted(os.listdir(session_dir))

    results = []
    print(f"\n>>> Running fatigue inference on: {session_dir}\n")

    for item in block_items:
        item_path = os.path.join(session_dir, item)

        # 只处理 block-xxxxx*
        if not item.startswith("block-"):
            continue

        block_id = item.split("-")[-1]

        # 情况 A: 现成的人脸目录 block-xxxxx/
        face_dir = os.path.join(session_dir, f"block-{block_id}")
        # 情况 B: 视频文件 block-xxxxx.mp4
        video_path = os.path.join(session_dir, f"block-{block_id}.mp4")

        if os.path.isdir(face_dir):
            # 已有人脸，跳过抽帧
            try:
                fatigue_score = get_block_fatigue(face_dir, face_dir, model_path)
                print(f"[OK] {item} (faces)    → fatigue={fatigue_score:.4f}")
                results.append((block_id, fatigue_score))
            except Exception as e:
                print(f"[ERROR] Failed on {item}: {e}")

        elif os.path.isfile(video_path):
            # 无人脸目录 → 自动抽帧
            try:
                os.makedirs(face_dir, exist_ok=True)
                fatigue_score = get_block_fatigue(video_path, face_dir, model_path)
                print(f"[OK] {item} (video)    → fatigue={fatigue_score:.4f}")
                results.append((block_id, fatigue_score))
            except Exception as e:
                print(f"[ERROR] Failed on {item}: {e}")

    return results


def save_results(session_dir, results):
    csv_path = os.path.join(session_dir, "fatigue_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "fatigue_score"])
        writer.writerows(results)
    print(f"\n>>> Results saved to: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Fatigue inference for all blocks in a session.")
    parser.add_argument("--session_dir", type=str, required=True, help="Path to session folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model (.pth)")
    args = parser.parse_args()

    results = infer_all_blocks(args.session_dir, args.model_path)
    save_results(args.session_dir, results)


if __name__ == "__main__":
    main()
