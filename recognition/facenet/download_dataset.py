from roboflow import Roboflow
from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    args, _ = parser.parse_known_args()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str)
    args = parser.parse_args()
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("melnikum").project("my-photo-search-2")
    version = project.version(4)
    dataset = version.download("folder")