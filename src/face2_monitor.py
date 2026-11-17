"""Compatibility shortcut for face2_api.py SESSION_ID GROUP_ARM_ID monitor."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run FACE HMI with a separate fatigue monitoring window.")
    parser.add_argument("session_id", help="Session external ID, e.g. FT19_2")
    parser.add_argument("group_arm_id", help="0 = control, 1 = FACE (same as face2_api.py)")
    args = parser.parse_args()

    # Explicitly request monitoring through the driver's optional argument.
    sys.argv = [sys.argv[0], args.session_id, args.group_arm_id, "monitor"]
    import face2_api

    face2_api.main()


if __name__ == "__main__":
    main()
