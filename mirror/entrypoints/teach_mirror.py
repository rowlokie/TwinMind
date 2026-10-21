"""
CLI Entrypoint to interactively teach the Mirror Agent about yourself.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mirror.engine import MirrorEngine

def main():
    parser = argparse.ArgumentParser(description="Interactively teach your Mirror Agent")
    parser.add_argument("-n", "--mirror-name", type=str, default="Piyush", help="Subject name")
    args = parser.parse_args()

    engine = MirrorEngine(subject_name=args.mirror_name)
    engine.reset_interview()

    print("\n" + "="*60)
    print(f"🎓 Teach Your Mirror Agent: {engine.subject_name}")
    print("Answer the questions below. Your answers will be automatically indexed.")
    print("="*60 + "\n")

    for i, q in enumerate(engine.interview_questions):
        print(f"\n[Question {i+1}/{len(engine.interview_questions)}]: {q}")
        try:
            ans = input("Your answer: ").strip()
            if ans:
                engine.process_interview_answer(ans)
                print(f"✅ Recorded. Total facts in memory: {len(engine.facts)}")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            break

    print(f"\n🎉 Teaching complete! {len(engine.facts)} facts are now ready in memory.")
    print("Start chatting by running: python mirror/entrypoints/run_mirror.py")

if __name__ == "__main__":
    main()
