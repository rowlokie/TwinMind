"""
CLI Entrypoint to run the Mirror Agent interactively.
"""
import os
import sys
import argparse

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mirror.engine import MirrorEngine

def main():
    parser = argparse.ArgumentParser(description="Run MirrorGPT interactive conversation CLI")
    parser.add_argument("-n", "--mirror-name", type=str, default="Piyush", help="Subject name of the mirror agent")
    parser.add_argument("-p", "--preset", type=str, default="piyush", choices=["piyush", "thomas"], help="Persona preset")
    parser.add_argument("-m", "--model", type=str, default="demo", choices=["demo", "openai", "gemini"], help="Model provider")
    parser.add_argument("-k", "--api-key", type=str, default=None, help="API key for selected provider")
    parser.add_argument("-g", "--greeting", type=str, default="Who are you and what do you do?", help="Initial inquiry")
    
    args = parser.parse_args()

    engine = MirrorEngine(subject_name=args.mirror_name)
    engine.load_preset(args.preset)

    print("\n" + "="*60)
    print(f"🪞 MirrorGPT CLI Session: {engine.subject_name} ({engine.title})")
    print(f"📚 Indexed Memories: {len(engine.facts)} facts loaded")
    print(f"⚙️ Provider: {args.model}")
    print("="*60 + "\n")

    current_prompt = args.greeting
    while True:
        print(f"\nUser: {current_prompt}")
        result = engine.generate_response(
            user_message=current_prompt,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            provider=args.model
        )
        print(f"\n🪞 {engine.subject_name}: {result['response']}")
        
        if result.get("cited_facts"):
            print(f"   [Used {len(result['cited_facts'])} personal memories]")

        try:
            current_prompt = input("\nEnter next message (or 'exit' to quit): ").strip()
            if not current_prompt or current_prompt.lower() in ["exit", "quit", "q"]:
                print("Ending MirrorGPT session. Goodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
