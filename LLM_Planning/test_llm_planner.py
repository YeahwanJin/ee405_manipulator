#!/usr/bin/env python3
"""
LLM Planner Test Script
========================
Test the LLM task planner with custom map configuration and block state.
No robot hardware required - just tests the Gemini LLM planning.

Now with VOICE INPUT support! Run voice_server.py first.

Usage:
    # Terminal 1: Start voice server
    python3 voice_server.py
    
    # Terminal 2: Run this test
    python3 test_llm_planner.py

Requirements:
    - API_KEY environment variable set
    - pip install google-generativeai requests
"""

import os
import json
import time
import requests
import google.generativeai as genai

# Voice Server Configuration
VOICE_SERVER_URL = "https://localhost:5000"

def get_voice_input(timeout=60):
    """
    Wait for voice command from the voice server.
    Returns the command text or None if timeout/error.
    """
    print(f"\n🎤 VOICE INPUT MODE")
    print(f"   Open {VOICE_SERVER_URL} in your browser to speak.")
    print(f"   Or press Ctrl+C to type instead.")
    print(f"   Waiting {timeout}s for voice input...\n")
    
    start_time = time.time()
    poll_interval = 1.0
    
    try:
        while (time.time() - start_time) < timeout:
            try:
                # verify=False for self-signed certificate
                response = requests.get(f"{VOICE_SERVER_URL}/get_command", timeout=2, verify=False)
                data = response.json()
                
                if data.get("status") == "ok" and data.get("command"):
                    text = data["command"]["text"]
                    print(f"📥 Voice Command Received: '{text}'")
                    return text
                    
            except requests.exceptions.RequestException:
                pass
            
            # Print waiting indicator
            elapsed = int(time.time() - start_time)
            print(f"\r   ⏳ Waiting... ({elapsed}s)", end="", flush=True)
            time.sleep(poll_interval)
        
        print("\n⏰ Voice input timeout.")
        return None
        
    except KeyboardInterrupt:
        print("\n⌨️  Switching to keyboard input...")
        return None

# System Prompt for the planner
LLM_SYSTEM_PROMPT = """
You are the Task Planner for a mobile manipulator robot.
Your goal is to convert a Natural Language Instruction into a JSON sequence of atomic actions.

### INPUT DATA:
You will be provided with:
1. **Map Configuration:** Which visual marker (e.g., bird, chair, horse) is in which Zone.
2. **World State:** Which block (red, green, blue) is in which Zone.
3. **Instruction:** The task to perform.

### AVAILABLE ACTIONS:
1. `navigate(target)`: Go to 'zone_1', 'zone_2', 'zone_3', or 'start_point'.
2. `pick(target)`: Pick up 'red_block', 'green_block', or 'blue_block'.
3. `place(target)`: Place the held object at a zone ('zone_1', 'zone_2', 'zone_3').
4. `say(text)`: Speak text aloud.

### LOGIC RULES:
- **Resolve Targets:** If instruction says "zone with the Bird", look at Map Configuration to find the Zone ID.
- **Conditional Logic:** If instruction says "If there is a cube...", check the World State.
- **Switching:** A "switch" or "swap" requires a temporary place location.

### OUTPUT FORMAT:
Output ONLY a valid JSON list.
Example: [{"action": "navigate", "target": "zone_2"}, {"action": "pick", "target": "red_block"}]
"""


class LLMPlannerTest:
    def __init__(self):
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable not set!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=LLM_SYSTEM_PROMPT
        )
        print("✅ LLM Planner initialized with gemini-2.5-flash\n")
    
    def generate_plan(self, instruction, map_config, block_state):
        """Generate action plan from instruction and world state."""
        
        prompt = f"""
### 1. DYNAMIC MAP CONFIGURATION (Visual Markers):
{json.dumps(map_config, indent=2)}

### 2. CURRENT WORLD STATE (Blocks):
{json.dumps(block_state, indent=2)}

### 3. INSTRUCTION:
"{instruction}"

Generate the execution plan JSON.
"""
        
        print("=" * 60)
        print("📤 SENDING TO LLM:")
        print("=" * 60)
        print(f"Map Config:  {map_config}")
        print(f"Block State: {block_state}")
        print(f"Instruction: {instruction}")
        print("=" * 60)
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            print("\n📥 LLM RAW RESPONSE:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            
            # Clean up markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            # Parse JSON
            plan = json.loads(text)
            
            print("\n✅ PARSED PLAN:")
            print(json.dumps(plan, indent=2))
            
            return plan
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON Parse Error: {e}")
            return []
        except Exception as e:
            print(f"\n❌ LLM Error: {e}")
            return []


def interactive_test():
    """Interactive testing mode."""
    planner = LLMPlannerTest()
    
    print("\n" + "=" * 60)
    print("🧪 LLM PLANNER TEST - Interactive Mode")
    print("=" * 60)
    print("Enter your own values or press Enter for defaults.\n")
    
    while True:
        print("\n" + "-" * 40)
        
        # Get map configuration
        print("\n📍 MAP CONFIGURATION (zone -> marker)")
        zone1_marker = input("   Zone 1 marker [default: bird]: ").strip() or "bird"
        zone2_marker = input("   Zone 2 marker [default: chair]: ").strip() or "chair"
        zone3_marker = input("   Zone 3 marker [default: horse]: ").strip() or "horse"
        
        map_config = {
            "zone_1": zone1_marker,
            "zone_2": zone2_marker,
            "zone_3": zone3_marker
        }
        
        # Get block state
        print("\n📦 BLOCK STATE (block -> zone)")
        red_zone = input("   Red block zone [default: zone_1]: ").strip() or "zone_1"
        green_zone = input("   Green block zone [default: zone_2]: ").strip() or "zone_2"
        blue_zone = input("   Blue block zone [default: zone_3]: ").strip() or "zone_3"
        
        block_state = {
            "red_block": red_zone,
            "green_block": green_zone,
            "blue_block": blue_zone
        }
        
        # Get instruction
        print("\n📝 INSTRUCTION")
        instruction = input("   Task instruction: ").strip()
        
        if not instruction:
            instruction = "Place the red cube in the zone with the horse"
            print(f"   Using default: {instruction}")
        
        # Generate plan
        print("\n")
        plan = planner.generate_plan(instruction, map_config, block_state)
        
        # Continue?
        print("\n")
        again = input("🔄 Test another? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print("\n👋 Goodbye!")


def quick_test():
    """Run quick predefined tests."""
    planner = LLMPlannerTest()
    
    # Test cases
    tests = [
        {
            "instruction": "Place the red cube in the zone with the horse",
            "map_config": {"zone_1": "bird", "zone_2": "chair", "zone_3": "horse"},
            "block_state": {"red_block": "zone_1", "blue_block": "zone_3"}
        },
        {
            "instruction": "Swap the red and blue blocks",
            "map_config": {"zone_1": "bird", "zone_2": "chair", "zone_3": "horse"},
            "block_state": {"red_block": "zone_1", "blue_block": "zone_3"}
        },
        {
            "instruction": "Move all blocks to zone 2",
            "map_config": {"zone_1": "bird", "zone_2": "chair", "zone_3": "horse"},
            "block_state": {"red_block": "zone_1", "green_block": "zone_3"}
        },
    ]
    
    print("\n" + "=" * 60)
    print("🧪 LLM PLANNER TEST - Quick Tests")
    print("=" * 60)
    
    for i, test in enumerate(tests, 1):
        print(f"\n\n{'#' * 60}")
        print(f"TEST {i}: {test['instruction']}")
        print('#' * 60)
        
        planner.generate_plan(
            test['instruction'],
            test['map_config'],
            test['block_state']
        )
        
        input("\nPress Enter for next test...")


def voice_test():
    """Voice input testing mode - uses voice for task instruction."""
    planner = LLMPlannerTest()
    
    # Default map and block state
    map_config = {
        "zone_1": "bird",
        "zone_2": "chair", 
        "zone_3": "horse"
    }
    
    block_state = {
        "red_block": "zone_1",
        "green_block": "zone_2",
        "blue_block": "zone_3"
    }
    
    print("\n" + "=" * 60)
    print("🎤 LLM PLANNER TEST - Voice Mode")
    print("=" * 60)
    print("\nUsing default configuration:")
    print(f"   Map: {map_config}")
    print(f"   Blocks: {block_state}")
    print("\nYou can change these by entering custom values, or just enter for defaults.")
    
    # Allow customizing map config
    custom = input("\nCustomize map/block config? (y/n): ").strip().lower()
    if custom == 'y':
        print("\n📍 MAP CONFIGURATION")
        zone1_marker = input("   Zone 1 marker [bird]: ").strip() or "bird"
        zone2_marker = input("   Zone 2 marker [chair]: ").strip() or "chair"
        zone3_marker = input("   Zone 3 marker [horse]: ").strip() or "horse"
        map_config = {"zone_1": zone1_marker, "zone_2": zone2_marker, "zone_3": zone3_marker}
        
        print("\n📦 BLOCK STATE")
        red_zone = input("   Red block zone [zone_1]: ").strip() or "zone_1"
        green_zone = input("   Green block zone [zone_2]: ").strip() or "zone_2"
        blue_zone = input("   Blue block zone [zone_3]: ").strip() or "zone_3"
        block_state = {"red_block": red_zone, "green_block": green_zone, "blue_block": blue_zone}
    
    while True:
        print("\n" + "-" * 40)
        print("📍 Current Config:")
        print(f"   Map: {map_config}")
        print(f"   Blocks: {block_state}")
        
        # Get instruction via voice
        instruction = get_voice_input(timeout=30)
        
        # Fallback to keyboard
        if not instruction:
            instruction = input("\n📝 Type your task instruction: ").strip()
        
        if not instruction:
            instruction = "Place the red cube in the zone with the horse"
            print(f"   Using default: {instruction}")
        
        # Generate plan
        print("\n")
        plan = planner.generate_plan(instruction, map_config, block_state)
        
        # Continue?
        print("\n")
        again = input("🔄 Test another? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    print("\n🤖 LLM Planner Test Script")
    print("=" * 40)
    print("1. Interactive mode (keyboard input)")
    print("2. Quick test (predefined test cases)")
    print("3. Voice mode (speak your commands)")
    
    choice = input("\nSelect mode (1/2/3): ").strip()
    
    if choice == "2":
        quick_test()
    elif choice == "3":
        voice_test()
    else:
        interactive_test()
