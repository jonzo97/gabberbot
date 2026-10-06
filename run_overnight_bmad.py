#!/usr/bin/env python3
"""
BMAD Overnight Production Runner
Runs continuous BMAD agent sessions for hardcore music production
"""

import time
import json
from datetime import datetime
from pathlib import Path
from overnight_agent_coordinator import BMADAgentCoordinator

def run_continuous_bmad_production():
    """Run continuous BMAD agent production sessions"""
    print("BMAD OVERNIGHT HARDCORE PRODUCTION STARTING")
    print("=" * 60)
    
    session_count = 0
    total_outputs = {
        "sessions": 0,
        "patterns": 0, 
        "tracks": 0,
        "presets": 0
    }
    
    while True:
        try:
            session_count += 1
            print(f"\nSTARTING SESSION #{session_count}")
            print(f"TIME: {datetime.now().strftime('%H:%M:%S')}")
            
            # Create new coordinator for each session
            coordinator = BMADAgentCoordinator()
            
            # Run full agent session
            summary = coordinator.run_overnight_session()
            
            if summary.get("success"):
                # Update totals
                total_outputs["sessions"] += 1
                total_outputs["patterns"] += summary["outputs"]["generated_patterns"]
                total_outputs["tracks"] += summary["outputs"]["mixed_tracks"]
                total_outputs["presets"] += 20  # Estimated presets per session
                
                print(f"\nSESSION #{session_count} COMPLETE!")
                print(f"TOTAL OUTPUTS SO FAR:")
                print(f"   Sessions: {total_outputs['sessions']}")
                print(f"   Patterns: {total_outputs['patterns']}")
                print(f"   Tracks: {total_outputs['tracks']}")
                print(f"   Presets: {total_outputs['presets']}")
                
                # Save running totals
                totals_file = Path("overnight_production_totals.json")
                with open(totals_file, 'w') as f:
                    json.dump({
                        "last_updated": datetime.now().isoformat(),
                        "total_sessions": total_outputs["sessions"],
                        "total_outputs": total_outputs,
                        "current_session": session_count,
                        "status": "running"
                    }, f, indent=2)
                
                # Brief pause between sessions (adjust as needed)
                print(f"\nPausing 10 seconds before next session...")
                time.sleep(10)
                
            else:
                print(f"SESSION #{session_count} FAILED: {summary.get('error', 'Unknown error')}")
                print("Waiting 30 seconds before retry...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\nBMAD PRODUCTION STOPPED BY USER")
            print(f"FINAL TOTALS:")
            print(f"   Sessions Completed: {total_outputs['sessions']}")
            print(f"   Total Patterns: {total_outputs['patterns']}")
            print(f"   Total Tracks: {total_outputs['tracks']}")
            print(f"   Total Presets: {total_outputs['presets']}")
            break
        except Exception as e:
            print(f"ERROR IN SESSION #{session_count}: {e}")
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

if __name__ == "__main__":
    run_continuous_bmad_production()